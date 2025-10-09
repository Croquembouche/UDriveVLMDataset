#!/usr/bin/env python3
"""
Utility helpers for packaging the local driving-scene dataset into a JSONL
format that works with ms-swift when fine-tuning Qwen2-VL style models.

The script pairs each image under ``<dataset-dir>/images`` with its analysis
JSON file from ``<dataset-dir>/labels`` and generates conversational samples
where the model is asked to provide a structured scene assessment.
"""

# Script overview:
# - Parse CLI arguments that decide which split to process, where to write the JSONL,
#   whether to use absolute or relative image paths, and what conversational prompts to use.
# - Load defaults from a YAML config file so the script can run without long command-line
#   invocations while still allowing light overrides through optional flags. Debug mode
#   emits extra diagnostics plus previews of the first few image/label pairs. Each
#   assistant response contains both a natural-language summary and the original label
#   JSON so the supervision covers structured and free-form outputs.
# - Build a lookup table that maps each label JSON to the dataset folder name and image
#   filename, catching duplicate mappings up front so downstream samples stay consistent.
# - Walk the image tree while filtering for known image extensions, yielding the dataset
#   source name and the image path for every candidate frame.
# - For every image, locate the companion label JSON, load it, and convert the structured
#   metadata into natural language sentences that align with the Overall.json vocabulary.
# - Assemble ms-swift friendly conversation records containing system, user, and assistant
#   messages plus the associated image path; skip items with missing labels or empty content.
# - Stream the records into the target JSONL file so the output stays large-dataset friendly,
#   and summarize how many samples were written along with any skipped image paths.

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections.abc import Iterable as IterableABC
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    yaml = None

# Valid image extensions to include in the output dataset.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Message templates used for each training sample.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that provides detailed traffic scene analyses."
)
DEFAULT_USER_PROMPT = (
    "Drive Scene Analyzer is tailored to deliver highly accurate and consistent driving "
    "scene analyses. Every new image is considered a completely independent scene, which "
    "allows for a fresh perspective in every analysis. Each new analysis must ignore "
    "previous images, previous results, and conversation history. The analyzer must only "
    "analyze the most recent uploaded scene. It is crucial that the analysis is not "
    "affected by previous analysis. It rigorously double-checks its evaluations with "
    "visual verification to ensure the highest level of reliability, focusing on the "
    "status of other vehicles and the maneuvers of the ego vehicle. It uses logical "
    "reasoning to determine the ego vehicle and other vehicles' direction and maneuver. "
    "The analysis result must follow the Overall.json structure. The analyzer calculates "
    "a severity score and adheres to the Overall.json schema, enhancing the clarity and "
    "utility of its assessments. Additionally, the severity score must take potential "
    "risks into account. Logical thinking must be used to assess the what-ifs in the "
    "scene. The what-ifs analysis must be reflected in the severity score calculation. "
    "The what-ifs are not limited to pedestrians and are extended to all items identified "
    "in the scene. Additionally, the severity score should also consider the proximity of "
    "the identified items to the ego vehicle. Additionally, the severity score must also "
    "consider the proximity of objects and potential risks to calculate a comprehensive "
    "severity core. Additionally, the analyzer must reassess the severity score based on "
    "proximity and what-ifs. Outputs are presented in a concise JSON format, emphasizing "
    "essential details without redundant information. Additionally, the terminology of "
    "the analysis must come from the corresponding field in Overall.json. It's crucial "
    "that the analysis excludes the TrafficLightState if no traffic lights are present in "
    "the scene, maintaining relevance and precision. A reassessment step is included to "
    "bolster accuracy and eliminate potential inconsistencies. Additionally, it must "
    "visually confirm that the analysis matches the current scene, ensuring the validity "
    "of its assessments. Do not show any intermediate steps in your analysis. Don't "
    "explain the analysis unless explicitly asked by the user. Additionally, the "
    "directionality of the lane needs to be determined based on the lane markings and "
    "where the vehicles are facing. Additionally, special lanes and traffic signs must be "
    "visually verified and can not be assumed. Do not confuse billboards with traffic "
    "signs. Additionally, the analysis should not contain any contradictions with one "
    "another. Always reassess the analysis and visually confirm the analysis with the "
    "scene. Additionally, you must use logical thinking to determine the direction and "
    "the maneuver of the ego vehicle. You can obtain this information by comparing the "
    "ego vehicle's orientation with the road and other identified items in the scene. The "
    "analyzer must incorporate potential risks and what-if scenarios when calculating the "
    "severity score. It is crucial to visually confirm the analysis before reporting the "
    "analysis. The analyzer must also assess the potential risks and what-if scenarios "
    "when calculating the severity score using logical reasoning and a chain of thought. "
    "The analyzer must visually confirm the analysis and ensure that identified features "
    "in the output JSON are included in the current scene. Do not include the 'conflicting "
    "lane markings' field if there are no conflicting lane markings. Analyze the scene "
    "and pay close attention to the lane markings and directionality of the road. Ensure "
    "that the road is accurately identified as one-way or two-way based on visible "
    "markings like dashed or solid lines. Double-check for potential mistakes in "
    "identifying the directionality before finalizing the output. Analyze the driving "
    "scene in full detail. Capture all vehicles, including their type, motion state "
    "(moving, parked, or blocking parts of the road), and their influence on the ego "
    "vehicle's maneuver. Identify road markings, lane directionality, special lanes, and "
    "any visible traffic signs. Assess potential risks like vehicles obstructing the "
    "road, pedestrians, or other hazards. Ensure lane markings and road conditions are "
    "accurately described, and evaluate the ego vehicle’s direction and maneuver. "
    "Calculate a severity score considering proximity to obstacles, what-if scenarios, "
    "and potential risks in the scene. Provide a complete JSON output based on the "
    "analysis. If there is a traffic light in the scene, then it must be included in the "
    "traffic sign field, and it must have a TrafficLightState field. The analyzer must "
    "strictly follow the data structure and terminology in Overall.json. The analyzer "
    "must double check every single detail in the uploaded scene."
)

DEFAULT_CONFIG_PATH = Path("configs") / "qwen_dataset.yaml"
DEFAULT_CONFIG = {
    "dataset_dir": Path("dataset") / "train",
    "output": Path("dataset") / "train_qwen.jsonl",
    "relative_paths": False,
    "limit": None,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "user_prompt": DEFAULT_USER_PROMPT,
    "debug": False,
    "append_label_json": True,
}

CAMEL_CASE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Specific tokens that benefit from more natural phrasing.
SPECIAL_CASES = {
    "False": "no",
    "True": "yes",
    "Few": "a few",
    "MultipleVehicles": "multiple vehicles",
    "NoVehicle": "no vehicles",
    "LaneVisible": "visible",
    "Lane Not Clearly Visible": "not clearly visible",
    "MultipleLanes": "multiple lanes",
    "SingleLane": "a single lane",
    "NoSpecialLanes": "no special lanes",
    "NoTrafficSigns": "no traffic signs",
    "Not Applicable": "not applicable",
    "SignNotVisible": "not visible",
    "NoPed": "no pedestrians",
    "MultiplePed": "multiple pedestrians",
    "Ped Crossing": "pedestrians crossing",
    "Ped Waiting": "pedestrians waiting",
    "NoImpairments": "no specific impairments",
    "InQueue": "queued",
    "In Queue": "queued",
    "Camera glare": "camera glare",
    "CAMERA_GLARE": "camera glare",
    "Shadows": "shadows",
    "SUV": "SUV",
    "HOV Lane": "HOV lane",
    "Dry": "dry",
    "Wet": "wet",
    "Snowy": "snowy",
    "Clear": "clear",
    "Forward": "forward",
    "Two-Way": "two-way",
    "One-Way": "one-way",
}


def coerce_bool(value: Any) -> bool:
    """Convert truthy values that may be represented as strings or numbers."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def coerce_optional_int(value: Any) -> Optional[int]:
    """Convert optional integer configuration fields."""
    if value in {None, "", "null"}:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return int(stripped)
    raise ValueError(f"Cannot convert {value!r} to an integer.")


def debug_log(enabled: bool, message: str) -> None:
    """Print debug messages when enabled."""
    if enabled:
        print(f"[DEBUG] {message}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Configure CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert the local driving dataset into an ms-swift friendly JSONL "
            "for fine-tuning Qwen2-VL style models."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file that describes dataset settings.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional override for the sample cap defined in the config.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error console output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging (overrides config value).",
    )
    return parser.parse_args(argv)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load dataset conversion settings from a YAML configuration file."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to parse configuration files. Install it with "
            "`pip install pyyaml`."
        )

    resolved_path = config_path.expanduser()
    if not resolved_path.is_absolute():
        resolved_path = (Path.cwd() / resolved_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, Mapping):
        raise ValueError(
            f"Config file must contain a mapping of settings: {resolved_path}"
        )

    config: Dict[str, Any] = dict(DEFAULT_CONFIG)
    config.update(data)

    base_dir = resolved_path.parent

    def resolve_path(value: Any, *, default: Path) -> Path:
        if value is None:
            candidate = default
            value_str = None
        else:
            candidate = Path(value)
            value_str = str(value)

        if candidate.is_absolute():
            return candidate

        base = base_dir if value_str and value_str.startswith(("./", "../")) else Path.cwd()
        return (base / candidate).resolve()

    config["dataset_dir"] = resolve_path(
        config.get("dataset_dir"), default=DEFAULT_CONFIG["dataset_dir"]
    )
    config["output"] = resolve_path(
        config.get("output"), default=DEFAULT_CONFIG["output"]
    )
    config["relative_paths"] = coerce_bool(config.get("relative_paths"))
    config["limit"] = coerce_optional_int(config.get("limit"))
    config["system_prompt"] = str(
        config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    )
    config["user_prompt"] = str(config.get("user_prompt", DEFAULT_USER_PROMPT))
    config["debug"] = coerce_bool(config.get("debug"))
    config["append_label_json"] = coerce_bool(
        config.get("append_label_json", DEFAULT_CONFIG["append_label_json"])
    )

    return config


def build_label_index(label_root: Path) -> Dict[Tuple[str, str], Path]:
    """
    Construct a lookup mapping (source_dataset, image_filename) to label JSON paths.
    """
    index: Dict[Tuple[str, str], Path] = {}
    for json_path in label_root.rglob("*.json"):
        if not json_path.is_file():
            continue
        try:
            relative = json_path.relative_to(label_root)
        except ValueError:
            continue
        if not relative.parts:
            continue
        dataset_name = relative.parts[0]
        image_filename = json_path.name[:-5]  # Strip trailing ".json"
        key = (dataset_name, image_filename)
        if key in index:
            # Keep the first entry encountered to avoid silent overwrites.
            raise ValueError(
                f"Duplicate label mapping detected for {dataset_name}/{image_filename}"
            )
        index[key] = json_path
    return index


def iter_image_files(image_root: Path) -> Iterator[Tuple[str, Path]]:
    """
    Iterate over image files grouped by their top-level dataset folder.
    """
    if not image_root.exists():
        raise FileNotFoundError(f"Image folder not found: {image_root}")
    for dataset_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        for image_path in sorted(dataset_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            yield dataset_dir.name, image_path


def select_article(phrase: str) -> str:
    """Return the appropriate indefinite article for the given phrase."""
    stripped = phrase.strip().lower()
    if not stripped:
        return "a"
    return "an" if stripped[0] in {"a", "e", "i", "o", "u"} else "a"


def humanize_token(value: Optional[str]) -> str:
    """
    Convert camel case or compound identifiers into human-friendly lowercase text.
    """
    if value is None:
        return ""
    value_str = str(value)
    if value_str in SPECIAL_CASES:
        return SPECIAL_CASES[value_str]
    text = CAMEL_CASE_RE.sub(" ", value_str)
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " or ")
    text = text.strip()
    if not text:
        return ""
    return text.lower()


def normalize_list(items: Optional[Iterable[Optional[str]]]) -> List[str]:
    """Apply humanize_token to each member, accepting scalar values as well."""
    if items is None:
        return []

    if isinstance(items, (str, bytes)):
        iterable = [items]
    elif isinstance(items, IterableABC):
        iterable = items
    else:
        iterable = [items]

    normalized: List[str] = []
    for item in iterable:
        text = humanize_token(item)
        if text:
            normalized.append(text)
    return normalized


def join_phrases(items: Sequence[str]) -> str:
    """Join a list of short phrases using commas and 'and'."""
    filtered = [item for item in items if item]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 2:
        return f"{filtered[0]} and {filtered[1]}"
    return f"{', '.join(filtered[:-1])}, and {filtered[-1]}"


def format_environment(label: Mapping[str, object]) -> str:
    scene = humanize_token(label.get("Scene")) if "Scene" in label else ""
    time_of_day = humanize_token(label.get("TimeOfDay")) if "TimeOfDay" in label else ""
    weather = humanize_token(label.get("Weather")) if "Weather" in label else ""
    road = humanize_token(label.get("RoadConditions")) if "RoadConditions" in label else ""

    if not any([scene, time_of_day, weather, road]):
        return ""

    parts: List[str] = []
    intro = "It is"
    if scene and time_of_day:
        article = select_article(time_of_day)
        parts.append(f"{intro} {article} {time_of_day} {scene} scene")
    elif scene:
        article = select_article(scene)
        parts.append(f"{intro} {article} {scene} scene")
    elif time_of_day:
        article = select_article(time_of_day)
        parts.append(f"{intro} {article} {time_of_day} setting")
    else:
        parts.append("The environment is described as")

    if weather:
        conjunction = " with"
        parts[-1] += f"{conjunction} {weather} weather"
    if road:
        if weather:
            parts[-1] += f" and {road} road conditions"
        else:
            parts[-1] += f" with {road} road conditions"

    if not parts[-1].endswith("."):
        parts[-1] += "."
    return " ".join(parts)


def format_lane_information(label: Mapping[str, object]) -> str:
    info = label.get("LaneInformation")
    if not isinstance(info, Mapping):
        return ""

    statements: List[str] = []
    number = humanize_token(info.get("NumberOfLanes")) if "NumberOfLanes" in info else ""
    if number:
        noun = number if "lane" in number else f"{number} lanes"
        statements.append(f"The roadway has {noun}.")

    lane_markings = humanize_token(info.get("LaneMarkings")) if "LaneMarkings" in info else ""
    if lane_markings:
        statements.append(f"Lane markings are {lane_markings}.")

    specials_raw = normalize_list(info.get("SpecialLanes"))
    specials = [item for item in specials_raw if item != "no special lanes"]
    if specials:
        statements.append(
            f"Special lanes include {join_phrases(specials)}."
        )
    elif specials_raw and not specials:
        statements.append("There are no special lanes.")

    return " ".join(statements)


def format_traffic_signs(label: Mapping[str, object]) -> str:
    info = label.get("TrafficSigns")
    if not isinstance(info, Mapping):
        return ""

    statements: List[str] = []
    sign_types_raw = normalize_list(info.get("TrafficSignsTypes"))
    sign_types = [item for item in sign_types_raw if item != "no traffic signs"]
    if sign_types:
        statements.append(
            f"Traffic signs present include {join_phrases(sign_types)}."
        )
    elif sign_types_raw and not sign_types:
        statements.append("No traffic signs are observed.")

    visibility = humanize_token(info.get("TrafficSignsVisibility")) if "TrafficSignsVisibility" in info else ""
    if visibility and visibility not in {"no traffic signs", "not applicable", "not indicated"}:
        statements.append(f"The traffic signs are {visibility}.")

    vehicle_types = normalize_list(info.get("VehicleTypes"))
    if vehicle_types:
        statements.append(
            f"Relevant vehicle types mentioned are {join_phrases(vehicle_types)}."
        )

    return " ".join(statements)


def format_vehicles(label: Mapping[str, object]) -> str:
    info = label.get("Vehicles")
    if not isinstance(info, Mapping):
        return ""

    statements: List[str] = []
    total = humanize_token(info.get("TotalNumber")) if "TotalNumber" in info else ""
    if total:
        if total == "no vehicles":
            statements.append("No vehicles are present.")
        else:
            statements.append(f"There are {total} in the scene.")

    states = normalize_list(info.get("States"))
    if states:
        statements.append(f"Vehicle states include {join_phrases(states)}.")

    motion_flags = {humanize_token(item) for item in normalize_list(info.get("InMotion"))}
    if "yes" in motion_flags:
        statements.append("Some vehicles are moving.")
    elif "no" in motion_flags and total != "no vehicles":
        statements.append("Vehicles appear stationary.")

    return " ".join(statements)


def format_pedestrians(label: Mapping[str, object]) -> str:
    pedestrians = normalize_list(label.get("Pedestrians"))
    if not pedestrians:
        return ""
    if pedestrians == ["no pedestrians"]:
        return "No pedestrians are observed."
    return f"Pedestrian activity includes {join_phrases(pedestrians)}."


def format_directionality(label: Mapping[str, object]) -> str:
    directionality = humanize_token(label.get("Directionality")) if "Directionality" in label else ""
    if not directionality:
        return ""
    return f"The road operates as a {directionality} route."


def format_ego_vehicle(label: Mapping[str, object]) -> str:
    info = label.get("Ego-Vehicle")
    if not isinstance(info, Mapping):
        return ""

    direction = humanize_token(info.get("Direction")) if "Direction" in info else ""
    maneuver = humanize_token(info.get("Maneuver")) if "Maneuver" in info else ""

    if direction and maneuver:
        return f"The ego vehicle is {maneuver} while heading {direction}."
    if maneuver:
        return f"The ego vehicle is {maneuver}."
    if direction:
        return f"The ego vehicle is heading {direction}."
    return ""


def format_visibility(label: Mapping[str, object]) -> str:
    info = label.get("Visibility")
    if not isinstance(info, Mapping):
        return ""

    statements: List[str] = []
    general = humanize_token(info.get("General")) if "General" in info else ""
    if general:
        statements.append(f"Overall visibility is {general}.")

    specifics_raw = info.get("SpecificImpairments")
    specifics = normalize_list(specifics_raw) if specifics_raw else []
    meaningful_specifics = [item for item in specifics if item != "no specific impairments"]

    if meaningful_specifics:
        statements.append(
            f"Specific visibility impairments include {join_phrases(meaningful_specifics)}."
        )
    elif specifics and not meaningful_specifics:
        statements.append("There are no specific visibility impairments noted.")

    return " ".join(statements)


def format_camera_condition(label: Mapping[str, object]) -> str:
    camera = humanize_token(label.get("CameraCondition")) if "CameraCondition" in label else ""
    if not camera:
        return ""
    return f"The camera condition is {camera}."


def format_severity(label: Mapping[str, object]) -> str:
    severity = label.get("Severity")
    if severity in (None, ""):
        return ""
    return f"The risk severity score provided is {severity}."


def build_description(label: Mapping[str, object]) -> str:
    """
    Convert a label dictionary into a natural-language description of the scene.
    """
    segments = [
        format_environment(label),
        format_lane_information(label),
        format_traffic_signs(label),
        format_vehicles(label),
        format_pedestrians(label),
        format_directionality(label),
        format_ego_vehicle(label),
        format_visibility(label),
        format_camera_condition(label),
        format_severity(label),
    ]
    return " ".join(segment for segment in segments if segment)


def resolve_image_path(image_path: Path, dataset_dir: Path, use_relative: bool) -> str:
    if use_relative:
        return image_path.relative_to(dataset_dir).as_posix()
    return image_path.resolve().as_posix()


def export_dataset(config: Mapping[str, Any], *, quiet: bool = False) -> int:
    dataset_dir = Path(config["dataset_dir"]).resolve()
    image_root = dataset_dir / "images"
    label_root = dataset_dir / "labels"
    debug = bool(config.get("debug", False))
    preview_limit = 10 if debug else 0
    preview_count = 0

    debug_log(debug, f"Dataset directory: {dataset_dir}")
    debug_log(debug, f"Image root: {image_root}")
    debug_log(debug, f"Label root: {label_root}")

    if not image_root.exists():
        raise FileNotFoundError(f"Missing images folder: {image_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"Missing labels folder: {label_root}")

    label_index = build_label_index(label_root)
    debug_log(debug, f"Indexed {len(label_index)} label files.")
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    missing_labels: List[str] = []
    limit = config.get("limit")
    system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    user_prompt = config.get("user_prompt", DEFAULT_USER_PROMPT)
    use_relative_paths = bool(config.get("relative_paths"))
    append_label_json = bool(config.get("append_label_json", True))

    with output_path.open("w", encoding="utf-8") as handle:
        for dataset_name, image_path in iter_image_files(image_root):
            if limit is not None and limit >= 0 and total_written >= limit:
                break

            key = (dataset_name, image_path.name)
            label_path = label_index.get(key)
            if not label_path:
                debug_log(
                    debug,
                    f"Missing label for image: {dataset_name}/{image_path.name}",
                )
                missing_labels.append(f"{dataset_name}/{image_path.name}")
                continue

            try:
                with label_path.open("r", encoding="utf-8") as label_handle:
                    label_data = json.load(label_handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse label JSON: {label_path}") from exc

            description = build_description(label_data)
            if not description:
                # Skip entries that fail to produce a description.
                # Provide insight when debugging.
                debug_log(
                    debug,
                    f"Empty description generated for {dataset_name}/{image_path.name}; skipping.",
                )
                continue

            if preview_count < preview_limit:
                resolved_image = resolve_image_path(
                    image_path, dataset_dir, use_relative_paths
                )
                description_sample = (
                    description
                    if len(description) <= 280
                    else f"{description[:277]}..."
                )
                debug_log(
                    debug,
                    f"Sample {preview_count + 1}: image={resolved_image}, label={label_path}",
                )
                debug_log(debug, f"  Description: {description_sample}")
                preview_count += 1

            if append_label_json:
                label_json = json.dumps(
                    label_data, ensure_ascii=False, sort_keys=True, indent=2
                )
                assistant_content = f"{description}\n{label_json}"
                if preview_count <= preview_limit:
                    debug_log(
                        debug,
                        "  JSON snippet: "
                        + (
                            label_json.replace("\n", "\\n")[:240]
                            + ("..." if len(label_json) > 240 else "")
                        ),
                    )
            else:
                assistant_content = description

            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": [
                    resolve_image_path(image_path, dataset_dir, use_relative_paths)
                ],
            }

            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")
            total_written += 1

    debug_log(debug, f"Total samples written: {total_written}")
    debug_log(debug, f"Images without labels: {len(missing_labels)}")

    if not quiet:
        path_display = output_path.resolve()
        print(f"Wrote {total_written} samples to {path_display}")
        if missing_labels:
            print(
                f"Skipped {len(missing_labels)} images with no matching labels "
                f"(e.g. {missing_labels[0]})."
            )

    return total_written


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config_path = args.config
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        config = load_config(config_path)
        if args.limit is not None:
            config["limit"] = args.limit
        if args.debug:
            config["debug"] = True
        debug_log(
            config.get("debug", False),
            f"Loaded configuration from {config_path} with settings: "
            f"dataset_dir={config['dataset_dir']}, output={config['output']}, "
            f"limit={config['limit']}, relative_paths={config['relative_paths']}",
        )
        export_dataset(config, quiet=args.quiet)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
