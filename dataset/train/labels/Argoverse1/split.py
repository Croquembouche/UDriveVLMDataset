#!/usr/bin/env python3
import json
from pathlib import Path

def main(input_path: str, output_dir: str, content: str = "value") -> None:
    """
    Split a nested JSON into one JSON per JPG entry.

    Params
    ------
    input_path : path to the nested JSON file
    output_dir : directory to write per-image JSON files, grouped by top-level key
    content    : what to write per file:
                 - "value"       -> entry["value"] only  (default)
                 - "entry"       -> the entire entry dict (id + value + anything else)
                 - "value+meta"  -> {"id": <id>, "group": <top_key>, "value": <value>}
    """
    in_path = Path(input_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object mapping group IDs to lists of entries.")

    written = 0
    for group_key, entries in data.items():
        if not isinstance(entries, list):
            continue
        group_dir = out_root / str(group_key)
        group_dir.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            img_id = entry.get("id")
            if not (isinstance(img_id, str) and img_id.endswith(".jpg")):
                continue

            # Choose payload
            if content == "entry":
                payload = entry
            elif content == "value+meta":
                payload = {"id": img_id, "group": group_key}
                if "value" in entry:
                    payload["value"] = entry["value"]
            elif content == "value":
                payload = entry.get("value")
            else:
                raise ValueError('content must be one of: "value", "entry", "value+meta"')

            # Safe filename "<jpg>.json" under the group's directory
            out_path = group_dir / (Path(img_id).name + ".json")
            with out_path.open("w", encoding="utf-8") as out_f:
                json.dump(payload, out_f, ensure_ascii=False, indent=2)
            written += 1

    print(f"Wrote {written} files under {out_root}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Create one JSON file per JPG entry from a nested JSON.")
    p.add_argument("input", help="Path to the nested JSON file.")
    p.add_argument("output_dir", help="Directory to write the per-image JSON files.")
    p.add_argument("--content",
                   choices=["value", "entry", "value+meta"],
                   default="value",
                   help='What to write to each JSON (default: "value").')
    args = p.parse_args()
    main(args.input, args.output_dir, args.content)

