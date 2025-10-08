#!/usr/bin/env python3
import json
from pathlib import Path

def main(input_path: str,
         output_dir: str,
         content: str = "annotation",
         exts: str = ".png,.jpg,.jpeg",
         group_by_prefix_tokens: int = 0) -> None:
    """
    Write one JSON file per image-key in a dict-of-annotations JSON.

    Params
    ------
    input_path : Path to the input JSON file.
    output_dir : Directory to write per-image JSON files.
    content    : What to write per file:
                 - "annotation"        -> value only (default)
                 - "annotation+meta"   -> {"id": <key>, "annotation": <value>}
    exts       : Comma-separated list of file extensions to include (case-insensitive).
    group_by_prefix_tokens :
                 If >0, create subfolders based on the first N '_'-separated tokens
                 of the filename (e.g., 1 -> "aachen", "bochum", "bremen").
    """
    in_path = Path(input_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object mapping image filenames to annotations.")

    allowed = tuple(e.strip().lower() for e in exts.split(",") if e.strip())
    written = 0

    for key, annotation in data.items():
        if not isinstance(key, str):
            continue
        if not key.lower().endswith(allowed):
            continue

        # Determine output directory (optional grouping by filename prefix)
        if group_by_prefix_tokens > 0:
            parts = key.split("_")
            group = "_".join(parts[:group_by_prefix_tokens]) if parts else "misc"
            out_dir = out_root / group
        else:
            out_dir = out_root
        out_dir.mkdir(parents=True, exist_ok=True)

        # Payload selection
        if content == "annotation":
            payload = annotation
        elif content == "annotation+meta":
            payload = {"id": key, "annotation": annotation}
        else:
            raise ValueError('content must be one of: "annotation", "annotation+meta"')

        # Safe filename: "<original-filename>.json"
        out_path = out_dir / (Path(key).name + ".json")
        with out_path.open("w", encoding="utf-8") as out_f:
            json.dump(payload, out_f, ensure_ascii=False, indent=2)
        written += 1

    print(f"Wrote {written} files under {out_root.resolve()}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Create one JSON file per image-key from a dict-of-annotations JSON.")
    p.add_argument("input", help="Path to the input JSON.")
    p.add_argument("output_dir", help="Directory to write per-image JSON files.")
    p.add_argument("--content", choices=["annotation", "annotation+meta"], default="annotation",
                   help='Payload written per file (default: "annotation").')
    p.add_argument("--exts", default=".png,.jpg,.jpeg",
                   help="Comma-separated image extensions to include (default: .png,.jpg,.jpeg).")
    p.add_argument("--group-by-prefix-tokens", type=int, default=0,
                   help="If >0, group outputs into folders named by the first N '_' tokens of the filename.")
    args = p.parse_args()
    main(args.input, args.output_dir, args.content, args.exts, args.group_by_prefix_tokens)

