# UDriveVLMDataset

This repository packages multimodal driving-scene data into training-ready JSONL
files for fine-tuning visual-language models (VLMs) such as
`qwen2-vl-2b-instruct` using the
[ms-swift](https://github.com/modelscope/ms-swift) toolkit. It ships with a
converter script that pairs driving images with structured scene analyses and
creates conversational supervision samples that include both natural-language
summaries and the original JSON annotations.

## Project Layout

- `dataset/` – Source data with split folders (e.g. `train/images` and
  `train/labels`) plus the generated JSONL outputs.
- `configs/qwen_dataset.yaml` – Default configuration for running the converter.
- `utils/qwen_custom_dataset.py` – Core script that builds the ms-swift friendly
  dataset.

## Prerequisites

- Python 3.8+ (tested with 3.10)
- `PyYAML` for reading configuration files

Install the Python dependency if needed:

```bash
pip install pyyaml
```

## Running the Converter

The script reads its settings (input paths, output location, prompts, etc.)
from `configs/qwen_dataset.yaml`. You can run it with no arguments to produce
the full training JSONL:

```bash
python3 utils/qwen_custom_dataset.py
```

Key configuration fields:

- `dataset_dir` – Folder containing `images/` and `labels/` subdirectories.
- `output` – Destination JSONL file.
- `relative_paths` – Set `true` to store image paths relative to
  `dataset_dir`; `false` keeps absolute paths.
- `limit` – Optional sample cap (useful for smoke tests or validation splits).
- `system_prompt` / `user_prompt` – Prompts injected into each conversation.
- `debug` – When `true`, prints dataset diagnostics and previews the first N
  image/label pairs.
- `append_label_json` – When `true`, appends the raw label JSON after the
  narrative summary in each assistant response.

### Overriding Config Values Temporarily

You can adjust specific settings at runtime without editing the YAML:

```bash
python3 utils/qwen_custom_dataset.py --limit 100 --debug
```

This example processes only 100 samples and enables verbose logging.

## Output Format

Each JSONL line includes:

- `messages`: system, user, and assistant turns formatted for ms-swift SFT.
  The assistant message combines a natural-language summary of the scene and
  the original structured label JSON.
- `images`: list containing the associated image path (absolute or relative).

Sample entry:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "Natural-language summary...\n{ \"Scene\": \"Urban\", ... }"}
  ],
  "images": ["dataset/train/images/Argoverse1/example.jpg"]
}
```

The resulting JSONL can be supplied directly to `swift train` or other
ms-swift workflows for VLM fine-tuning.

## Next Steps

- Adjust prompts or formatting in `configs/qwen_dataset.yaml` to match your
  training objectives.
- Split the output JSONL into train/validation sets if needed.
- Integrate the generated dataset into an ms-swift training command, e.g.:

```bash
swift train \
  --model_type qwen2-vl-2b-instruct \
  --train_dataset dataset/train_qwen.jsonl \
  [other-options...]
```

Feel free to extend the utilities for other annotation schemas or VLM models.
