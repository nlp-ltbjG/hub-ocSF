import argparse
import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
DEFAULT_INPUT_DIR = ROOT / "data" / "peoples_daily"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "peoples_daily_entities"


def bio_to_entities(tokens: list[str], tags: list[str]) -> list[dict[str, str]]:
    entities = []
    current_type = None
    current_tokens = []

    def flush_current() -> None:
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            entities.append({
                "text": "".join(current_tokens),
                "type": current_type,
            })
        current_type = None
        current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag == "O":
            flush_current()
            continue

        if "-" not in tag:
            flush_current()
            continue

        prefix, entity_type = tag.split("-", 1)
        if prefix == "B":
            flush_current()
            current_type = entity_type
            current_tokens = [token]
        elif prefix == "I":
            if current_type == entity_type:
                current_tokens.append(token)
            else:
                flush_current()
                current_type = entity_type
                current_tokens = [token]
        else:
            flush_current()

    flush_current()
    return entities


def convert_record(record: dict) -> dict:
    tokens = record["tokens"]
    tags = record["ner_tags"]
    return {
        "text": "".join(tokens),
        "entities": bio_to_entities(tokens, tags),
    }


def convert_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    converted = [convert_record(record) for record in records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"{input_path.name}: {len(converted)} records -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Peoples Daily BIO NER data to text/entities format."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset split names without .json suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        convert_file(args.input_dir / f"{split}.json", args.output_dir / f"{split}.json")


if __name__ == "__main__":
    main()
