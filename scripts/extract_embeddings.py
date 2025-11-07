#!/usr/bin/env python3
"""Extract BERT embeddings for all sentences in the dataset."""

import argparse
from pathlib import Path
import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.dataset import Database  # pylint: disable=C0413,E0401


def main():
    """Extract and store BERT embeddings for the dataset."""
    parser = argparse.ArgumentParser(
        description="Extract BERT embeddings for sentences in the dataset."
    )
    parser.add_argument(
        "-d",
        "--data-config-path",
        type=Path,
        required=True,
        help="Path to the YAML configuration containing data configuration.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model identifier",
    )
    args = parser.parse_args()

    with open(args.data_config_path, "r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)

    db_path = Path(data_config.get("dataset_path"))

    if not db_path.exists():
        print(f"Database not found at {db_path}.")
        return

    print("Loading dataset from %s", db_path)
    with Database.from_db(db_path, model_name=args.model) as database:
        sentences_table = database.sentences_table
        print(f"Found {len(sentences_table)} sentences in the dataset.")

        print(f"Initialized BERT model: {args.model}.")
        embeddings_table = database.embeddings_table

        print("Extracting embeddings...")
        embeddings_table.process_dataset(sentences_table)

        print(f"Done! Embeddings saved to {db_path}.")


if __name__ == "__main__":
    main()
