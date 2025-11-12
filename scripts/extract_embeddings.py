#!/usr/bin/env python3
"""Extract BERT embeddings for all sentences in the dataset."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.helpers import get_args  # pylint: disable=C0413, E0401
from src.dataset import Database  # pylint: disable=C0413,E0401


def run_extract_embeddings(
    db_path: Path, model_name: str = "bert-base-uncased"
) -> None:
    """
    Extract and store BERT embeddings for the dataset.

    Args:
        db_path: Path to the SQLite database
        model_name: HuggingFace model identifier
    """
    if not db_path.exists():
        print(f"Database not found at {db_path}.")
        return

    print(f"Loading dataset from {db_path}")
    with Database.from_db(db_path, model_name=model_name) as database:
        sentences_table = database.sentences_table
        print(f"Found {len(sentences_table)} sentences in the dataset.")

        print(f"Initialized BERT model: {model_name}.")
        embeddings_table = database.embeddings_table

        print("Extracting embeddings...")
        embeddings_table.reset()
        embeddings_table.process_dataset(sentences_table)

        print(f"Done! Embeddings saved to {db_path}.")


def main() -> None:
    """Program entrypoint."""
    args = get_args(__doc__, get_model_arg=True)
    run_extract_embeddings(args.config.paths.dataset_path, args.model)


if __name__ == "__main__":
    main()
