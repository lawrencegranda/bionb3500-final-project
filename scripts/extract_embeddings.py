#!/usr/bin/env python3
"""Extract BERT embeddings for all sentences in the dataset."""

import logging
import argparse
from pathlib import Path
import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.embeddings import EmbeddingStore  # pylint: disable=C0413
from src.dataset import Dataset  # pylint: disable=C0413

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExtractEmbeddings")


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
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="BERT layer to extract embeddings up to (-1 for last layer)",
    )

    args = parser.parse_args()

    with open(args.data_config_path, "r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)

    db_path = Path(data_config.get("dataset_path"))

    if not db_path.exists():
        logger.error("Database not found at %s", db_path)
        return

    logger.info("Loading dataset from %s", db_path)
    dataset = Dataset.from_db(db_path)

    logger.info("Initializing BERT model: %s", args.model)
    with EmbeddingStore.from_db(db_path, model_name=args.model) as store:
        logger.info("Extracting embeddings up to layer %d", args.layer)
        store.process_dataset(dataset, layer=args.layer)

    logger.info("Done! Embeddings saved to %s", db_path)
    dataset.close()


if __name__ == "__main__":
    main()
