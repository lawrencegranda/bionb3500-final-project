"""Summarize dataset statistics from a SQLite database specified in data.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import yaml
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.dataset import Database  # pylint: disable=C0413,E0401


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--data-config-path",
        type=Path,
        required=True,
        help="Path to the YAML data configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Program entrypoint."""

    args = _parse_args(argv)

    # Load the data configuration
    with args.data_config_path.open("r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle) or {}

    model_name = data_config.get("model_name")
    dataset_path = Path(data_config.get("dataset_path"))

    if not dataset_path.exists():
        print(f"Error: Database file not found at {dataset_path}")
        sys.exit(1)

    # Load the dataset and convert to pandas DataFrame
    database = Database.from_db(dataset_path, model_name)
    df = database.sentences_table.to_df()

    # Print overall statistics
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Database path: {dataset_path}")
    print(f"Total rows: {len(df)}")
    print()

    # Print statistics by label
    print("-" * 50)
    print("STATISTICS BY LABEL")
    print("-" * 50)
    label_counts = df.groupby(["lemma", "label"]).size().sort_values(ascending=False)
    for (lemma, label), count in label_counts.items():
        print(f"  {lemma} ({label}): {count} sentences")
    print()

    # Print statistics by synset
    print("-" * 50)
    print("STATISTICS BY SYNSET")
    print("-" * 50)
    synset_counts = df.groupby(["lemma", "synset"]).size().sort_values(ascending=False)
    for (lemma, synset), count in synset_counts.items():
        print(f"  {lemma} ({synset}): {count} sentences")
    print()

    # Print statistics by source
    print("-" * 50)
    print("STATISTICS BY SOURCE")
    print("-" * 50)
    source_counts = df.groupby("source").size().sort_values(ascending=False)
    for source, count in source_counts.items():
        print(f"  {source}: {count} sentences")
    print()

    database.close()

    print("=" * 50)
    print("END OF SUMMARY")
    print("=" * 50)


if __name__ == "__main__":  # pragma: no cover
    main()
