"""Summarize dataset statistics from a SQLite database specified in data.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.dataset import Dataset  # pylint: disable=C0413


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

    dataset_path = Path(data_config.get("dataset_path"))

    if not dataset_path.exists():
        print(f"Error: Database file not found at {dataset_path}")
        sys.exit(1)

    # Load the dataset and convert to pandas DataFrame
    dataset = Dataset.from_db(dataset_path)
    df = dataset.to_df()
    dataset.close()

    # Print overall statistics
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Database path: {dataset_path}")
    print(f"Total rows: {len(df)}")
    print()

    # Print statistics by label
    print("-" * 80)
    print("STATISTICS BY LABEL")
    print("-" * 80)
    label_counts = df.groupby(["lemma", "label"]).size().sort_values(ascending=False)
    for (lemma, label), count in label_counts.items():
        print(f"  {lemma} ({label}): {count} sentences")
    print()

    # Print statistics by synset
    print("-" * 80)
    print("STATISTICS BY SYNSET")
    print("-" * 80)
    synset_counts = df.groupby(["lemma", "synset"]).size().sort_values(ascending=False)
    for (lemma, synset), count in synset_counts.items():
        print(f"  {lemma} ({synset}): {count} sentences")
    print()

    # Print statistics by source
    print("-" * 80)
    print("STATISTICS BY SOURCE")
    print("-" * 80)
    source_counts = df.groupby("source").size().sort_values(ascending=False)
    for source, count in source_counts.items():
        print(f"  {source}: {count} sentences")
    print()

    # Print selected columns (lemma, label, first 100 chars of text)
    print("=" * 80)
    print("ALL ROWS (lemma, label, text[:100])")
    print("=" * 80)
    display_df = df[["lemma", "label", "text"]][:100].copy()
    display_df["text"] = display_df["text"].str.slice(0, 100)
    print(display_df.to_string(index=False))
    print()

    print("=" * 80)
    print(f"Total: {len(display_df)} rows displayed")
    print("=" * 80)


if __name__ == "__main__":  # pragma: no cover
    main()
