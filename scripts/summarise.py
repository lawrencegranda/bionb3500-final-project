"""Summarize dataset statistics from a SQLite database specified in data.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.helpers import get_args  # pylint: disable=C0413, E0401
from src.dataset import Database  # pylint: disable=C0413,E0401


def run_summarise(dataset_path: Path) -> None:
    """Run dataset summarization."""
    if not dataset_path.exists():
        print(f"Error: Database file not found at {dataset_path}")
        sys.exit(1)

    # Load the dataset and convert to pandas DataFrame
    database = Database.from_db(dataset_path)
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


def main() -> None:
    """Program entrypoint."""
    args = get_args(__doc__)
    run_summarise(args.config.paths.dataset_path)


if __name__ == "__main__":  # pragma: no cover
    main()
