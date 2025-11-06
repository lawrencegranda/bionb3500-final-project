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


from src.dataset import Dataset  # pylint: disable=C0413
from src.embeddings import EmbeddingStore  # pylint: disable=C0413


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
    print("SAMPLE SENTENCES (lemma, label, text[:100])")
    print("=" * 80)
    display_df = df[["lemma", "label", "text"]][:20].copy()
    display_df["text"] = display_df["text"].str.slice(0, 100)
    print(display_df.to_string(index=False))
    print()

    embedding_store = EmbeddingStore.from_db(dataset_path)
    embeddings_df = embedding_store.to_df()

    if len(embeddings_df) > 0:
        print("=" * 80)
        print("EMBEDDINGS SUMMARY")
        print("=" * 80)
        print(f"Total embeddings: {len(embeddings_df)}")
        print(
            f"Unique sentences with embeddings: {embeddings_df['sentence_id'].nunique()}"
        )
        print(f"Number of layers: {embeddings_df['layer'].nunique()}")
        print(f"Layers available: {sorted(embeddings_df['layer'].unique())}")
        print()

        # Join with sentences to get lemma and label
        print("-" * 80)
        print("EMBEDDINGS TABLE (first 20 rows)")
        print("-" * 80)
        merged_df = embeddings_df.merge(
            df[["lemma", "label"]],
            left_on="sentence_id",
            right_index=True,
            how="left",
        )
        merged_df["dimension"] = merged_df["embedding"].apply(len)

        # Show without the embedding blob for readability
        display_cols = ["sentence_id", "lemma", "label", "layer", "dimension"]
        print(merged_df[display_cols].head(20).to_string(index=False))
        print()

        # Statistics by layer
        print("-" * 80)
        print("EMBEDDINGS BY LAYER")
        print("-" * 80)
        layer_counts = embeddings_df.groupby("layer").size().sort_index()
        for layer, count in layer_counts.items():
            print(f"  Layer {layer}: {count} embeddings")
        print()

        # Mean and std dev per label for each layer
        print("=" * 80)
        print("MEAN EMBEDDING ANALYSIS PER LABEL AND LAYER")
        print("=" * 80)

        # Convert embeddings from BLOB to numpy arrays
        merged_df["embedding_array"] = merged_df["embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )

        # Group by label and layer
        for label in sorted(merged_df["label"].dropna().unique()):
            print(f"\nLabel: {label}")
            print("-" * 80)

            label_df = merged_df[merged_df["label"] == label]

            for layer in sorted(label_df["layer"].unique()):
                layer_embeddings = label_df[label_df["layer"] == layer][
                    "embedding_array"
                ]

                if len(layer_embeddings) == 0:
                    continue

                # Stack embeddings into a 2D array
                embeddings_matrix = np.stack(layer_embeddings.values)

                # Calculate mean and std
                mean_embedding = np.mean(embeddings_matrix, axis=0)
                std_embedding = np.std(embeddings_matrix, axis=0)
                max_norm = np.max(np.linalg.norm(embeddings_matrix, axis=0))
                min_norm = np.min(np.linalg.norm(embeddings_matrix, axis=0))

                # Summary statistics
                mean_norm = np.linalg.norm(mean_embedding)
                mean_std = np.mean(std_embedding)

                print(
                    f"  Layer {layer:2d}: "
                    f"n={len(layer_embeddings):4d}, "
                    f"mean_norm={mean_norm:8.4f}, "
                    f"avg_std={mean_std:8.4f}, "
                    f"max_norm={max_norm:8.4f}, "
                    f"min_norm={min_norm:8.4f}"
                )

        print()
        print("=" * 80)
        print("SUMMARY STATISTICS PER LAYER (ALL LABELS)")
        print("=" * 80)

        for layer in sorted(merged_df["layer"].unique()):
            layer_embeddings = merged_df[merged_df["layer"] == layer]["embedding_array"]

            if len(layer_embeddings) == 0:
                continue

            # Stack embeddings into a 2D array
            embeddings_matrix = np.stack(layer_embeddings.values)

            # Calculate statistics
            mean_embedding = np.mean(embeddings_matrix, axis=0)
            std_embedding = np.std(embeddings_matrix, axis=0)

            mean_norm = np.linalg.norm(mean_embedding)
            mean_std = np.mean(std_embedding)
            max_std = np.max(std_embedding)
            min_std = np.min(std_embedding)

            print(
                f"Layer {layer:2d}: "
                f"n={len(layer_embeddings):4d}, "
                f"mean_norm={mean_norm:8.4f}, "
                f"avg_std={mean_std:8.4f}, "
                f"std_range=[{min_std:.4f}, {max_std:.4f}]"
            )

        print()
    else:
        print("=" * 80)
        print("No embeddings found in database")
        print("=" * 80)
        print()
    embedding_store.close()

    print("=" * 80)
    print("END OF SUMMARY")
    print("=" * 80)


if __name__ == "__main__":  # pragma: no cover
    main()
