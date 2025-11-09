#!/usr/bin/env python3
"""Run the complete analysis pipeline: summarize, extract embeddings, and plot clusters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarise import run_summarise  # pylint: disable=C0413, E0401
from scripts.extract_embeddings import (  # pylint: disable=C0413, E0401
    run_extract_embeddings,
)
from scripts.plot_clusters import run_plot_clusters  # pylint: disable=C0413, E0401


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

    # Load configuration
    with args.data_config_path.open("r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle) or {}

    # Extract configuration parameters
    dataset_path = Path(data_config.get("dataset_path"))
    model_names = data_config.get("model_names")
    output_dir = Path(data_config.get("plots_dir"))
    random_state = data_config.get("random_state", 42)
    layers_to_plot = data_config.get("clustering_layers", [0, 6, 12])

    print("=" * 70)
    print(" " * 20 + "ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Configuration: {args.data_config_path}")
    print(f"Database: {dataset_path}")
    print("=" * 70)
    print()

    # Summarize dataset
    print("\n" + "=" * 70)
    print(" " * 25 + "SUMMARIZE DATASET")
    print("=" * 70)
    run_summarise(dataset_path)

    for model_name in model_names:
        print("\n" + "=" * 70)
        print(" " * 25 + f"MODEL: {model_name}")
        print("=" * 70)

        # Step 1: Extract embeddings
        print("\n" + "-" * 70)
        print(" " * 25 + "STEP 1: EXTRACT EMBEDDINGS")
        print("-" * 70)
        run_extract_embeddings(dataset_path, model_name)

        # Step 2: Plot clusters
        print("\n" + "-" * 70)
        print(" " * 25 + "STEP 2: PLOT CLUSTERS")
        print("-" * 70)
        run_plot_clusters(
            dataset_path, model_name, output_dir, layers_to_plot, random_state
        )

    # Final summary
    print("\n" + "=" * 70)
    print(" " * 20 + "PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nAll analysis steps completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
