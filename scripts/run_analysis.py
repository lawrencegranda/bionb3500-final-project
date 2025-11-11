#!/usr/bin/env python3
"""Run the complete analysis pipeline: summarize, extract embeddings, and plot clusters."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.helpers import get_args  # pylint: disable=C0413, E0401
from scripts.summarise import run_summarise  # pylint: disable=C0413, E0401
from scripts.extract_embeddings import (  # pylint: disable=C0413, E0401
    run_extract_embeddings,
)
from scripts.plot_clusters import run_plot_clusters  # pylint: disable=C0413, E0401
from scripts.evaluate_metrics import (  # pylint: disable=C0413, E0401
    run_evaluate_metrics,
)


def main() -> None:
    """Program entrypoint."""
    args = get_args(__doc__)

    # Load configuration
    dataset_path = args.config.paths.dataset_path
    model_names = args.config.model.model_names
    output_dir = args.config.paths.plots_dir
    random_states = args.config.model.random_states
    clustering_layers = args.config.model.clustering_layers
    clustering_methods = args.config.model.clustering_methods

    print("=" * 70)
    print(" " * 20 + "ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Database: {dataset_path}")
    print(f"Random states: {random_states}")
    print("=" * 70)
    print()

    # Summarize dataset
    print("\n" + "=" * 70)
    print(" " * 25 + "SUMMARIZE DATASET")
    print("=" * 70)
    run_summarise(dataset_path)

    for model_name in model_names:
        layers_to_plot = clustering_layers.get(model_name)

        print("\n" + "=" * 70)
        print(" " * 25 + f"MODEL: {model_name}")
        print("=" * 70)

        # Step 1: Extract embeddings
        print("\n" + "-" * 70)
        print(" " * 25 + "STEP 1: EXTRACT EMBEDDINGS")
        print("-" * 70)
        run_extract_embeddings(dataset_path, model_name)

        # Step 2: Plot clusters (only once with first seed)
        print("\n" + "-" * 70)
        print(" " * 25 + "STEP 2: PLOT CLUSTERS")
        print("-" * 70)
        print(f"Using seed {random_states[0]} for plotting")
        run_plot_clusters(
            dataset_path, model_name, output_dir, layers_to_plot, random_states[0]
        )

        # Step 3: Evaluate metrics (with all seeds)
        print("\n" + "-" * 70)
        print(" " * 25 + "STEP 3: EVALUATE METRICS")
        print("-" * 70)
        print(f"Evaluating with {len(random_states)} seeds: {random_states}")
        run_evaluate_metrics(
            dataset_path,
            model_name,
            args.config.paths.metrics_dir,
            layers_to_plot,
            clustering_methods,
            random_states,
        )

    # Final summary
    print("\n" + "=" * 70)
    print(" " * 20 + "PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nAll analysis steps completed successfully!")
    print(f"Seeds evaluated: {len(random_states)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
