#!/usr/bin/env python3
"""Evaluate clustering metrics across layers and export to CSV."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Sequence
from collections import defaultdict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.helpers import get_args  # pylint: disable=C0413, E0401
from src.dataset import Database  # pylint: disable=C0413, E0401
from src.analysis.metrics import (  # pylint: disable=C0413, E0401
    make_metrics,
    metric_models,
)


def run_evaluate_metrics(
    dataset_path: Path,
    model_name: str,
    output_dir: Path,
    layers_to_evaluate: Sequence[int],
    clustering_methods: Sequence[str],
    random_states: Sequence[int],
) -> None:
    """
    Evaluate clustering metrics for the dataset and save to CSV.

    Args:
        dataset_path: Path to the SQLite database
        model_name: Model name for the database
        output_dir: Directory to save metrics CSV
        layers_to_evaluate: List of layer indices to evaluate
        clustering_methods: List of clustering methods to evaluate
        random_states: List of random seeds for reproducibility
    """
    if not dataset_path.exists():
        print(f"Error: Database file not found at {dataset_path}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("CLUSTERING METRICS EVALUATION")
    print("=" * 50)
    print(f"Database: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"Layers to evaluate: {layers_to_evaluate}")
    print(f"Clustering methods: {clustering_methods}")
    print(f"Output directory: {output_dir}")
    print(f"Random states: {random_states}")
    print()

    # Load database
    database = Database.from_db(dataset_path, model_name)

    # Get all metric classes
    all_metric_classes = list(metric_models.values())

    print(f"Computing metrics: {', '.join(metric_models.keys())}")
    print("-" * 50)

    # Evaluate each clustering method
    for clustering_method in clustering_methods:
        print(f"\n{'=' * 50}")
        print(f"CLUSTERING METHOD: {clustering_method.upper()}")
        print("=" * 50)

        # Store metrics for each seed: metric_name -> (lemma, layer) -> [values for each seed]
        metrics_by_seed = {
            metric_name: defaultdict(list) for metric_name in metric_models.keys()
        }

        # Run metrics for each random state
        for seed_idx, random_state in enumerate(random_states):
            print(f"\n--- Seed {seed_idx + 1}/{len(random_states)}: {random_state} ---")

            # Compute metrics for all lemmas
            metrics_by_lemma = make_metrics(
                database, all_metric_classes, random_state, clustering_method
            )

            for lemma, lemma_metrics in metrics_by_lemma.items():
                for layer, layer_metrics in lemma_metrics.layers.items():
                    # Only include specified layers
                    if layers_to_evaluate and layer not in layers_to_evaluate:
                        continue

                    for metric_name, metric_record in layer_metrics.labels.items():
                        key = (metric_record.lemma, metric_record.layer)
                        metrics_by_seed[metric_name][key].append(metric_record.value)

        # Create model-specific directory with clustering method subdirectory
        model_output_dir = output_dir / model_name / clustering_method
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Write one CSV per metric with all seed values and median
        for metric_name, metrics_dict in metrics_by_seed.items():
            if not metrics_dict:
                continue

            # Prepare rows for CSV
            rows = []
            for (lemma, layer), values in sorted(metrics_dict.items()):
                # Calculate median
                median_value = float(np.median(values))

                # Create row with lemma, layer, all seed values, and median
                row = {"lemma": lemma, "layer": layer}
                for seed_idx, value in enumerate(values):
                    row[f"seed_{random_states[seed_idx]}"] = value
                row["median"] = median_value

                rows.append(row)

                # Print summary for first lemma
                if lemma == sorted(set(k[0] for k in metrics_dict.keys()))[0]:
                    print(
                        f"  Layer {layer:2d} | {metric_name:25s}: "
                        f"{median_value:.4f} (median of {len(values)} seeds)"
                    )

            # Write to CSV
            output_file = model_output_dir / f"{metric_name}.csv"
            with output_file.open("w", newline="", encoding="utf-8") as csvfile:
                # Fieldnames: lemma, layer, seed_1, seed_2, ..., median
                fieldnames = ["lemma", "layer"]
                fieldnames.extend([f"seed_{rs}" for rs in random_states])
                fieldnames.append("median")

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"\nSaved {metric_name}: {output_file}")

    database.close()

    print("\n" + "=" * 50)
    print("METRICS EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {output_dir / model_name}")
    print(f"Clustering methods evaluated: {', '.join(clustering_methods)}")
    print(f"Seeds evaluated: {len(random_states)}")
    print("=" * 50)


__all__ = ["run_evaluate_metrics"]


def main() -> None:
    """Program entrypoint."""
    args = get_args(__doc__, get_model_arg=True)

    # Get layers from config or use all layers
    layers_to_evaluate = args.config.model.clustering_layers.get(args.model)

    # Get clustering methods from config
    clustering_methods = args.config.model.clustering_methods

    run_evaluate_metrics(
        args.config.paths.dataset_path,
        args.model,
        args.config.paths.metrics_dir,
        layers_to_evaluate,
        clustering_methods,
        args.config.model.random_states,
    )


if __name__ == "__main__":
    main()
