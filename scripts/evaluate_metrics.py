#!/usr/bin/env python3
"""Evaluate clustering metrics across layers and export to CSV."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Sequence

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
    random_state: int = 42,
) -> None:
    """
    Evaluate clustering metrics for the dataset and save to CSV.

    Args:
        dataset_path: Path to the SQLite database
        model_name: Model name for the database
        output_dir: Directory to save metrics CSV
        layers_to_evaluate: List of layer indices to evaluate
        random_state: Random seed for reproducibility
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
    print(f"Output directory: {output_dir}")
    print(f"Random state: {random_state}")
    print()

    # Load database
    database = Database.from_db(dataset_path, model_name)

    # Get all metric classes
    all_metric_classes = list(metric_models.values())

    print(f"Computing metrics: {', '.join(metric_models.keys())}")
    print("-" * 50)

    # Compute metrics for all lemmas
    metrics_by_lemma = make_metrics(database, all_metric_classes, random_state)

    # Organize data by metric
    # Structure: metric_name -> list of {lemma, layer, value}
    metrics_data = {metric_name: [] for metric_name in metric_models.keys()}

    for lemma, lemma_metrics in metrics_by_lemma.items():
        print(f"\nProcessing lemma: {lemma}")

        for layer, layer_metrics in lemma_metrics.layers.items():
            # Only include specified layers
            if layers_to_evaluate and layer not in layers_to_evaluate:
                continue

            for metric_name, metric_record in layer_metrics.labels.items():
                metrics_data[metric_name].append(
                    {
                        "lemma": metric_record.lemma,
                        "layer": metric_record.layer,
                        "value": metric_record.value,
                    }
                )
                print(
                    f"  Layer {layer:2d} | {metric_name:25s}: {metric_record.value:.4f}"
                )

    database.close()

    # Create model-specific directory
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Write one CSV per metric
    total_files = 0
    for metric_name, rows in metrics_data.items():
        if not rows:
            continue

        # Sort by lemma first, then by layer
        sorted_rows = sorted(rows, key=lambda x: (x["lemma"], x["layer"]))

        output_file = model_output_dir / f"{metric_name}.csv"
        with output_file.open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["lemma", "layer", "value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(sorted_rows)

        print(f"\nSaved {metric_name}: {output_file}")
        total_files += 1

    print("\n" + "=" * 50)
    print("METRICS EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {model_output_dir}")
    print(f"Total files: {total_files}")
    print("=" * 50)


def main() -> None:
    """Program entrypoint."""
    args = get_args(__doc__, get_model_arg=True)

    # Get layers from config or use all layers
    layers_to_evaluate = args.config.model.clustering_layers.get(args.model)

    run_evaluate_metrics(
        args.config.paths.dataset_path,
        args.model,
        args.config.paths.metrics_dir,
        layers_to_evaluate,
        args.config.model.random_state,
    )


if __name__ == "__main__":
    main()
