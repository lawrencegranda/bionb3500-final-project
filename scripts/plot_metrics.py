#!/usr/bin/env python3
"""Plot metrics across layers for each lemma and metric combination."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.helpers import get_args  # pylint: disable=C0413, E0401


def load_metric_data(
    metrics_dir: Path,
    model_name: str,
    clustering_method: str,
    metric_name: str,
) -> Dict[str, Dict[int, float]]:
    """
    Load metric data from CSV file.

    Returns:
        Dictionary mapping lemma -> layer -> median_value
    """
    csv_path = metrics_dir / model_name / clustering_method / f"{metric_name}.csv"

    if not csv_path.exists():
        print(f"Warning: File not found: {csv_path}")
        return {}

    data = defaultdict(dict)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = row["lemma"]
            layer = int(row["layer"])
            median = float(row["median"]) if row["median"] != "nan" else np.nan
            data[lemma][layer] = median

    return data


def get_all_lemmas_and_metrics(metrics_dir: Path) -> Tuple[List[str], List[str]]:
    """Get all unique lemmas and metrics from the first available CSV."""
    # Try to find any CSV file
    for model_dir in metrics_dir.iterdir():
        if model_dir.is_dir():
            for method_dir in model_dir.iterdir():
                if method_dir.is_dir():
                    for csv_file in method_dir.glob("*.csv"):
                        # Read this CSV to get lemmas
                        lemmas = set()
                        with csv_file.open("r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                lemmas.add(row["lemma"])

                        # Get all metric names from this directory
                        metrics = [f.stem for f in method_dir.glob("*.csv")]

                        return sorted(lemmas), sorted(metrics)

    return [], []


def plot_metric_by_lemma(
    metrics_dir: Path,
    output_dir: Path,
    model_names: List[str],
    clustering_methods: List[str],
    lemma: str,
    metric_name: str,
) -> None:
    """
    Create a plot for a specific lemma and metric.

    Plot shows median values across layers for each model and clustering method.
    Both models span from 0% to 100% on the x-axis (normalized by model depth).
    Layer numbers are shown on dual axes (BERT on bottom, DistilBERT on top).
    """
    # Define colors for models
    model_colors = {
        "bert-base-uncased": "#1f77b4",  # Blue
        "distilbert-base-uncased": "#ff7f0e",  # Orange
    }

    # Define linestyles for clustering methods
    method_linestyles = {
        "kmeans": "-",
        "hdbscan": "-",
        "gmm": "-",
        "random": "--",  # Dashed for random
    }

    # Define markers for clustering methods
    method_markers = {
        "kmeans": "o",
        "hdbscan": "s",
        "gmm": "^",
        "random": "x",
    }

    _fig, ax = plt.subplots(figsize=(12, 6))

    # Collect layer information for each model
    model_layers = {}
    for model_name in model_names:
        # Get layers from first available clustering method
        for clustering_method in clustering_methods:
            data = load_metric_data(
                metrics_dir, model_name, clustering_method, metric_name
            )
            if lemma in data and data[lemma]:
                model_layers[model_name] = sorted(data[lemma].keys())
                break

    # Plot each combination of model and clustering method
    for model_name in model_names:
        if model_name not in model_layers:
            continue

        layers = model_layers[model_name]
        n_layers = len(layers)

        # Create percentage-based x positions (0% to 100%)
        if n_layers > 1:
            x_positions = np.array([i / (n_layers - 1) * 100 for i in range(n_layers)])
        else:
            x_positions = np.array([50.0])  # Single layer at 50%

        for clustering_method in clustering_methods:
            data = load_metric_data(
                metrics_dir, model_name, clustering_method, metric_name
            )

            if lemma not in data or not data[lemma]:
                continue

            # Extract values in layer order
            values = [data[lemma].get(layer, np.nan) for layer in layers]

            # Skip if all values are NaN
            if all(np.isnan(v) for v in values):
                continue

            # Get style parameters
            color = model_colors.get(model_name, "#333333")
            linestyle = method_linestyles.get(clustering_method, "-")
            marker = method_markers.get(clustering_method, "o")

            # Create label
            model_short = "BERT" if "bert-base" in model_name else "DistilBERT"
            label = f"{model_short} - {clustering_method}"

            # Plot with percentage-based positions
            ax.plot(
                x_positions,
                values,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=8,
                linewidth=2.5,
                alpha=0.8,
            )

    # Set x-axis to percentage scale
    ax.set_xlim(-5, 105)
    ax.set_xlabel("Model Depth (%)", fontsize=12, fontweight="bold")

    # Create custom x-axis showing both models' layers
    if len(model_layers) == 2:
        # Assume we have both BERT and DistilBERT
        bert_model = (
            [m for m in model_names if "bert-base-uncased" in m][0]
            if any("bert-base-uncased" in m for m in model_names)
            else None
        )
        distilbert_model = (
            [m for m in model_names if "distilbert" in m][0]
            if any("distilbert" in m for m in model_names)
            else None
        )

        if bert_model and distilbert_model:
            bert_layers = model_layers[bert_model]
            distil_layers = model_layers[distilbert_model]

            # Calculate percentage positions for each model's layers
            n_bert = len(bert_layers)
            n_distil = len(distil_layers)

            if n_bert > 1:
                bert_positions = [i / (n_bert - 1) * 100 for i in range(n_bert)]
            else:
                bert_positions = [50.0]

            if n_distil > 1:
                distil_positions = [i / (n_distil - 1) * 100 for i in range(n_distil)]
            else:
                distil_positions = [50.0]

            # Set primary x-axis ticks at BERT positions (bottom)
            ax.set_xticks(bert_positions)
            ax.set_xticklabels([str(l) for l in bert_layers])
            ax.tick_params(
                axis="x", colors=model_colors["bert-base-uncased"], labelsize=10
            )
            ax.set_xlabel(
                "BERT Layers",
                fontsize=11,
                fontweight="bold",
                color=model_colors["bert-base-uncased"],
            )

            # Create secondary x-axis for DistilBERT (top)
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(distil_positions)
            ax2.set_xticklabels([str(l) for l in distil_layers])
            ax2.tick_params(
                axis="x", colors=model_colors["distilbert-base-uncased"], labelsize=10
            )
            ax2.set_xlabel(
                "DistilBERT Layers",
                fontsize=11,
                fontweight="bold",
                color=model_colors["distilbert-base-uncased"],
            )
    else:
        # Single model: show layer labels at their percentage positions
        model_name = list(model_layers.keys())[0]
        layers = model_layers[model_name]
        n_layers = len(layers)

        if n_layers > 1:
            positions = [i / (n_layers - 1) * 100 for i in range(n_layers)]
        else:
            positions = [50.0]

        ax.set_xticks(positions)
        ax.set_xticklabels([str(l) for l in layers])
        ax.set_xlabel("Layer (% through model)", fontsize=12, fontweight="bold")

    # Formatting
    ax.set_ylabel(
        f"{metric_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        f"{lemma.capitalize()} - {metric_name.replace('_', ' ').title()}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.legend(loc="best", fontsize=9, ncol=2)

    plt.tight_layout()

    # Save plot
    output_file = output_dir / f"{lemma}_{metric_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_file}")


def main() -> None:
    """Program entrypoint."""
    args = get_args(__doc__)

    metrics_dir = args.config.paths.metrics_dir
    model_names = args.config.model.model_names
    clustering_methods = args.config.model.clustering_methods

    # Create output directory for metric plots
    output_dir = Path("results/plots/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" " * 20 + "PLOTTING METRICS")
    print("=" * 70)
    print(f"Metrics directory: {metrics_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Clustering methods: {', '.join(clustering_methods)}")
    print("=" * 70)
    print()

    # Get all lemmas and metrics
    lemmas, metrics = get_all_lemmas_and_metrics(metrics_dir)

    if not lemmas or not metrics:
        print("Error: No data found in metrics directory")
        sys.exit(1)

    print(f"Found lemmas: {', '.join(lemmas)}")
    print(f"Found metrics: {', '.join(metrics)}")
    print()

    # Generate plots
    total_plots = len(lemmas) * len(metrics)
    current_plot = 0

    for lemma in lemmas:
        print(f"\nProcessing lemma: {lemma}")
        for metric_name in metrics:
            current_plot += 1
            print(f"  [{current_plot}/{total_plots}] Plotting {metric_name}...")

            plot_metric_by_lemma(
                metrics_dir,
                output_dir,
                model_names,
                clustering_methods,
                lemma,
                metric_name,
            )

    print("\n" + "=" * 70)
    print(" " * 20 + "PLOTTING COMPLETE")
    print("=" * 70)
    print(f"Generated {total_plots} plots in {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
