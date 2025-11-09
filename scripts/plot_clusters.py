#!/usr/bin/env python3
"""Cluster embeddings and evaluate sense separation across layers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import Database  # pylint: disable=C0413, E0401
from src.analysis.clustering import (  # pylint: disable=C0413, E0401
    make_clusters,
    reduction_models,
)
from src.types.clusters import LemmaClusters  # pylint: disable=C0413, E0401


def _get_available_layers(
    lemma_clusters: LemmaClusters,
    layers_to_plot: Sequence[int],
) -> Sequence[int]:
    """Return layers present in lemma_clusters from the layers_to_plot list."""
    return [layer for layer in layers_to_plot if layer in lemma_clusters.layers]


def _get_label_colors(
    lemma_clusters: LemmaClusters, layers: Sequence[int]
) -> Mapping[str, str]:
    """Assign a color to each label for plotting."""
    color_palette = ["#FF8C00", "#32CD32", "#1E90FF", "#FF1493", "#9370DB", "#FFD700"]

    # Collect all unique labels for coloring
    label_names = set()
    for layer in layers:
        if layer in lemma_clusters.layers:
            label_names.update(lemma_clusters.layers[layer].labels.keys())

    return {
        label: color_palette[i % len(color_palette)]
        for i, label in enumerate(sorted(label_names))
    }


def _plot_layer_clusters(
    ax,
    layer: int,
    layer_clusters,
    label_colors: dict[str, str],
    method: str,
):
    """Helper to plot data for a single layer on an axis."""

    for label, label_cluster in layer_clusters.labels.items():
        coords = [rec.coords for rec in label_cluster.records]
        if coords:
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            ax.scatter(
                x_coords,
                y_coords,
                c=label_colors[label],
                label=label,
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
        else:
            print(f"[DEBUG]   No coordinates to plot for label '{label}'")
    ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} 1", fontsize=10)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)


def visualize_lemma_clusters(
    lemma: str,
    lemma_clusters: LemmaClusters,
    method: str,
    layers_to_plot: Sequence[int],
    output_dir: Path,
) -> None:
    """
    Visualize clusters for a single lemma using one method.
    Creates one figure with subplots for each layer.
    """
    available_layers = _get_available_layers(lemma_clusters, layers_to_plot)
    if not available_layers:
        print(f"  {lemma}: No data in specified layers")
        return

    n_plots = len(available_layers)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Get label colors
    label_colors = _get_label_colors(lemma_clusters, available_layers)

    # Plot each layer
    for idx, layer in enumerate(available_layers):
        ax = axes[idx]
        layer_clusters = lemma_clusters.layers[layer]
        _plot_layer_clusters(ax, layer, layer_clusters, label_colors, method)

    fig.suptitle(
        f"{lemma.capitalize()} - {method.upper()} Visualization",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    output_path = output_dir / f"{lemma}_{method}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  {lemma}: Saved {output_path.name}")


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

    dataset_path = Path(data_config.get("dataset_path"))

    if not dataset_path.exists():
        print(f"Error: Database file not found at {dataset_path}")
        sys.exit(1)

    # Get model name from config
    model_name = data_config.get("model_name", "bert-base-uncased")

    # Create output directory
    output_dir = Path(data_config.get("plots_dir")) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get random state from config
    random_state = data_config.get("random_state", 42)

    # Determine layers to plot
    layers_to_plot: Sequence[int] = data_config.get("clustering_layers")

    print("=" * 50)
    print("CLUSTERING VISUALIZATION")
    print("=" * 50)
    print(f"Database: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"Layers to plot: {layers_to_plot}")
    print(f"Output directory: {output_dir}")
    print(f"Random state: {random_state}")
    print()

    # Load database
    database = Database.from_db(dataset_path, model_name)

    # Process each dimensionality reduction method
    for method_name, model_class in reduction_models.items():
        print(f"\nProcessing method: {method_name.upper()}")
        print("-" * 50)

        # Generate clusters for all lemmas
        clusters_by_lemma = make_clusters(
            database, model_class, random_state=random_state
        )

        # Visualize each lemma
        for lemma, lemma_clusters in clusters_by_lemma.items():
            print(f"  Visualizing {lemma}...")
            visualize_lemma_clusters(
                lemma, lemma_clusters, method_name, layers_to_plot, output_dir
            )

    database.close()

    print("\n" + "=" * 50)
    print("CLUSTERING VISUALIZATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
