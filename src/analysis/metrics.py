"""Evaluate sense separation metrics across layers."""

from abc import ABC, abstractmethod
from typing import Mapping, Sequence
from collections import Counter

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from scipy.spatial.distance import cdist

from src.types.metrics import (
    LemmaMetrics,
    LayerMetrics,
    MetricRecord,
)

from src.types.embeddings import LayerEmbeddings
from src.dataset import Database


class MetricModel(ABC):  # pylint: disable=R0903
    """Base class for clustering evaluation metrics."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> float:
        """Compute metric score for embeddings, predicted labels, and true labels."""
        raise NotImplementedError


class SilhouetteMetric(MetricModel):  # pylint: disable=R0903
    """Silhouette coefficient metric."""

    def __init__(self):
        self.name = "silhouette"

    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> float:
        """Compute silhouette score."""
        return float(silhouette_score(embeddings, pred_labels, metric="cosine"))


class AdjustedRandMetric(MetricModel):  # pylint: disable=R0903
    """Adjusted Rand Index metric."""

    def __init__(self):
        self.name = "adjusted_rand"

    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> float:
        """Compute adjusted rand index."""
        return float(adjusted_rand_score(true_labels, pred_labels))


class NormalizedMutualInfoMetric(MetricModel):  # pylint: disable=R0903
    """Normalized Mutual Information metric."""

    def __init__(self):
        self.name = "normalized_mutual_info"

    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> float:
        """Compute normalized mutual information."""
        return float(normalized_mutual_info_score(true_labels, pred_labels))


class DaviesBouldinMetric(MetricModel):  # pylint: disable=R0903
    """Davies-Bouldin Index metric."""

    def __init__(self):
        self.name = "davies_bouldin"

    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> float:
        """Compute Davies-Bouldin index."""
        return float(davies_bouldin_score(embeddings, pred_labels))


class CalinskiHarabaszMetric(MetricModel):  # pylint: disable=R0903
    """Calinski-Harabasz Index metric."""

    def __init__(self):
        self.name = "calinski_harabasz"

    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> float:
        """Compute Calinski-Harabasz index."""
        return float(calinski_harabasz_score(embeddings, pred_labels))


def _cluster_kmeans(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Perform KMeans clustering and map cluster IDs to sense labels via majority voting.
    NOTE: This gives KMeans an advantage by providing the exact number of clusters.
    """
    n_clusters = len(set(true_labels))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    return cluster_ids


def _cluster_random(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Baseline: Random clustering (worst-case scenario).
    """
    n_samples = len(embeddings)
    n_clusters = len(set(true_labels))

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)
    cluster_ids = rng.integers(0, n_clusters, size=n_samples)

    return cluster_ids


def _cluster_hdbscan(
    embeddings: np.ndarray,
    true_labels: np.ndarray,  # pylint: disable=W0613
    random_state: int,  # pylint: disable=W0613
) -> np.ndarray:
    """
    HDBSCAN clustering - does NOT require knowing n_clusters.
    More realistic evaluation of unsupervised clustering.

    Note: HDBSCAN is deterministic and doesn't use random_state.
    """
    # HDBSCAN automatically determines the number of clusters
    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1)
    cluster_ids = clusterer.fit_predict(embeddings)

    # HDBSCAN uses -1 for noise points; assign them to nearest cluster
    if np.all(cluster_ids == -1):
        raise ValueError("No clusters found")
    elif -1 in cluster_ids:
        embeddings_non_noise = embeddings[cluster_ids != -1]
        embeddings_noise = embeddings[cluster_ids == -1]
        cluster_labels_non_noise = cluster_ids[cluster_ids != -1]
        centroids = np.array(
            [
                embeddings_non_noise[cluster_labels_non_noise == cid].mean(axis=0)
                for cid in np.unique(cluster_labels_non_noise)
            ]
        )
        unique_cluster_ids = np.unique(cluster_labels_non_noise)
        dists = cdist(embeddings_noise, centroids)
        nearest = np.argmin(dists, axis=1)
        cluster_ids_new = cluster_ids.copy()
        cluster_ids_new[cluster_ids == -1] = unique_cluster_ids[nearest]
        cluster_ids = cluster_ids_new

    return cluster_ids


def _cluster_gmm(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Gaussian Mixture Model clustering.
    NOTE: Like KMeans, this gives GMM the exact number of components.
    Use with caution for evaluation.
    """
    n_clusters = len(set(true_labels))

    # Convert to float64 for better numerical accuracy
    embeddings_64 = embeddings.astype(np.float64)

    # Add regularization to prevent singular covariance matrices
    gmm = GaussianMixture(
        n_components=n_clusters,
        random_state=random_state,
        covariance_type="full",
        n_init=10,
        reg_covar=1e-6,  # Add regularization for numerical stability
        max_iter=100,
    )
    cluster_ids = gmm.fit_predict(embeddings_64)

    return cluster_ids


def _map_clusters_to_labels(
    cluster_ids: np.ndarray,
    true_labels: np.ndarray,
) -> np.ndarray:
    """
    Map cluster IDs to sense labels via majority voting.
    This allows comparing different cluster assignments to ground truth.
    """
    unique_clusters = set(cluster_ids)
    cluster_to_label = {}

    for cluster_id in unique_clusters:
        # Get indices of all samples in this cluster
        cluster_mask = cluster_ids == cluster_id

        # Get the true labels for these samples
        labels_in_cluster = true_labels[cluster_mask]

        if len(labels_in_cluster) > 0:
            # Find the most common label in this cluster
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
            cluster_to_label[cluster_id] = most_common_label
        else:
            # If cluster is empty, map to a placeholder (shouldn't happen)
            cluster_to_label[cluster_id] = true_labels[0]

    # Map cluster IDs to actual sense labels
    predicted_labels = np.array([cluster_to_label[cid] for cid in cluster_ids])

    return predicted_labels


def make_metrics(
    database: Database,
    metric_classes: Sequence[type[MetricModel]],
    random_state: int,
    clustering_method: str,
) -> Mapping[str, LemmaMetrics]:
    """
    Return a dictionary of lemma -> LemmaMetrics after computing metrics.
    If gold sense labels are available, pass them as true_labels for ARI/NMI.
    """
    result: Mapping[str, LemmaMetrics] = {}

    embeddings = database.embeddings_table.get_embeddings()

    for lemma, lemma_embeddings in embeddings.items():
        if lemma not in result:
            result[lemma] = LemmaMetrics(lemma=lemma)

        for layer, layer_embeddings in lemma_embeddings.layers.items():
            result[lemma].layers[layer] = _compute_layer_metrics(
                lemma,
                layer,
                layer_embeddings,
                metric_classes,
                random_state,
                clustering_method,
            )

    return result


def _compute_layer_metrics(
    lemma: str,
    layer: int,
    layer_embeddings: LayerEmbeddings,
    metric_classes: Sequence[type[MetricModel]],
    random_state: int,
    clustering_method: str,
) -> LayerMetrics:
    """
    Compute all metrics for a single layer.
    Returns a LayerMetrics object with metric name -> MetricRecord mapping.

    For ARI and NMI:
    - True labels = sense labels (ground truth from the data)
    - Predicted labels = Clustering algorithm results
    """
    result: LayerMetrics = LayerMetrics(layer=layer)

    # Flatten the embeddings into arrays
    recs = [
        (label_emb.label, rec.embedding)
        for label_emb in layer_embeddings.labels.values()
        for rec in label_emb.records
    ]

    if len(recs) < 2:
        # Not enough data to compute metrics
        return result

    # Extract true labels (sense labels) and embeddings
    true_labels = np.array([rec[0] for rec in recs])
    embeddings = np.array([rec[1] for rec in recs])

    # Select clustering method
    clustering_methods = {
        "kmeans": _cluster_kmeans,
        "random": _cluster_random,
        "hdbscan": _cluster_hdbscan,
        "gmm": _cluster_gmm,
    }

    if clustering_method not in clustering_methods:
        raise ValueError(f"Unknown clustering method: {clustering_method}")

    cluster_func = clustering_methods[clustering_method]

    # Perform clustering and assign labels via majority voting
    # This maps cluster IDs to actual sense labels for meaningful comparison
    failed = False

    try:
        predicted_labels = cluster_func(embeddings, true_labels, random_state)
        if len(np.unique(predicted_labels)) < 2:
            raise ValueError(
                f"No clusters found for lemma {lemma}, layer {layer}, "
                f"clustering method {clustering_method}"
            )
    except ValueError as e:
        print(f"Warning: {e}")
        failed = True
        predicted_labels = np.zeros(len(embeddings))

    # Compute each metric
    for metric_class in metric_classes:
        metric = metric_class()

        # All metrics now use predicted labels for consistent evaluation
        if failed:
            value = np.nan
        elif metric.name in ("adjusted_rand", "normalized_mutual_info"):
            value = metric.compute(
                embeddings,
                true_labels,
                predicted_labels,
            )
        else:
            value = metric.compute(embeddings, true_labels, predicted_labels)

        metric_record = MetricRecord(
            lemma=lemma,
            layer=layer,
            metric=metric.name,
            value=value,
        )

        result.labels[metric.name] = metric_record

    return result


metric_models: Mapping[str, type[MetricModel]] = {
    "silhouette": SilhouetteMetric,
    "adjusted_rand": AdjustedRandMetric,
    "normalized_mutual_info": NormalizedMutualInfoMetric,
    "davies_bouldin": DaviesBouldinMetric,
    "calinski_harabasz": CalinskiHarabaszMetric,
}


__all__ = [
    "metric_models",
    "make_metrics",
]
