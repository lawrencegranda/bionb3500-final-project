"""Evaluate sense separation metrics across layers."""

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

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


def _cluster_random(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """
    Baseline: Random clustering (worst-case scenario).
    """
    n_samples = len(embeddings)

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)
    cluster_ids = rng.integers(0, n_clusters, size=n_samples)

    return cluster_ids


def _cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """
    Perform KMeans clustering and map cluster IDs to sense labels via majority voting.
    NOTE: This gives KMeans an advantage by providing the exact number of clusters.
    """
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    return cluster_ids


def _cluster_gmm(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """
    Gaussian Mixture Model clustering.
    """
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


def _cluster_kmeans_plus_plus(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """
    Perform KMeans++ clustering and map cluster IDs to sense labels via majority voting.
    KMeans++ controls initialization, but otherwise equivalent to standard KMeans.
    """
    kmeans = KMeans(
        n_clusters=n_clusters, init="k-means++", random_state=random_state, n_init=10
    )
    cluster_ids = kmeans.fit_predict(embeddings)
    return cluster_ids


def _cluster_agglomerative(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,  # pylint: disable=unused-argument
) -> np.ndarray:
    """
    Perform Agglomerative Clustering.
    Note: AgglomerativeClustering does not support random_state.
    """
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    )
    cluster_ids = agg.fit_predict(embeddings)
    return cluster_ids


def _cluster_spectral(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """
    Perform Spectral Clustering.
    """
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        random_state=random_state,
        affinity="nearest_neighbors",
        n_init=10,
        assign_labels="kmeans",
    )
    cluster_ids = spectral.fit_predict(embeddings)
    return cluster_ids


_clustering_methods = {
    "random": _cluster_random,
    "kmeans": _cluster_kmeans,
    "kmeans++": _cluster_kmeans_plus_plus,
    "gmm": _cluster_gmm,
    "agglomerative": _cluster_agglomerative,
    "spectral": _cluster_spectral,
}


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

    if clustering_method not in _clustering_methods:
        raise ValueError(f"Unknown clustering method: {clustering_method}")

    cluster_func = _clustering_methods[clustering_method]

    # Perform clustering and assign labels via majority voting
    # This maps cluster IDs to actual sense labels for meaningful comparison
    failed = False

    try:
        n_clusters = len(set(true_labels))
        predicted_labels = cluster_func(embeddings, n_clusters, random_state)
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
