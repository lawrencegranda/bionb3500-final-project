"""Evaluate sense separation metrics across layers."""

from abc import ABC, abstractmethod
from typing import Mapping, Sequence, Optional
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
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
        pred_labels: Optional[np.ndarray] = None,
    ) -> float:
        """Compute metric score for embeddings, predicted labels, and (if required) true labels."""
        raise NotImplementedError


class SilhouetteMetric(MetricModel):  # pylint: disable=R0903
    """Silhouette coefficient metric."""

    def __init__(self):
        self.name = "silhouette"

    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: Optional[np.ndarray] = None,
    ) -> float:
        """Compute silhouette score."""
        return float(silhouette_score(embeddings, true_labels))


class AdjustedRandMetric(MetricModel):  # pylint: disable=R0903
    """Adjusted Rand Index metric."""

    def __init__(self):
        self.name = "adjusted_rand"

    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: Optional[np.ndarray] = None,
    ) -> float:
        """Compute adjusted rand index (requires true labels)."""
        return float(adjusted_rand_score(true_labels, pred_labels))


class NormalizedMutualInfoMetric(MetricModel):  # pylint: disable=R0903
    """Normalized Mutual Information metric."""

    def __init__(self):
        self.name = "normalized_mutual_info"

    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: Optional[np.ndarray] = None,
    ) -> float:
        """Compute normalized mutual information (requires true labels)."""
        return float(normalized_mutual_info_score(true_labels, pred_labels))


class DaviesBouldinMetric(MetricModel):  # pylint: disable=R0903
    """Davies-Bouldin Index metric."""

    def __init__(self):
        self.name = "davies_bouldin"

    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: Optional[np.ndarray] = None,
    ) -> float:
        """Compute Davies-Bouldin index."""
        return float(davies_bouldin_score(embeddings, true_labels))


class CalinskiHarabaszMetric(MetricModel):  # pylint: disable=R0903
    """Calinski-Harabasz Index metric."""

    def __init__(self):
        self.name = "calinski_harabasz"

    def compute(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: Optional[np.ndarray] = None,
    ) -> float:
        """Compute Calinski-Harabasz index."""
        return float(calinski_harabasz_score(embeddings, true_labels))


def _cluster_and_assign_labels(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Perform KMeans clustering and map cluster IDs to sense labels via majority voting.
    """
    n_clusters = len(set(true_labels))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    # Map each cluster ID to the majority sense label in that cluster
    cluster_to_label = {}
    for cluster_id in range(n_clusters):
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
                lemma, layer, layer_embeddings, metric_classes, random_state
            )

    return result


def _compute_layer_metrics(
    lemma: str,
    layer: int,
    layer_embeddings: LayerEmbeddings,
    metric_classes: Sequence[type[MetricModel]],
    random_state: int,
) -> LayerMetrics:
    """
    Compute all metrics for a single layer.
    Returns a LayerMetrics object with metric name -> MetricRecord mapping.

    For ARI and NMI:
    - True labels = sense labels (ground truth from the data)
    - Predicted labels = KMeans clustering results
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

    # Perform clustering and assign labels via majority voting
    # This maps cluster IDs to actual sense labels for meaningful comparison
    predicted_labels = _cluster_and_assign_labels(embeddings, true_labels, random_state)

    # Compute each metric
    for metric_class in metric_classes:
        metric = metric_class()

        # For supervised metrics (ARI, NMI), use predicted vs. true labels
        if metric.name in ("adjusted_rand", "normalized_mutual_info"):
            value = metric.compute(
                embeddings, true_labels, pred_labels=predicted_labels
            )
        else:
            # For unsupervised metrics, use true labels directly
            value = metric.compute(embeddings, true_labels)

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
