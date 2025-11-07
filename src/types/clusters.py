"""Cluster types."""

from dataclasses import dataclass, field
from typing import Mapping, Sequence
from collections import defaultdict
import numpy as np


@dataclass
class ClusterRecord:
    """Result of clustering a single lemma."""

    sentence_id: int
    layer: int
    lemma: str
    label: str
    coords: np.ndarray


@dataclass
class LabelClusters:
    """Group of clusters for a given label within a layer."""

    label: str
    records: Sequence[ClusterRecord] = field(default_factory=list)


@dataclass
class LayerClusters:
    """Clusters for all labels within a specific layer."""

    layer: int
    labels: Mapping[str, LabelClusters] = field(default_factory=defaultdict)


@dataclass
class LemmaClusters:
    """All clusters for a lemma, organized by layer and label."""

    lemma: str
    layers: Mapping[int, LayerClusters] = field(default_factory=defaultdict)


__all__ = ["ClusterRecord", "LabelClusters", "LayerClusters", "LemmaClusters"]
