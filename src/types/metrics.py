"""Metrics types."""

from dataclasses import dataclass, field
from typing import Mapping
from collections import defaultdict


@dataclass
class MetricRecord:
    """Result of evaluating a single metric."""

    lemma: str
    layer: int
    metric: str
    value: float


@dataclass
class LayerMetrics:
    """Metrics for all labels within a specific layer."""

    layer: int
    labels: Mapping[str, MetricRecord] = field(default_factory=defaultdict)


@dataclass
class LemmaMetrics:
    """All metrics for a lemma, organized by layer and metric."""

    lemma: str
    layers: Mapping[int, LayerMetrics] = field(default_factory=defaultdict)


__all__ = ["MetricRecord", "LayerMetrics", "LemmaMetrics"]
