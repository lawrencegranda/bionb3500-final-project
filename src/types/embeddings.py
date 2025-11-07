"""Embeddings types."""

from dataclasses import dataclass, field
from typing import Mapping, Sequence
from collections import defaultdict
import numpy as np


@dataclass
class EmbeddingRecord:
    """Record of a single embedding."""

    sentence_id: int
    layer: int
    embedding: np.ndarray


@dataclass
class LabelEmbeddings:
    """Group of embeddings for a given label within a layer."""

    label: str
    records: Sequence[EmbeddingRecord] = field(default_factory=list)


@dataclass
class LayerEmbeddings:
    """Embeddings for all labels within a specific layer."""

    layer: int
    labels: Mapping[str, LabelEmbeddings] = field(default_factory=defaultdict)


@dataclass
class LemmaEmbeddings:
    """All embeddings for a lemma, organized by layer and label."""

    lemma: str
    layers: Mapping[int, LayerEmbeddings] = field(default_factory=defaultdict)


__all__ = ["EmbeddingRecord", "LabelEmbeddings", "LayerEmbeddings", "LemmaEmbeddings"]
