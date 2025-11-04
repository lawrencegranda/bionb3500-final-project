"""Utilities for validating and cleaning sentences."""

from .sentences import (
    clean_sentence,
    is_valid_sentence,
    SentenceRecord,
)
from .senses import (
    SenseDefinition,
    SenseMap,
    SenseMapType,
    GlossMapType,
)

__all__ = [
    "clean_sentence",
    "is_valid_sentence",
    "SenseDefinition",
    "SentenceRecord",
    "SenseMap",
    "SenseMapType",
    "GlossMapType",
]
