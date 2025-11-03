"""Utilities for validating and cleaning sentences."""

from .sentences import (
    clean_sentence,
    is_valid_sentence,
    SenseDefinition,
    SentenceRecord,
)

__all__ = [
    "clean_sentence",
    "is_valid_sentence",
    "SenseDefinition",
    "SentenceRecord",
]
