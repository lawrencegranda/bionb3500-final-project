"""Utilities for validating and cleaning sentences."""

from .sentences import (
    clean_sentence,
)
from .senses import SenseMap

__all__ = [
    "clean_sentence",
    "SenseMap",
]
