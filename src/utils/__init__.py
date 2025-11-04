"""Utilities for validating and cleaning sentences."""

from .sentences import (
    clean_sentence,
    is_valid_sentence,
    SentenceRecord,
)
from .senses import (
    SenseDefinition,
    build_sense_map,
    synset_to_label,
    sense_map_to_json,
    SenseMap,
    GlossMap,
)

__all__ = [
    "clean_sentence",
    "is_valid_sentence",
    "build_sense_map",
    "synset_to_label",
    "sense_map_to_json",
    "SenseDefinition",
    "SentenceRecord",
    "SenseMap",
    "GlossMap",
]
