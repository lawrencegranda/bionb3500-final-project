"""Utilities for validating and cleaning sentences."""

import re


def clean_sentence(sentence: str) -> str:
    """
    Normalize a sentence by lowercasing, replacing underscores with spaces,
    and stripping punctuation.
    """

    lowered = sentence.lower()
    normalized = lowered.replace("_", " ")
    stripped = re.sub(r"[^a-z0-9\\s]", " ", normalized)
    collapsed = re.sub(r"\\s+", " ", stripped).strip()

    return collapsed
