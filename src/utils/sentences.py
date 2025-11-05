"""Utilities for validating and cleaning sentences."""

from dataclasses import dataclass
import re


@dataclass
class SentenceRecord:
    """Normalized representation of a cleaned and labeled sentence."""

    lemma: str  # the target lemma
    text: str  # the cleaned sentence
    synset: str  # the WordNet synset of the target lemma
    source: str  # the source of the sentence


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
