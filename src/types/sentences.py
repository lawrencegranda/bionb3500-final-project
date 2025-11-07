"""Sentences types."""

from dataclasses import dataclass


@dataclass
class SentenceRecord:
    """Normalized representation of a cleaned and labeled sentence."""

    lemma: str  # the target lemma
    text: str  # the cleaned sentence
    label: str  # the label of the target lemma
    synset: str  # the WordNet synset of the target lemma
    source: str  # the source of the sentence


__all__ = ["SentenceRecord"]
