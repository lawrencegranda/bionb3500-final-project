"""Utilities for validating and cleaning sentences."""

from dataclasses import dataclass
import re
from typing import Sequence

_VALID_SENTENCE_PATTERN = re.compile(r"^(?!\\s)([a-z0-9]+)(\\s[a-z0-9]+){0,18}$")


@dataclass
class SentenceRecord:
    """Normalized representation of a cleaned and labeled sentence."""

    lemma: str  # the target lemma
    text: str  # the cleaned sentence
    tokens: Sequence[str]  # the tokens in the sentence
    synset: str  # the WordNet synset of the target lemma
    source: str  # the source of the sentence


def clean_sentence(sentence: str) -> str:
    """Lowercase a sentence, strip punctuation, and collapse repeated spaces."""

    lowered = sentence.lower()
    stripped = re.sub(r"[^a-z0-9\\s]", " ", lowered)
    collapsed = re.sub(r"\\s+", " ", stripped).strip()
    return collapsed


def is_valid_sentence(cleaned: str, lemma: str) -> bool:
    """Check formatting constraints and ensure the lemma appears exactly once."""

    if not _VALID_SENTENCE_PATTERN.match(cleaned):
        return False
    tokens = cleaned.split()
    return tokens.count(lemma) == 1
