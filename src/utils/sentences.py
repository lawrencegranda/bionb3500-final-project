"""Utilities for validating and cleaning sentences."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

_VALID_SENTENCE_PATTERN = re.compile(r"^(?!\\s)([a-z0-9]+)(\\s[a-z0-9]+){0,18}$")


@dataclass(frozen=True)
class SenseDefinition:
    """Canonical information for a sense of a given lemma."""

    label: str
    wn_key: str


@dataclass
class SentenceRecord:
    """Normalized representation of a cleaned and labeled sentence."""

    lemma: str
    text: str
    tokens: Sequence[str]
    wn_key: str
    sense_label: str
    source: str

    @property
    def split_index(self) -> int:
        """Return the token index of the target lemma within the sentence."""

        return self.tokens.index(self.lemma)


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
