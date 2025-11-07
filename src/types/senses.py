"""Senses types."""

from typing import Dict, Set, TypeAlias

import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

SenseType: TypeAlias = wn.synset
SenseMapType: TypeAlias = Dict[str, Dict[str, Set[SenseType]]]
GlossMapType: TypeAlias = Dict[str, Dict[str, Set[str]]]


__all__ = ["SenseType", "SenseMapType", "GlossMapType"]
