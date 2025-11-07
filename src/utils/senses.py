"""Utilities to map glosses/labels to candidate wordNet sense keys for a given lemma."""

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Optional, Sequence, Set, Tuple

from nltk.corpus import wordnet as wn

from src.types.senses import SenseType, SenseMapType, GlossMapType


def _synsets_by_glosses(lemma: str, gloss_set: Set[str]) -> Set[SenseType]:
    """
    Pick a synset for a lemma by a set of substrings in its definition.

    - `lemma`: the target lemma.
    - `gloss_set`: the set of substrings in the definition to match.

    Returns:
        Set[SenseType]: Set of synsets matching the glosses.
    """
    synsets = set()

    for ss in wn.synsets(lemma, pos=wn.NOUN):
        for gloss in gloss_set:
            if gloss in ss.definition().lower():
                synsets.add(ss)

    if not synsets:
        raise LookupError(f"No synset for {lemma} containing: {gloss_set!r}")

    return synsets


@dataclass(init=False, slots=True)
class SenseMap:
    """Wrapper around a ``lemma -> label -> synset`` mapping."""

    _sense_map: SenseMapType

    def __init__(self, sense_map: SenseMapType) -> None:
        self._sense_map = sense_map

    @classmethod
    def from_gloss(cls, gloss_map: GlossMapType) -> "SenseMap":
        """Build a ``SenseMap`` from a ``GlossMap`` definition."""
        result: SenseMapType = defaultdict(lambda: defaultdict(set))

        for lemma, label_to_gloss in gloss_map.items():
            for label, gloss_set in label_to_gloss.items():
                result[lemma][label] = _synsets_by_glosses(lemma, gloss_set)

        return cls(result)

    @classmethod
    def from_dict(cls, serialisable: Dict[str, Dict[str, Sequence[str]]]) -> "SenseMap":
        """Build a ``SenseMap`` from a serialisable mapping of synset names."""
        result: SenseMapType = defaultdict(lambda: defaultdict(set))

        for lemma, label_to_names in serialisable.items():
            for label, synset_names in label_to_names.items():
                synsets = set(wn.synset(s) for s in synset_names)
                result[lemma][label] = synsets

        return cls(result)

    @property
    def sense_map(self) -> SenseMapType:
        """Return the internal sense map."""
        return self._sense_map

    def synset_to_label(self, syn: SenseType) -> Optional[Tuple[str, str]]:
        """
        Return the lemma and label for a synset in the sense map.

        - `syn`: the synset to look up.

        Returns:
            Optional[Tuple[str, str]]: the lemma and label for the synset, or None if not found.
        """
        for lemma, label_to_synsets in self._sense_map.items():
            for label, synsets in label_to_synsets.items():
                if syn in synsets:
                    return (lemma, label)
        return None

    def to_json(self) -> Dict[str, Dict[str, Sequence[str]]]:
        """Convert the SenseMap of synsets into JSON-serialisable synset names."""
        serialisable: Dict[str, Dict[str, Sequence[str]]] = {}
        for lemma, label_to_synsets in self._sense_map.items():
            serialisable[lemma] = {}
            for label, synsets in label_to_synsets.items():
                keys = [synset.name() for synset in synsets]
                serialisable[lemma][label] = sorted(set(keys))
        return serialisable
