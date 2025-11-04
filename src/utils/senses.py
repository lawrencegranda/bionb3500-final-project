"""Utilities to map glosses/labels to candidate WordNet sense keys for a given word."""

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple, TypeAlias, Sequence

import nltk  # pylint: disable=E0401

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet as wn  # pylint: disable=E0401,C0413


Sense = wn.synset
SenseMap: TypeAlias = Dict[str, Dict[str, Set[Sense]]]
GlossMap: TypeAlias = Dict[str, Dict[str, Set[str]]]


@dataclass(frozen=True)
class SenseDefinition:
    """Canonical information for a sense of a given lemma."""

    label: str
    wn_key: str


def _synsets_by_glosses(word: str, gloss_set: Set[str]) -> Set[Sense]:
    """
    Pick a synset for a lemma by a set of substrings in its definition.

    - `word`: the target lemma.
    - `gloss_set`: the set of substrings in the definition to match.

    Returns:
        Set[Sense]: Set of synsets matching the glosses.
    """
    synsets = set()

    for ss in wn.synsets(word, pos=wn.NOUN):
        for gloss in gloss_set:
            if gloss in ss.definition().lower():
                synsets.add(ss)

    if not synsets:
        raise LookupError(f"No synset for {word} containing: {gloss_set!r}")

    return synsets


def build_sense_map(
    gloss_map: Optional[GlossMap],
) -> SenseMap:
    """
    Build a mapping from word -> label -> set of matching WordNet synset keys.

    Returns:
        SenseMap: mapping from word to label to set of wn.Synset keys.
    """
    result: SenseMap = defaultdict(lambda: defaultdict(set))

    for word, label_to_gloss in gloss_map.items():
        for label, gloss_set in label_to_gloss.items():
            result[word][label] = _synsets_by_glosses(word, gloss_set)

    return result


def synset_to_label(syn: Sense, sense_map: SenseMap) -> Optional[Tuple[str, str]]:
    """
    Return the word and label for a synset in the sense map.

    - `syn`: the synset to look up.
    - `sense_map`: the sense map to use.

    Returns:
        Optional[Tuple[str, str]]: the word and label for the synset, or None if not found.
    """
    for word, label_to_synsets in sense_map.items():
        for label, synsets in label_to_synsets.items():
            if syn in synsets:
                return (word, label)

    return None


def sense_map_to_json(sense_map: SenseMap) -> Dict[str, Dict[str, Sequence[str]]]:
    """Convert the ``SenseMap`` of synsets into JSON-serialisable sensekeys.

    - `sense_map`: the sense map to convert.

    Returns:
        Dict[str, Dict[str, Sequence[str]]]: the JSON-serialisable sensemap.
    """

    serialisable: Dict[str, Dict[str, Sequence[str]]] = {}
    for word, label_to_synsets in sense_map.items():
        serialisable[word] = {}
        for label, synsets in label_to_synsets.items():
            keys = [synset.name() for synset in synsets]
            serialisable[word][label] = sorted(set(keys))

    return serialisable
