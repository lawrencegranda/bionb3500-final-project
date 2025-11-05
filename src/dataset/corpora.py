"""
Load corpora from UFSAC-style XML files and convert them to SenseDefinition objects.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, Set, Tuple, Sequence
from pathlib import Path
from xml.etree import ElementTree as ET
from nltk.corpus import wordnet as wn  # pylint: disable=E0401,C0413
from nltk.corpus.reader.wordnet import WordNetError  # pylint: disable=E0401

from src.utils import SenseType, SentenceRecord, clean_sentence, SenseMap

logger = logging.getLogger(__name__)


@dataclass
class Corpora:
    """
    Corpus loader that collects sentences from UFSAC-style XML files
    and converts them to ``SentenceRecord`` objects grouped by label.

    Args:
        paths: Set of paths to UFSAC-style XML files.
        sense_map: SenseMap object containing lemma -> label -> synsets mapping.
        sentences: Mapping from lemma to label to sequence of sentence records.
    """

    paths: Set[Path]
    sense_map: SenseMap
    sentences: Dict[str, Dict[str, Sequence[SentenceRecord]]]

    def __init__(
        self,
        wn_key_type: str,
        paths: Sequence[str | Path],
        sense_map: SenseMap,
    ) -> None:
        self.paths = set(Path(path) for path in paths)
        self.sense_map = sense_map
        self.sentences = _load_sentences(wn_key_type, self.paths, self.sense_map)

    def __iter__(self) -> Iterator[SentenceRecord]:
        """Iterate over all sentence records across all lemmas and labels."""
        for label_to_sentences in self.sentences.values():
            for sentence_list in label_to_sentences.values():
                yield from sentence_list

    def __len__(self) -> int:
        """Return total count of sentences across all lemmas and labels."""
        return sum(
            len(sentence_list)
            for label_to_sentences in self.sentences.values()
            for sentence_list in label_to_sentences.values()
        )

    def get_sentences(self, lemma: str, label: str) -> Sequence[SentenceRecord]:
        """Get sentences for a specific lemma and label."""
        return self.sentences.get(lemma, {}).get(label, [])


def _parse_xml_root(path: Path):
    """Parse the XML root of a UFSAC-style XML file."""
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        logger.error("Failed to parse XML corpus %s: %s", path, exc)

    return None


def _process_sentence(
    wn_key_type: str, sentence: ET.Element, sense_map: SenseMap
) -> Tuple[Sequence[str], Dict[str, Dict[str, SenseType]]]:
    """
    Process a sentence and return the words and synsets found for each lemma-label pair.

    Returns:
        Tuple containing:
        - words: sequence of words in the sentence
        - Dict mapping lemma -> label -> synset found
    """

    words: Sequence[str] = []

    # For each lemma, track (label, synset) pairs found
    lemma_synsets: Dict[str, Sequence[Tuple[str, SenseType]]] = defaultdict(list)

    for word in sentence.findall("word"):
        w: str = word.attrib.get("surface_form", "").lower()

        # Skip non-alphanumeric tokens
        if not w or not all(c.isalnum() or c == "_" for c in w):
            continue

        # Add the word to the list of words
        words.append(w)

        # Add the sense key to the list of sense keys
        wn_key: str = word.attrib.get(wn_key_type, "")
        if not wn_key:
            continue
        wn_key = wn_key.split(";")[0]

        try:
            synset: SenseType = wn.synset_from_sense_key(wn_key)
        except WordNetError:
            continue

        # Check if this synset is in the sense map for this w (lemma)
        if w in sense_map.sense_map:
            for label, synsets in sense_map.sense_map[w].items():
                if synset in synsets:
                    lemma_synsets[w].append((label, synset))

    # Filter to only lemmas with exactly one synset found
    result: Dict[str, Dict[str, SenseType]] = {}
    for lemma, label_synset_list in lemma_synsets.items():
        if len(label_synset_list) == 1:
            label, synset = label_synset_list[0]
            result[lemma] = {label: synset}

    return words, result


def _load_sentences(
    wn_key_type: str, paths: set[Path], sense_map: SenseMap
) -> Dict[str, Dict[str, Sequence[SentenceRecord]]]:
    """
    Load sentences from XML files and group them by lemma and label.

    Returns:
        Dict mapping lemma -> label -> list of sentence records
    """
    # Initialize nested structure
    sentences: Dict[str, Dict[str, Sequence[SentenceRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path in sorted(paths):
        if not path.exists():
            logger.warning("Corpus file %s does not exist; skipping", path)
            continue

        # Parse the XML root of the corpus file
        root = _parse_xml_root(path)
        if root is None:
            logger.error("Failed to parse XML corpus %s", path)
            continue

        for sentence in root.iterfind(".//sentence"):
            words, lemma_label_synsets = _process_sentence(
                wn_key_type, sentence, sense_map
            )

            # Process each lemma-label-synset found in the sentence
            for lemma, label_to_synset in lemma_label_synsets.items():
                for label, synset in label_to_synset.items():
                    text: str = clean_sentence(" ".join(words))

                    # Add the sentence to the appropriate lemma-label bucket
                    sentences[lemma][label].append(
                        SentenceRecord(
                            lemma=lemma,
                            text=text,
                            synset=synset.name(),
                            source=path.name,
                        )
                    )

    return sentences
