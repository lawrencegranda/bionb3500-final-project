"""
Load corpora from UFSAC-style XML files and convert them to SenseDefinition objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Set, Tuple, Sequence
from pathlib import Path
from xml.etree import ElementTree as ET
from nltk.corpus import wordnet as wn  # pylint: disable=E0401,C0413
from nltk.corpus.reader.wordnet import WordNetError  # pylint: disable=E0401

from src.utils import SenseType, SentenceRecord

logger = logging.getLogger(__name__)


@dataclass
class Corpora:
    """
    Corpus loader that collects sentences from UFSAC-style XML files
    and converts them to ``SentenceRecord`` objects.

    Args:
        paths: Set of paths to UFSAC-style XML files.
        lemma: The target lemma to filter by.
        senses: Set of WordNet synsets to filter by.
    """

    paths: Set[Path]
    sentences: Sequence[SentenceRecord]
    lemma: str
    senses: Set[SenseType]

    def __init__(
        self, paths: Sequence[str | Path], lemma: str, senses: Set[SenseType]
    ) -> None:
        self.paths = set(Path(path) for path in paths)
        self.senses = senses
        self.lemma = lemma
        self.sentences = _load_sentences(self.paths, self.lemma, self.senses)

    def __iter__(self) -> Iterator[SentenceRecord]:
        return iter(self.sentences)

    def __len__(self) -> int:
        return len(self.sentences)


def _parse_xml_root(path: Path):
    """Parse the XML root of a UFSAC-style XML file."""
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        logger.error("Failed to parse XML corpus %s: %s", path, exc)

    return None


def _process_sentence(
    sentence: ET.Element, lemma: str, senses: Set[SenseType]
) -> Tuple[Sequence[str], Sequence[SenseType]]:
    """Process a sentence and return the words and the sense keys of the target lemma."""

    words: Sequence[str] = []
    synsets_found: Sequence[SenseType] = []

    for word in sentence.findall("word"):
        text: str = word.attrib.get("surface_form", "").lower()

        # Skip non-alphanumeric tokens
        if not text or not all(c.isalnum() or c == "_" for c in text):
            continue

        # Add the word to the list of words
        words.append(text)

        # Add the sense key to the list of sense keys
        wn_key: str = word.attrib.get("wn30_key", "")
        if not wn_key:
            continue
        wn_key = wn_key.split(";")[0]

        try:
            synset: SenseType = wn.synset_from_sense_key(wn_key)
        except WordNetError:
            continue

        if text == lemma and synset in senses:
            synsets_found.append(synset)

    return words, synsets_found


def _load_sentences(
    paths: set[Path], lemma: str, senses: Set[SenseType]
) -> Sequence[SentenceRecord]:
    sentences: Sequence[SentenceRecord] = []

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
            words, synsets_found = _process_sentence(sentence, lemma, senses)

            # Skip sentences with none or multiple sense keys
            if len(synsets_found) != 1:
                continue

            # Add the sentence to the list of sentences as a record
            sentences.append(
                SentenceRecord(
                    lemma=lemma,
                    text=" ".join(words),
                    tokens=words,
                    synset=synsets_found[0].name(),
                    source=path.name,
                )
            )

    return sentences
