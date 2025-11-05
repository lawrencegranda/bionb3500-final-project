"""
Load corpora from UFSAC-style XML files and convert them to SenseDefinition objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

from src.utils import SenseDefinition

logger = logging.getLogger(__name__)


@dataclass
class Corpora:
    """
    Corpus loader that collects sentences from UFSAC-style XML files
    and converts them to ``SenseDefinition`` objects.
    """

    paths: set[Path]
    sentences: list[SenseDefinition]

    def __init__(self, paths: set[str | Path]) -> None:
        self.paths = {Path(path) for path in paths}
        self.sentences = []
        self._load_sentences()

    def _load_sentences(self) -> None:
        for path in sorted(self.paths):
            if not path.exists():
                logger.warning("Corpus file %s does not exist; skipping", path)
                continue

            try:
                root = ET.parse(path).getroot()
            except ET.ParseError as exc:
                logger.error("Failed to parse XML corpus %s: %s", path, exc)
                continue

            for sentence in root.iterfind(".//sentence"):
                for word in sentence.findall("word"):
                    text = word.attrib.get("surface_form").lower()
                    if not text or not all(c.isalnum() or c == "_" for c in text):
                        continue

                    self.sentences.append(
                        SenseDefinition(
                            label=text,
                            wn_key=word.attrib.get("wn_key", ""),
                        )
                    )

    def __iter__(self) -> Iterator[SenseDefinition]:
        return iter(self.sentences)

    def __len__(self) -> int:
        return len(self.sentences)
