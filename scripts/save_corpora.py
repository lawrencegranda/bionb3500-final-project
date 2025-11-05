"""Persist SemCor corpora filtered by a ``SenseMap`` into a SQLite database."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.dataset import Corpora, Dataset  # pylint: disable=C0413
from src.utils import SenseMap, SentenceRecord  # pylint: disable=C0413


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SaveCorpora")


def _collect_sentences(
    corpus_paths: Sequence[Path],
    sense_map: SenseMap,
    max_sentences: int | None,
) -> List[SentenceRecord]:
    records: List[SentenceRecord] = []

    for lemma, label_map in sense_map.sense_map.items():
        senses = set().union(*label_map.values()) if label_map else set()
        if not senses:
            logger.info("No senses defined for lemma %s; skipping", lemma)
            continue

        corpora = Corpora(corpus_paths, lemma, senses)
        lemma_records = list(corpora)
        if max_sentences is not None:
            lemma_records = lemma_records[:max_sentences]

        logger.info("Collected %d sentences for lemma %s", len(lemma_records), lemma)
        records.extend(lemma_records)

    return records


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--data-config-path",
        type=Path,
        required=True,
        help="Path to the YAML data configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Program entrypoint."""

    args = _parse_args(argv)

    with args.data_config_path.open("r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle) or {}

    raw_data_entry = Path(data_config.get("raw_data_path"))
    sense_map_path = Path(data_config.get("sense_map_path"))
    dataset_path = Path(data_config.get("dataset_path"))
    max_sentences = int(data_config.get("max_sentences"))

    with sense_map_path.open("r", encoding="utf-8") as handle:
        sense_map_payload = json.load(handle)
    sense_map = SenseMap.from_dict(sense_map_payload)

    collected = _collect_sentences([raw_data_entry], sense_map, max_sentences)

    Dataset.from_sentences(dataset_path, collected)
    logger.info("Persisted %d sentences to %s", len(collected), dataset_path)


if __name__ == "__main__":  # pragma: no cover
    main()
