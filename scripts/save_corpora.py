"""Persist SemCor corpora filtered by a ``SenseMap`` into a SQLite database."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.helpers import get_args  # pylint: disable=C0413, E0401
from src.builders import Corpora, SenseMap  # pylint: disable=C0413,E0401
from src.dataset import Database  # pylint: disable=C0413,E0401
from src.types.sentences import SentenceRecord  # pylint: disable=C0413,E0401


def _collect_sentences(
    wn_key_type: str,
    corpus_paths: Sequence[Path],
    sense_map: SenseMap,
    max_sentences: int | None,
) -> List[SentenceRecord]:
    """Collect sentences from corpora and convert them to ``SentenceRecord`` objects."""
    # Create a single Corpora object with the entire sense map
    corpora = Corpora(wn_key_type, corpus_paths, sense_map)

    records: List[SentenceRecord] = []

    # Collect sentences organized by lemma and label
    for lemma, label_to_sentences in corpora.sentences.items():
        for label, sentence_list in label_to_sentences.items():
            # Apply max_sentences limit per label if specified
            limited_sentences = sentence_list
            if max_sentences is not None:
                limited_sentences = sentence_list[:max_sentences]

            print(
                "Collected %d sentences for lemma %s, label %s",
                len(limited_sentences),
                lemma,
                label,
            )
            records.extend(limited_sentences)

    return records


def main() -> None:
    """Program entrypoint."""

    args = get_args(__doc__)

    # Extract configuration parameters
    wn_key_type = args.config.corpora.wn_key_type
    corpora_paths = args.config.paths.corpora_paths
    sense_map_path = args.config.paths.sense_map_path
    dataset_path = args.config.paths.dataset_path
    max_sentences = args.config.corpora.max_sentences
    model_name = args.model

    with sense_map_path.open("r", encoding="utf-8") as handle:
        sense_map_payload = json.load(handle)
    sense_map = SenseMap.from_dict(sense_map_payload)

    collected = _collect_sentences(wn_key_type, corpora_paths, sense_map, max_sentences)

    database = Database.from_db(dataset_path, model_name)
    sentences_table = database.sentences_table
    sentences_table.add_sentences(collected)
    database.close()

    print(f"Persisted {len(collected)} sentences to {dataset_path}")


if __name__ == "__main__":
    main()
