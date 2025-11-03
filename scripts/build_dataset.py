"""Utilities for curating polysemous-word sentence datasets."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping
import yaml

from src.utils import clean_sentence, is_valid_sentence, SenseDefinition, SentenceRecord

DATA_DIR = Path("data")
INDEX_DB = DATA_DIR / "index.db"
SENSES_FILE = Path("senses.json")


def load_sense_inventory(
    lemma: str, source: Path = SENSES_FILE
) -> Dict[str, SenseDefinition]:
    """Load sense definitions for a lemma from ``senses.json``."""

    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get(lemma)
    if not entries:
        raise KeyError(f"No sense definitions found for lemma '{lemma}'.")
    return {
        entry["wn_key"]: SenseDefinition(label=entry["label"], wn_key=entry["wn_key"])
        for entry in entries
    }


def ensure_database(connection: sqlite3.Connection) -> None:
    """Create the ``sentences`` table if it does not exist."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sentences (
            id INTEGER PRIMARY KEY,
            word TEXT NOT NULL,
            sense_label TEXT NOT NULL,
            wn_key TEXT NOT NULL,
            sent_text TEXT NOT NULL,
            n_tokens INTEGER NOT NULL,
            source TEXT NOT NULL,
            split_idx INTEGER NOT NULL
        )
        """
    )


def candidate_sentences_with_word(lemma: str) -> Iterator[str]:
    """Yield raw sentences containing the lemma."""

    # TODO: Implement sentence harvesting logic
    return []


def predict_wn_key(
    cleaned_sentence: str, lemma: str, senses: Mapping[str, SenseDefinition]
) -> str:
    """Return a WordNet sense key for the sentence."""

    # TODO: Implement sense prediction logic
    return ""


def record_sentence(record: SentenceRecord, data_dir: Path = DATA_DIR) -> None:
    """Append the sentence to its sense-specific text file."""

    sense_file = data_dir / record.lemma / f"{record.sense_label}.txt"
    sense_file.parent.mkdir(parents=True, exist_ok=True)
    with sense_file.open("a", encoding="utf-8") as handle:
        handle.write(record.text + "\n")


def insert_into_index(record: SentenceRecord, connection: sqlite3.Connection) -> None:
    """Persist the sentence metadata inside ``index.db``."""

    connection.execute(
        """
        INSERT INTO sentences (word, sense_label, wn_key, sent_text, n_tokens, source, split_idx)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.lemma,
            record.sense_label,
            record.wn_key,
            record.text,
            len(record.tokens),
            record.source,
            record.split_index,
        ),
    )


def build_dataset_for_word(lemma: str, source_tag: str = "local") -> None:
    """Curate cleaned sentences for a single lemma and persist them to disk."""

    senses = load_sense_inventory(lemma)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(INDEX_DB) as connection:
        ensure_database(connection)
        for raw_sentence in candidate_sentences_with_word(lemma):
            cleaned = clean_sentence(raw_sentence)
            if not is_valid_sentence(cleaned, lemma):
                continue
            wn_key = predict_wn_key(cleaned, lemma, senses)
            sense_definition = senses[wn_key]
            tokens = cleaned.split()
            record = SentenceRecord(
                lemma=lemma,
                text=cleaned,
                tokens=tokens,
                wn_key=wn_key,
                sense_label=sense_definition.label,
                source=source_tag,
            )
            record_sentence(record)
            insert_into_index(record, connection)
        connection.commit()


def build_dataset(words: Iterable[str]) -> None:
    """Execute the dataset builder for each lemma in ``words``."""

    for lemma in words:
        build_dataset_for_word(lemma)


if __name__ == "__main__":
    words_path = Path("config/words.yaml")
    with open(words_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        DEFAULT_WORDS = tuple(word_entry["word"] for word_entry in config["targets"])
    build_dataset(DEFAULT_WORDS)
