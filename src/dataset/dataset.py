"""_summary_"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    Optional,
    Set,
)

from src.utils import SenseMapType, SenseType, SentenceRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _AnnotatedToken:
    surface: str
    lemma: Optional[str]
    sense_key: Optional[str]
    synset_name: Optional[str]


@dataclass(frozen=True, slots=True)
class _TargetEntry:
    lemma: str
    label: str
    synset_name: str
    sense_keys: Set[str]


def _normalise_lemma(value: str | None) -> str:
    """Normalise a lemma by replacing underscores with spaces and converting to lowercase."""
    return "" if value is None else value.replace("_", " ").strip().lower()


def _sense_keys_for_target(synset: SenseType, lemma: str) -> Set[str]:
    """Get the sense keys for a target lemma in a synset."""
    lemma_norm = _normalise_lemma(lemma)
    sense_keys: Set[str] = set()
    for lemma_obj in synset.lemmas():
        lemma_name_norm = _normalise_lemma(lemma_obj.name())
        if lemma_name_norm == lemma_norm:
            sense_keys.add(lemma_obj.key())
    if not sense_keys:
        # Fall back to all lemma keys for the synset if the explicit lemma was not found.
        sense_keys = {lemma_obj.key() for lemma_obj in synset.lemmas()}
    return sense_keys


def _build_target_index(sense_map: SenseMapType) -> Dict[str, _TargetEntry]:
    """Build an index of target entries from a sense map."""
    index: Dict[str, _TargetEntry] = {}
    for lemma, label_to_synsets in sense_map.items():
        for label, synsets in label_to_synsets.items():
            for synset in synsets:
                synset_name = synset.name()
                sense_keys = _sense_keys_for_target(synset, lemma)
                index[synset_name] = _TargetEntry(
                    lemma=lemma,
                    label=label,
                    synset_name=synset_name,
                    sense_keys=sense_keys,
                )
    return index


class Dataset:
    """SQLite-backed collection of SemCor-UFSAC sentences filtered by a ``SenseMap``."""

    TABLE_NAME = "semcor_sentences"
    SOURCE = "semcor_ufsac"

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def build_from_sense_map(
        cls,
        db_path: str | Path,
        overwrite: bool = True,
    ) -> "Dataset":
        """Create a new dataset.

        Args:
            db_path: Where to persist the SQLite database.
            overwrite: If true, existing databases at ``db_path`` are removed before
                building the dataset.
        """

        # Create the database file if it doesn't exist.
        path = Path(db_path)
        if overwrite and path.exists():
            path.unlink()

        # Connect to the database and create the dataset.
        connection = sqlite3.connect(path)
        dataset = cls(connection)

        return dataset

    @classmethod
    def from_db(cls, db_path: str | Path) -> "Dataset":
        """Load an existing SQLite-backed dataset from disk."""

        connection = sqlite3.connect(Path(db_path))
        return cls(connection)

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        if self._conn:
            self._conn.commit()
            self._conn.close()

    def __enter__(self) -> "Dataset":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[override]
        self.close()

    def __len__(self) -> int:
        # Count the number of records in the dataset
        query = f"SELECT COUNT(*) FROM {self.TABLE_NAME}"
        cursor = self._conn.execute(query)

        # Return the number of records in the dataset
        row = cursor.fetchone()

        return int(row[0]) if row else 0

    def insert_record(self, record: SentenceRecord, synset_name: str) -> None:
        """Insert a ``SentenceRecord`` into the dataset, ignoring duplicates."""

        tokens_json = json.dumps(list(record.tokens))

        # Insert the record into the database, ignoring duplicates.
        self._conn.execute(
            f"""
            INSERT OR IGNORE INTO {self.TABLE_NAME}
            (lemma, sense_label, wn_key, synset_name, sentence, tokens, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.lemma,
                record.sense_label,
                record.wn_key,
                synset_name,
                record.text,
                tokens_json,
                record.source,
            ),
        )

    def iter_records(
        self,
        *,
        lemma: Optional[str] = None,
        sense_label: Optional[str] = None,
    ) -> Iterator[SentenceRecord]:
        """Yield ``SentenceRecord`` instances stored in the dataset."""

        conditions = []
        params: list[str] = []
        if lemma is not None:
            conditions.append("lemma = ?")
            params.append(lemma)
        if sense_label is not None:
            conditions.append("sense_label = ?")
            params.append(sense_label)

        query = f"SELECT lemma, sense_label, wn_key, sentence, tokens, source FROM {self.TABLE_NAME}"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = self._conn.execute(query, params)
        for row in cursor:
            tokens = json.loads(row["tokens"])
            yield SentenceRecord(
                lemma=row["lemma"],
                text=row["sentence"],
                tokens=tuple(tokens),
                wn_key=row["wn_key"],
                sense_label=row["sense_label"],
                source=row["source"],
            )

    def _ensure_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                sense_label TEXT NOT NULL,
                wn_key TEXT NOT NULL,
                synset_name TEXT NOT NULL,
                sentence TEXT NOT NULL,
                tokens TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_unique
            ON {self.TABLE_NAME} (lemma, sense_label, wn_key, sentence)
            """
        )
        self._conn.commit()
