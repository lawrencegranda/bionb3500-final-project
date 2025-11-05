"""_summary_"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import (
    Iterator,
    Optional,
    Set,
    Tuple,
    Sequence,
)

from src.utils import SenseType, SentenceRecord

logger = logging.getLogger(__name__)


class Dataset:
    """SQLite-backed collection of SemCor-UFSAC sentences filtered by a ``SenseMap``."""

    TABLE_NAME = "sentences"

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def from_sentences(
        cls,
        db_path: str | Path,
        sentences: Sequence[SentenceRecord],
        overwrite: bool = True,
    ) -> "Dataset":
        """Create a new dataset.

        Args:
            db_path: Where to persist the SQLite database.
            sentences: The sentences to insert into the dataset.
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

        # Insert the sentences into the dataset.
        for sentence in sentences:
            dataset.insert_record(sentence)
        dataset._conn.commit()

        return dataset

    @classmethod
    def from_db(cls, db_path: str | Path) -> "Dataset":
        """Load an existing SQLite-backed dataset from disk."""

        connection = sqlite3.connect(Path(db_path))
        return cls(connection)

    def insert_record(self, record: SentenceRecord) -> None:
        """Insert a ``SentenceRecord`` into the dataset, ignoring duplicates."""

        tokens_json: str = json.dumps(list(record.tokens))

        # Insert the record into the database, ignoring duplicates.
        self._conn.execute(
            f"""
            INSERT OR IGNORE INTO {self.TABLE_NAME}
            (lemma, text, synset, tokens, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.lemma,
                record.text,
                record.synset,
                tokens_json,
                record.source,
            ),
        )

    def iter_records(
        self,
        lemma: Optional[str] = None,
        senses: Optional[Set[SenseType]] = None,
    ) -> Iterator[SentenceRecord]:
        """Yield ``SentenceRecord`` instances stored in the dataset."""

        # Fetch all records from the dataset
        query = f"SELECT lemma, text, synset, tokens, source FROM {self.TABLE_NAME}"

        params: Tuple[str, ...] = ()

        # Filter by lemma if provided
        if lemma is not None:
            query += " WHERE lemma = ?"
            params = (lemma,)

        # Filter by senses if provided
        if senses is not None and len(senses) > 0:
            query += (
                " WHERE synset IN (SELECT synset FROM senses WHERE sense_key IN (?))"
            )
            params += tuple(s.name() for s in senses)

        # Execute the query and yield the records
        cursor = self._conn.execute(query, params)
        for row in cursor:
            # Load the tokens from the database
            tokens: Sequence[str] = json.loads(row["tokens"])

            # Yield the record
            yield SentenceRecord(
                lemma=row["lemma"],
                text=row["text"],
                tokens=tokens,
                synset=row["synset"],
                source=row["source"],
            )

    def _ensure_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                text TEXT NOT NULL,
                tokens TEXT NOT NULL,
                synset TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_unique
            ON {self.TABLE_NAME} (lemma, text, synset)
            """
        )
        self._conn.commit()

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

    def __iter__(self) -> Iterator[SentenceRecord]:
        return self.iter_records()
