"""SQLite-backed collection of SemCor-UFSAC sentences filtered by a ``SenseMap``."""

from __future__ import annotations

import sqlite3
from typing import (
    Iterator,
    Optional,
    Set,
    Tuple,
    Sequence,
)
import logging
import pandas as pd

from src.types.senses import SenseType
from src.types.sentences import SentenceRecord

logger = logging.getLogger(__name__)


def _ensure_schema(connection: sqlite3.Connection, table_name: str) -> None:
    connection.execute(
        f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                synset TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
    )
    connection.execute(
        f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_unique
            ON {table_name} (lemma, text, label, synset)
            """
    )
    connection.commit()


class SentencesTable:
    """SQLite-backed collection of SemCor-UFSAC sentences filtered by a ``SenseMap``."""

    TABLE_NAME = "sentences"

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        _ensure_schema(self._conn, self.TABLE_NAME)

    def insert_record(self, record: SentenceRecord) -> None:
        """Insert a ``SentenceRecord`` into the dataset, ignoring duplicates."""

        # Insert the record into the database, ignoring duplicates.
        self._conn.execute(
            f"""
            INSERT OR IGNORE INTO {self.TABLE_NAME}
            (lemma, text, label, synset, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.lemma,
                record.text,
                record.label,
                record.synset,
                record.source,
            ),
        )

    def reset(self) -> None:
        """Reset the sentences table."""
        self._conn.execute(f"DELETE FROM {self.TABLE_NAME}")
        self._conn.commit()
        _ensure_schema(self._conn, self.TABLE_NAME)

    def add_sentences(
        self,
        sentences: Sequence[SentenceRecord],
    ) -> None:
        """Add sentences to the dataset."""

        # Insert the sentences into the dataset.
        for sentence in sentences:
            self.insert_record(sentence)

        # Commit the changes to the database.
        self._conn.commit()

    def iter_records(
        self,
        lemma: Optional[str] = None,
        senses: Optional[Set[SenseType]] = None,
    ) -> Iterator[SentenceRecord]:
        """Yield ``SentenceRecord`` instances stored in the dataset."""

        # Fetch all records from the dataset
        query = f"SELECT lemma, text, label, synset, source FROM {self.TABLE_NAME}"

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
            # Yield the record
            yield SentenceRecord(
                lemma=row["lemma"],
                text=row["text"],
                label=row["label"],
                synset=row["synset"],
                source=row["source"],
            )

    def to_df(self) -> "pd.DataFrame":
        """Convert the dataset to a pandas DataFrame."""

        query = f"SELECT * FROM {self.TABLE_NAME}"
        return pd.read_sql_query(query, self._conn)

    def __len__(self) -> int:
        # Count the number of records in the dataset
        query = f"SELECT COUNT(*) FROM {self.TABLE_NAME}"
        cursor = self._conn.execute(query)

        # Return the number of records in the dataset
        row = cursor.fetchone()

        return int(row[0]) if row else 0

    def __iter__(self) -> Iterator[SentenceRecord]:
        return self.iter_records()


__all__ = ["SentencesTable"]
