"""Extract and store BERT embeddings for lemma words in sentences."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .embeddings_table import EmbeddingsTable
from .sentences_table import SentencesTable


class Database:
    """Store BERT embeddings for target lemmas in a database."""

    def __init__(self, connection: sqlite3.Connection, model_name: str) -> None:
        """Initialize the database."""
        self._conn = connection
        self.embeddings_table = EmbeddingsTable(self._conn, model_name)
        self.sentences_table = SentencesTable(self._conn)

    @classmethod
    def from_db(cls, db_path: str | Path, model_name: str) -> "Database":
        """Load an existing database."""
        connection = sqlite3.connect(Path(db_path))
        return cls(connection, model_name)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.commit()
            self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["Database"]
