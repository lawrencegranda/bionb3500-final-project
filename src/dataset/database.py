"""Extract and store BERT embeddings for lemma words in sentences."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .embeddings_table import EmbeddingsTable
from .sentences_table import SentencesTable


class Database:
    """Store BERT embeddings for target lemmas in a database."""

    def __init__(
        self, connection: sqlite3.Connection, model_name: Optional[str] = None
    ) -> None:
        """Initialize the database."""
        self._conn = connection
        self.sentences_table = SentencesTable(self._conn)

        self.embeddings_table = (
            EmbeddingsTable(self._conn, model_name) if model_name else None
        )

    @classmethod
    def from_db(
        cls, db_path: str | Path, model_name: Optional[str] = None
    ) -> "Database":
        """Load an existing database."""
        connection = sqlite3.connect(Path(db_path))
        return cls(connection, model_name)

    def reset(self) -> None:
        """Reset the database."""
        self.sentences_table.reset()
        if self.embeddings_table:
            self.embeddings_table.reset()

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
