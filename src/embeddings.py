"""Extract and store BERT embeddings for lemma words in sentences."""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch  # pylint: disable=E0401
from transformers import BertTokenizer, BertModel  # pylint: disable=E0401

from src.dataset import Dataset

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Store BERT embeddings for target lemmas in a database."""

    TABLE_NAME = "embeddings"

    def __init__(
        self,
        connection: sqlite3.Connection,
        model_name: str = "bert-base-uncased",
    ) -> None:
        """
        Initialize the embedding store.

        Args:
            connection: SQLite connection (should be same as Dataset connection).
            model_name: HuggingFace model identifier for BERT.
        """
        self._conn = connection
        self._tokenizer = BertTokenizer.from_pretrained(model_name)
        self._model = BertModel.from_pretrained(model_name)
        self._model.eval()
        self._ensure_schema()

    @classmethod
    def from_db(
        cls, db_path: str | Path, model_name: str = "bert-base-uncased"
    ) -> "EmbeddingStore":
        """Load an existing database and attach embedding store."""
        connection = sqlite3.connect(Path(db_path))
        return cls(connection, model_name)

    def _ensure_schema(self) -> None:
        """Create the embeddings table if it doesn't exist."""
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                sentence_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                layer INTEGER NOT NULL,
                PRIMARY KEY (sentence_id, layer),
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            )
            """
        )
        self._conn.commit()

    def _find_lemma_token_index(self, tokens: list[str], lemma: str) -> Optional[int]:
        """
        Find the index of the lemma token in the tokenized sentence.

        Args:
            tokens: List of tokens from the tokenizer.
            lemma: Target lemma to find.

        Returns:
            Index of the lemma token, or None if not found.
        """
        # Try to find exact match first
        lemma_lower = lemma.lower()
        for i, token in enumerate(tokens):
            if token == lemma_lower:
                return i

        # Try to find partial match (for subword tokens)
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                continue
            if lemma_lower.startswith(token):
                return i

        return None

    def extract_embedding(
        self,
        sentence_id: int,
        text: str,
        lemma: str,
        layer: int = -1,
    ) -> None:
        """
        Extract BERT embedding for the lemma in the sentence and store it.

        Args:
            sentence_id: The sentence ID from the sentences table.
            text: The sentence text.
            lemma: The target lemma to extract embedding for.
            layer: Which layer to extract up to (-1 for all layers).
        """
        # Tokenize the sentence
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Get tokens for debugging
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Find the lemma token index
        lemma_idx = self._find_lemma_token_index(tokens, lemma)
        if lemma_idx is None:
            logger.warning(
                "Lemma '%s' not found in sentence %d: %s",
                lemma,
                sentence_id,
                text,
            )
            return

        # Extract embeddings
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            hidden_states = (
                outputs.hidden_states
            )  # Tuple of (batch, seq_len, hidden_dim)

            # Determine which layers to extract
            num_layers = len(hidden_states)
            if layer == -1:
                # Extract all layers
                layers_to_extract = range(num_layers)
            else:
                # Extract layers 0 to layer (inclusive)
                layers_to_extract = range(min(layer + 1, num_layers))

            # Extract and store embeddings for each layer
            for layer_idx in layers_to_extract:
                layer_output = hidden_states[
                    layer_idx
                ]  # shape: (1, seq_len, hidden_dim)
                lemma_embedding = layer_output[
                    0, lemma_idx, :
                ].numpy()  # shape: (hidden_dim,)

                # Store in database
                self._store_embedding(sentence_id, lemma_embedding, layer_idx)

    def _store_embedding(
        self,
        sentence_id: int,
        embedding: np.ndarray,
        layer: int,
    ) -> None:
        """Store an embedding in the database."""

        # Convert numpy array to bytes for storage
        embedding_bytes = embedding.tobytes()

        self._conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.TABLE_NAME}
            (sentence_id, embedding, layer)
            VALUES (?, ?, ?)
            """,
            (sentence_id, embedding_bytes, layer),
        )
        self._conn.commit()

    def process_dataset(self, dataset: Dataset, layer: int = -1) -> None:
        """
        Process all sentences in a dataset and extract embeddings.

        Args:
            dataset: The dataset to process.
            layer: Which BERT layer to extract embeddings up to (-1 for all layers).
        """
        # Query sentences with their IDs
        query = f"SELECT id, lemma, text FROM {dataset.TABLE_NAME}"
        cursor = self._conn.execute(query)

        total = len(dataset)
        for idx, row in enumerate(cursor, 1):
            sentence_id = row[0]
            lemma = row[1]
            text = row[2]

            if idx % 100 == 0:
                logger.info("Processing sentence %d/%d", idx, total)

            self.extract_embedding(sentence_id, text, lemma, layer)

        logger.info("Finished processing %d sentences", total)

    def get_embedding(
        self, sentence_id: int, layer: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Retrieve an embedding for a given sentence_id and layer.

        Args:
            sentence_id: The sentence ID to retrieve.
            layer: Specific layer to retrieve. If None, returns the last layer stored.

        Returns:
            The embedding as a numpy array, or None if not found.
        """
        if layer is not None:
            # Get specific layer
            cursor = self._conn.execute(
                f"""
                SELECT embedding FROM {self.TABLE_NAME}
                WHERE sentence_id = ? AND layer = ?
                """,
                (sentence_id, layer),
            )
        else:
            # Get the highest layer available for this sentence_id
            cursor = self._conn.execute(
                f"""
                SELECT embedding FROM {self.TABLE_NAME}
                WHERE sentence_id = ?
                ORDER BY layer DESC
                LIMIT 1
                """,
                (sentence_id,),
            )

        row = cursor.fetchone()
        if row is None:
            return None

        embedding_bytes = row[0]
        # Reconstruct numpy array (assuming 768 dimensions for BERT-base)
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        return embedding

    def get_all_layers(self, sentence_id: int) -> dict[int, np.ndarray]:
        """
        Retrieve embeddings for all layers of a given sentence_id.

        Args:
            sentence_id: The sentence ID to retrieve.

        Returns:
            Dictionary mapping layer number to embedding array.
        """
        cursor = self._conn.execute(
            f"""
            SELECT layer, embedding FROM {self.TABLE_NAME}
            WHERE sentence_id = ?
            ORDER BY layer ASC
            """,
            (sentence_id,),
        )

        result = {}
        for row in cursor:
            layer_num = row[0]
            embedding_bytes = row[1]
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            result[layer_num] = embedding

        return result

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.commit()
            self._conn.close()

    def __enter__(self) -> "EmbeddingStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[override]
        self.close()
