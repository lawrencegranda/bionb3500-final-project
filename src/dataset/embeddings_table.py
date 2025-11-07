"""Extract and store BERT embeddings for lemma words in sentences."""

from __future__ import annotations

import sqlite3
import logging
from typing import Optional, Mapping, Sequence
from collections import defaultdict

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

from src.types.embeddings import (
    EmbeddingRecord,
    LabelEmbeddings,
    LayerEmbeddings,
    LemmaEmbeddings,
)
from .sentences_table import SentencesTable

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("EmbeddingsTable")


@dataclass
class _EmbeddingsTableConfig:
    """Configuration for the embeddings table."""

    connection: sqlite3.Connection
    model_name: str
    tokenizer: BertTokenizer
    model: BertModel


def _ensure_schema(connection: sqlite3.Connection, table_name: str) -> None:
    """Create the embeddings table if it doesn't exist."""
    connection.execute(
        f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                sentence_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                layer INTEGER NOT NULL,
                PRIMARY KEY (sentence_id, layer),
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            )
            """
    )
    connection.commit()


class EmbeddingsTable:
    """Store BERT embeddings for target lemmas in a database."""

    TABLE_NAME = "embeddings"
    embeddings: Mapping[str, LemmaEmbeddings] = defaultdict(LemmaEmbeddings)

    def __init__(
        self,
        connection: sqlite3.Connection,
        model_name: str = "bert-base-uncased",
    ) -> None:
        """Initialize the embeddings table."""
        self._config = _EmbeddingsTableConfig(
            connection=connection,
            model_name=model_name,
            tokenizer=BertTokenizer.from_pretrained(model_name),
            model=BertModel.from_pretrained(model_name),
        )
        self._config.model.eval()
        _ensure_schema(self._config.connection, self.TABLE_NAME)

    def to_df(self) -> "pd.DataFrame":
        """Convert the embeddings table to a pandas DataFrame."""

        query = f"SELECT * FROM {self.TABLE_NAME}"
        return pd.read_sql_query(query, self._config.connection)

    def process_dataset(self, sentences_table: SentencesTable) -> None:
        """Process all sentences in a sentences_table and extract embeddings."""
        query = f"SELECT id, lemma, text FROM {sentences_table.TABLE_NAME}"
        cursor = self._config.connection.execute(query)

        total = len(sentences_table)
        for idx, row in enumerate(cursor, 1):
            sentence_id = row[0]
            lemma = row[1]
            text = row[2]

            if idx % 100 == 0:
                logger.info("Processing sentence %d/%d", idx, total)

            _extract_embedding(
                self._config,
                sentence_id,
                text,
                lemma,
            )

        logger.info("Finished processing %d sentences", total)

        # Compile the embeddings
        self.embeddings = _compile_embeddings(self._config)

    def get_embeddings(self) -> LemmaEmbeddings:
        """Get the compiled embeddings."""
        if self.embeddings is None:
            self.embeddings = _compile_embeddings(self._config)

        return self.embeddings


def _store_embedding(
    config: _EmbeddingsTableConfig,
    sentence_id: int,
    embedding: np.ndarray,
    layer: int,
) -> None:
    """Store an embedding in the database."""

    # Convert numpy array to bytes for storage
    embedding_bytes = embedding.tobytes()

    config.connection.execute(
        f"""
            INSERT OR REPLACE INTO {EmbeddingsTable.TABLE_NAME}
            (sentence_id, embedding, layer)
            VALUES (?, ?, ?)
            """,
        (sentence_id, embedding_bytes, layer),
    )
    config.connection.commit()


def _compile_embeddings(config: _EmbeddingsTableConfig) -> LemmaEmbeddings:
    """Compile the embeddings into a LemmaEmbeddings object."""
    query = f"""
            SELECT s.lemma, e.layer, s.label, e.embedding, e.sentence_id
            FROM {EmbeddingsTable.TABLE_NAME} e
            JOIN sentences s ON e.sentence_id = s.id
        """
    cursor = config.connection.execute(query)

    result: Mapping[str, LemmaEmbeddings] = {}
    for lemma, layer, label, embedding_bytes, sentence_id in cursor:
        # Build the embedding record
        embedding_record = _build_embedding_record(sentence_id, layer, embedding_bytes)

        lemma_obj = result.get(lemma, LemmaEmbeddings(lemma=lemma))
        layer_obj = lemma_obj.layers.get(layer, LayerEmbeddings(layer=layer))
        label_obj = layer_obj.labels.get(label, LabelEmbeddings(label=label))
        label_obj.records.append(embedding_record)

    return result


def _build_embedding_record(
    sentence_id: int, layer: int, embedding_bytes: bytes
) -> EmbeddingRecord:
    """Build an EmbeddingRecord from a sentence_id, layer, and bytes object."""
    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
    return EmbeddingRecord(sentence_id=sentence_id, layer=layer, embedding=embedding)


def _find_lemma_token_index(tokens: Sequence[str], lemma: str) -> Optional[int]:
    """Find the index of the lemma token in the tokenized sentence."""
    lemma_lower = lemma.lower()
    for i, token in enumerate(tokens):
        if token == lemma_lower:
            return i

    return None


def _extract_embedding(
    config: _EmbeddingsTableConfig,
    sentence_id: int,
    text: str,
    lemma: str,
) -> None:
    """Extract BERT embedding for the lemma in the sentence and store it."""
    # Tokenize the sentence
    inputs = config.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # Get tokens for debugging
    tokens = config.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Find the lemma token index
    lemma_idx = _find_lemma_token_index(tokens, lemma)
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
        outputs = config.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)

        # Determine which layers to extract
        num_layers = len(hidden_states)
        layers_to_extract = range(num_layers)

        # Extract and store embeddings for each layer
        for layer_idx in layers_to_extract:
            # shape: (1, seq_len, hidden_dim)
            layer_output = hidden_states[layer_idx]

            # shape: (hidden_dim,)
            lemma_embedding = layer_output[0, lemma_idx, :].numpy()

            # Store in database
            _store_embedding(
                config,
                sentence_id,
                lemma_embedding,
                layer_idx,
            )


__all__ = ["EmbeddingsTable"]
