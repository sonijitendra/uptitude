"""
Embedding generation using sentence-transformers.

Provides vector representations of legal clause text for
semantic similarity search using a lightweight, local model.
"""

from typing import List

import numpy as np

from config.settings import get_settings
from src.utils import setup_logging

logger = setup_logging(__name__)


class EmbeddingEngine:
    """Generates text embeddings using sentence-transformers.

    Uses a lightweight model (all-MiniLM-L6-v2 by default) that
    runs locally without API calls, producing 384-dimensional
    vectors suitable for cosine similarity search.

    Example:
        >>> engine = EmbeddingEngine()
        >>> vectors = engine.encode(["termination clause text..."])
        >>> print(vectors.shape)  # (1, 384)
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the embedding engine.

        Lazy-loads the model on first encode() call to avoid
        unnecessary memory usage if embeddings aren't needed.

        Args:
            model_name: Sentence-transformer model name.
                Defaults to settings value.
        """
        self._settings = get_settings()
        self._model_name = model_name or self._settings.embedding_model
        self._model = None

    def _load_model(self) -> None:
        """Load the sentence-transformer model.

        Called lazily on first encode() to avoid import overhead
        when embeddings aren't needed.
        """
        if self._model is not None:
            return

        logger.info("Loading embedding model: %s", self._model_name)

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        logger.info(
            "Embedding model loaded (dim=%d)",
            self._model.get_sentence_embedding_dimension(),
        )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into embedding vectors.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar.

        Returns:
            NumPy array of shape (len(texts), embedding_dim).
        """
        self._load_model()

        if not texts:
            logger.warning("Empty text list provided for encoding")
            return np.array([])

        logger.debug("Encoding %d texts", len(texts))

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # For cosine similarity via dot product
        )

        logger.debug(
            "Generated embeddings: shape=%s", embeddings.shape
        )

        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            Dimensionality of the embedding vectors.
        """
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
