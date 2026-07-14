"""
FAISS-based semantic search over extracted legal clauses.

Builds an in-memory vector index from extracted clause texts,
enabling natural-language queries to find similar clauses across
all processed contracts.
"""

import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from config.settings import get_settings
from src.embeddings import EmbeddingEngine
from src.schemas import ContractResult, SearchResult, SearchResponse
from src.utils import setup_logging

logger = setup_logging(__name__)


class SemanticSearchEngine:
    """FAISS-powered semantic search over legal contract clauses.

    Builds a flat inner-product index over clause embeddings and
    supports natural-language queries to find similar clauses.

    Example:
        >>> engine = SemanticSearchEngine()
        >>> engine.build_index(results)
        >>> response = engine.search("early termination without cause")
        >>> for r in response.results:
        ...     print(r.contract_id, r.similarity_score)
    """

    INDEX_FILE = "clause_index.faiss"
    METADATA_FILE = "clause_metadata.pkl"

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the semantic search engine.

        Args:
            model_name: Sentence-transformer model override.
        """
        self._settings = get_settings()
        self._embedder = EmbeddingEngine(model_name)
        self._index = None
        self._metadata: List[dict] = []

    def _extract_clauses(
        self,
        results: List[ContractResult],
    ) -> tuple[List[str], List[dict]]:
        """Extract non-null clauses with metadata from pipeline results.

        Args:
            results: List of contract results.

        Returns:
            Tuple of (clause_texts, metadata_dicts) with matching indices.
        """
        texts: List[str] = []
        metadata: List[dict] = []

        clause_types = [
            ("termination", "termination_clause"),
            ("confidentiality", "confidentiality_clause"),
            ("liability", "liability_clause"),
        ]

        for result in results:
            for clause_type, field_name in clause_types:
                clause_text = getattr(result, field_name)
                if clause_text:
                    texts.append(clause_text)
                    metadata.append({
                        "contract_id": result.contract_id,
                        "clause_type": clause_type,
                        "clause_text": clause_text,
                    })

        logger.info(
            "Extracted %d non-null clauses from %d contracts",
            len(texts), len(results),
        )
        return texts, metadata

    def build_index(self, results: List[ContractResult]) -> None:
        """Build a FAISS index from pipeline results.

        Extracts all non-null clauses, generates embeddings,
        and builds an inner-product index for similarity search.

        Args:
            results: List of contract results from the pipeline.
        """
        import faiss

        texts, self._metadata = self._extract_clauses(results)

        if not texts:
            logger.warning("No clauses to index")
            return

        # Generate embeddings
        logger.info("Generating embeddings for %d clauses...", len(texts))
        embeddings = self._embedder.encode(texts, show_progress=True)

        # Build FAISS index (inner product on normalized vectors = cosine sim)
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)

        logger.info(
            "FAISS index built: %d vectors, %d dimensions",
            self._index.ntotal, dimension,
        )

        # Save index and metadata
        self._save_index()

    def _save_index(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        import faiss

        output_dir = self._settings.output_path

        index_path = output_dir / self.INDEX_FILE
        metadata_path = output_dir / self.METADATA_FILE

        faiss.write_index(self._index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self._metadata, f)

        logger.info("Index saved to: %s", index_path)

    def load_index(self) -> None:
        """Load a previously saved FAISS index from disk.

        Raises:
            FileNotFoundError: If index files don't exist.
        """
        import faiss

        output_dir = self._settings.output_path

        index_path = output_dir / self.INDEX_FILE
        metadata_path = output_dir / self.METADATA_FILE

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {output_dir}. "
                "Run the pipeline first to build the index."
            )

        self._index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self._metadata = pickle.load(f)

        logger.info(
            "Index loaded: %d vectors", self._index.ntotal
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> SearchResponse:
        """Search for clauses similar to a natural-language query.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return.
                Defaults to settings value.

        Returns:
            SearchResponse with ranked results.

        Raises:
            RuntimeError: If index hasn't been built or loaded.
        """
        if self._index is None:
            raise RuntimeError(
                "No index available. Call build_index() or load_index() first."
            )

        k = min(top_k or self._settings.search_top_k, self._index.ntotal)

        # Encode query
        query_vector = self._embedder.encode([query])

        # Search
        scores, indices = self._index.search(query_vector, k)

        # Build results
        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            meta = self._metadata[idx]
            # Clamp score to [0, 1] range
            clamped_score = float(max(0.0, min(1.0, score)))

            results.append(SearchResult(
                contract_id=meta["contract_id"],
                clause_type=meta["clause_type"],
                clause_text=meta["clause_text"],
                similarity_score=clamped_score,
            ))

        logger.info(
            "Search returned %d results for query: '%s'",
            len(results), query[:50],
        )

        return SearchResponse(query=query, results=results)
