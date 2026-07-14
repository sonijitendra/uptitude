"""
Pydantic schemas for input/output validation across the pipeline.

Defines strict data contracts for:
    - LLM clause extraction responses
    - LLM summary responses
    - Per-contract pipeline results
    - Semantic search results
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# LLM Response Schemas
# ---------------------------------------------------------------------------

class ClauseExtractionResponse(BaseModel):
    """Schema for validating LLM clause extraction output.

    Each field is nullable — if a clause is absent in the contract,
    the LLM should return null rather than hallucinate.
    """

    termination_clause: Optional[str] = Field(
        default=None,
        description="Exact text of the termination clause, or null if absent.",
    )
    confidentiality_clause: Optional[str] = Field(
        default=None,
        description="Exact text of the confidentiality clause, or null if absent.",
    )
    liability_clause: Optional[str] = Field(
        default=None,
        description="Exact text of the liability/limitation of liability clause, or null if absent.",
    )

    @field_validator("termination_clause", "confidentiality_clause", "liability_clause", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: Optional[str]) -> Optional[str]:
        """Convert empty strings and whitespace-only strings to None.

        Prevents LLMs from returning empty strings instead of null
        when a clause is not found.
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v


class SummaryResponse(BaseModel):
    """Schema for validating LLM contract summary output."""

    summary: str = Field(
        ...,
        min_length=50,
        description="Concise 100-150 word contract summary.",
    )

    @field_validator("summary", mode="before")
    @classmethod
    def clean_summary(cls, v: str) -> str:
        """Strip leading/trailing whitespace from summary."""
        if isinstance(v, str):
            return v.strip()
        return v


# ---------------------------------------------------------------------------
# Pipeline Output Schemas
# ---------------------------------------------------------------------------

class ContractResult(BaseModel):
    """Complete extraction result for a single contract.

    This is the final output schema written to CSV and JSON.
    """

    contract_id: str = Field(
        ...,
        description="Unique identifier for the contract (filename).",
    )
    summary: str = Field(
        ...,
        description="100-150 word contract summary.",
    )
    termination_clause: Optional[str] = Field(
        default=None,
        description="Extracted termination clause text, or null.",
    )
    confidentiality_clause: Optional[str] = Field(
        default=None,
        description="Extracted confidentiality clause text, or null.",
    )
    liability_clause: Optional[str] = Field(
        default=None,
        description="Extracted liability clause text, or null.",
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON serialization.

        Returns:
            Dictionary with all fields, None values preserved.
        """
        return self.model_dump()


class PipelineOutput(BaseModel):
    """Aggregated output for the entire pipeline run.

    Contains metadata about the run and all contract results.
    """

    total_contracts: int = Field(
        ...,
        description="Total number of contracts processed.",
    )
    successful: int = Field(
        ...,
        description="Number of contracts successfully processed.",
    )
    failed: int = Field(
        ...,
        description="Number of contracts that failed processing.",
    )
    results: List[ContractResult] = Field(
        default_factory=list,
        description="List of per-contract extraction results.",
    )


# ---------------------------------------------------------------------------
# Semantic Search Schemas (Bonus)
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    """A single semantic search result."""

    contract_id: str = Field(
        ...,
        description="Source contract identifier.",
    )
    clause_type: str = Field(
        ...,
        description="Type of clause (termination, confidentiality, liability).",
    )
    clause_text: str = Field(
        ...,
        description="Full text of the matching clause.",
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (0-1).",
    )


class SearchResponse(BaseModel):
    """Collection of semantic search results."""

    query: str = Field(
        ...,
        description="The search query text.",
    )
    results: List[SearchResult] = Field(
        default_factory=list,
        description="Ranked list of similar clauses.",
    )
