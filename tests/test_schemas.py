"""Unit tests for Pydantic schema validation."""

import pytest

from src.schemas import (
    ClauseExtractionResponse,
    SummaryResponse,
    ContractResult,
    SearchResult,
    SearchResponse,
)


class TestClauseExtractionResponse:
    """Tests for clause extraction schema validation."""

    def test_valid_full_response(self):
        """Should accept response with all clauses present."""
        response = ClauseExtractionResponse(
            termination_clause="Party may terminate upon notice.",
            confidentiality_clause="Information shall be confidential.",
            liability_clause="Liability shall not exceed $1M.",
        )
        assert response.termination_clause is not None
        assert response.confidentiality_clause is not None
        assert response.liability_clause is not None

    def test_valid_null_clauses(self):
        """Should accept null values for absent clauses."""
        response = ClauseExtractionResponse(
            termination_clause=None,
            confidentiality_clause=None,
            liability_clause=None,
        )
        assert response.termination_clause is None

    def test_empty_string_to_none(self):
        """Empty strings should be converted to None."""
        response = ClauseExtractionResponse(
            termination_clause="",
            confidentiality_clause="   ",
            liability_clause="\n\t  ",
        )
        assert response.termination_clause is None
        assert response.confidentiality_clause is None
        assert response.liability_clause is None

    def test_defaults_to_none(self):
        """All clause fields should default to None."""
        response = ClauseExtractionResponse()
        assert response.termination_clause is None
        assert response.confidentiality_clause is None
        assert response.liability_clause is None

    def test_partial_clauses(self):
        """Should handle mix of present and absent clauses."""
        response = ClauseExtractionResponse(
            termination_clause="Terminable upon 30 days notice.",
            confidentiality_clause=None,
            liability_clause="",
        )
        assert response.termination_clause == "Terminable upon 30 days notice."
        assert response.confidentiality_clause is None
        assert response.liability_clause is None


class TestSummaryResponse:
    """Tests for summary schema validation."""

    def test_valid_summary(self):
        """Should accept a valid summary."""
        summary_text = "A " * 60  # ~60 words, 120 chars
        response = SummaryResponse(summary=summary_text)
        assert len(response.summary) > 50

    def test_short_summary_rejected(self):
        """Should reject summaries shorter than 50 characters."""
        with pytest.raises(Exception):
            SummaryResponse(summary="Too short.")

    def test_whitespace_stripped(self):
        """Summary whitespace should be stripped."""
        response = SummaryResponse(summary="  " + "A " * 60 + "  ")
        assert not response.summary.startswith(" ")
        assert not response.summary.endswith(" ")


class TestContractResult:
    """Tests for the pipeline output schema."""

    def test_valid_result(self):
        """Should accept a complete result."""
        result = ContractResult(
            contract_id="test_contract",
            summary="A comprehensive summary of the contract " * 5,
            termination_clause="Terminate with 30 days notice.",
            confidentiality_clause=None,
            liability_clause=None,
        )
        assert result.contract_id == "test_contract"

    def test_to_dict(self):
        """to_dict should return all fields including None values."""
        result = ContractResult(
            contract_id="test",
            summary="A summary " * 15,
            termination_clause="clause",
            confidentiality_clause=None,
            liability_clause=None,
        )
        d = result.to_dict()
        assert "contract_id" in d
        assert "confidentiality_clause" in d
        assert d["confidentiality_clause"] is None


class TestSearchResult:
    """Tests for semantic search result schema."""

    def test_valid_search_result(self):
        """Should accept a valid search result."""
        result = SearchResult(
            contract_id="test",
            clause_type="termination",
            clause_text="Some clause text.",
            similarity_score=0.95,
        )
        assert result.similarity_score == 0.95

    def test_score_bounds(self):
        """Similarity score should be between 0 and 1."""
        with pytest.raises(Exception):
            SearchResult(
                contract_id="test",
                clause_type="termination",
                clause_text="text",
                similarity_score=1.5,
            )

        with pytest.raises(Exception):
            SearchResult(
                contract_id="test",
                clause_type="termination",
                clause_text="text",
                similarity_score=-0.1,
            )
