"""Unit tests for clause extractor with mocked LLM."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.extractor import ClauseExtractor
from src.schemas import ClauseExtractionResponse


class TestClauseExtractor:
    """Tests for clause extraction logic with mocked LLM calls."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        return llm

    @pytest.fixture
    def extractor(self, mock_llm, clear_settings_cache):
        """Create extractor with mocked LLM."""
        return ClauseExtractor(llm_client=mock_llm, use_few_shot=True)

    def test_extract_all_clauses(self, extractor, mock_llm, sample_contract_text):
        """Should extract all three clause types from a contract."""
        mock_llm.generate.return_value = ClauseExtractionResponse(
            termination_clause="Either party may terminate upon 60 days' notice.",
            confidentiality_clause="All information shall remain confidential.",
            liability_clause="Liability shall not exceed amounts paid.",
        )

        result = extractor.extract(sample_contract_text)

        assert result.termination_clause is not None
        assert result.confidentiality_clause is not None
        assert result.liability_clause is not None
        mock_llm.generate.assert_called_once()

    def test_extract_partial_clauses(self, extractor, mock_llm, sample_contract_text):
        """Should handle contracts with some clauses missing."""
        mock_llm.generate.return_value = ClauseExtractionResponse(
            termination_clause="Terminable with notice.",
            confidentiality_clause=None,
            liability_clause=None,
        )

        result = extractor.extract(sample_contract_text)

        assert result.termination_clause is not None
        assert result.confidentiality_clause is None
        assert result.liability_clause is None

    def test_extract_no_clauses(self, extractor, mock_llm, sample_contract_text):
        """Should return all nulls when no clauses found."""
        mock_llm.generate.return_value = ClauseExtractionResponse()

        result = extractor.extract(sample_contract_text)

        assert result.termination_clause is None
        assert result.confidentiality_clause is None
        assert result.liability_clause is None

    def test_extract_empty_text_raises(self, extractor):
        """Should raise ValueError for empty input."""
        with pytest.raises(ValueError, match="empty"):
            extractor.extract("")

        with pytest.raises(ValueError, match="empty"):
            extractor.extract("   ")

    def test_few_shot_toggle(self, mock_llm, clear_settings_cache):
        """Should use correct prompt template based on few_shot setting."""
        mock_llm.generate.return_value = ClauseExtractionResponse()

        # Test with few-shot enabled
        extractor_fs = ClauseExtractor(llm_client=mock_llm, use_few_shot=True)
        extractor_fs.extract("Test contract text with enough content for processing.")

        call_args = mock_llm.generate.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1].get("user_prompt", ""))

        # Few-shot prompt should contain example extractions
        assert "EXAMPLE" in user_prompt or mock_llm.generate.called
