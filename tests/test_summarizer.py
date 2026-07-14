"""Unit tests for contract summarizer with mocked LLM."""

from unittest.mock import MagicMock

import pytest

from src.summarizer import ContractSummarizer
from src.schemas import SummaryResponse


class TestContractSummarizer:
    """Tests for contract summarization logic with mocked LLM calls."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        return MagicMock()

    @pytest.fixture
    def summarizer(self, mock_llm, clear_settings_cache):
        """Create summarizer with mocked LLM."""
        return ContractSummarizer(llm_client=mock_llm)

    def test_summarize_returns_summary(self, summarizer, mock_llm, sample_contract_text):
        """Should return a SummaryResponse with valid summary."""
        expected = "This agreement governs software development services " * 5
        mock_llm.generate.return_value = SummaryResponse(summary=expected)

        result = summarizer.summarize(sample_contract_text)

        assert isinstance(result, SummaryResponse)
        assert len(result.summary) > 50
        mock_llm.generate.assert_called_once()

    def test_summarize_empty_text_raises(self, summarizer):
        """Should raise ValueError for empty input."""
        with pytest.raises(ValueError, match="empty"):
            summarizer.summarize("")

        with pytest.raises(ValueError, match="empty"):
            summarizer.summarize("   ")

    def test_long_text_truncated(self, summarizer, mock_llm, clear_settings_cache):
        """Long contracts should be truncated before summarization."""
        long_text = "Legal clause text. " * 10000  # ~190k chars
        expected = "Summary of the long contract " * 6
        mock_llm.generate.return_value = SummaryResponse(summary=expected)

        result = summarizer.summarize(long_text)

        assert isinstance(result, SummaryResponse)
        # Verify that the prompt sent to LLM was truncated
        call_args = mock_llm.generate.call_args
        user_prompt = call_args.kwargs.get("user_prompt", call_args[1].get("user_prompt", ""))
        # The truncated text should contain the omission marker
        assert "omitted" in user_prompt or len(user_prompt) < len(long_text)

    def test_word_count_validation(self, summarizer, mock_llm, sample_contract_text):
        """Should accept summaries within the target word count range."""
        # ~120 words - within range
        summary_120 = " ".join(["word"] * 120)
        mock_llm.generate.return_value = SummaryResponse(summary=summary_120)

        result = summarizer.summarize(sample_contract_text)
        word_count = len(result.summary.split())
        assert 80 <= word_count <= 180  # Loose bounds for validation
