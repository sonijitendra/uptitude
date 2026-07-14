"""
Contract summarization using LLM.

Generates concise 100-150 word summaries covering:
    - Purpose of the agreement
    - Key obligations of each party
    - Notable risks or penalties
"""

from typing import Optional

from config.settings import get_settings
from src.llm import LLMClient
from src.prompts import SUMMARY_SYSTEM_PROMPT, format_summary_prompt
from src.schemas import SummaryResponse
from src.utils import setup_logging, count_tokens, chunk_text

logger = setup_logging(__name__)


class ContractSummarizer:
    """Generates concise summaries of legal contracts using LLM.

    Produces factual, 100-150 word summaries suitable for executive
    review. For very long contracts, summarizes a representative
    portion (beginning + end) to stay within token limits.

    Example:
        >>> summarizer = ContractSummarizer()
        >>> result = summarizer.summarize(contract_text)
        >>> print(result.summary)
    """

    # Maximum characters to send for summarization
    MAX_SUMMARY_INPUT_CHARS: int = 50000

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        """Initialize the summarizer.

        Args:
            llm_client: Pre-configured LLM client. Creates one if None.
        """
        self._llm = llm_client or LLMClient()
        self._settings = get_settings()

    def _prepare_text(self, text: str) -> str:
        """Prepare contract text for summarization.

        For very long contracts, takes the beginning and end portions
        to capture the most important information (parties, purpose
        at the start; penalties, termination at the end).

        Args:
            text: Full contract text.

        Returns:
            Text trimmed to fit summarization context window.
        """
        if len(text) <= self.MAX_SUMMARY_INPUT_CHARS:
            return text

        # Take first 60% and last 40% of the budget
        head_size = int(self.MAX_SUMMARY_INPUT_CHARS * 0.6)
        tail_size = self.MAX_SUMMARY_INPUT_CHARS - head_size

        head = text[:head_size]
        tail = text[-tail_size:]

        combined = (
            head
            + "\n\n[... middle portion omitted for brevity ...]\n\n"
            + tail
        )

        logger.info(
            "Contract text truncated for summarization: %d → %d chars",
            len(text), len(combined),
        )
        return combined

    def _validate_word_count(self, summary: str) -> str:
        """Validate and log the word count of the summary.

        Args:
            summary: Generated summary text.

        Returns:
            The summary (unchanged).
        """
        word_count = len(summary.split())
        if word_count < 80:
            logger.warning(
                "Summary may be too short: %d words (target: 100-150)",
                word_count,
            )
        elif word_count > 180:
            logger.warning(
                "Summary may be too long: %d words (target: 100-150)",
                word_count,
            )
        else:
            logger.debug("Summary word count: %d (target: 100-150)", word_count)

        return summary

    def summarize(self, contract_text: str) -> SummaryResponse:
        """Generate a 100-150 word summary of a legal contract.

        Args:
            contract_text: Cleaned contract text.

        Returns:
            Validated SummaryResponse with the summary text.

        Raises:
            RuntimeError: If summarization fails after all retries.
            ValueError: If contract text is empty.
        """
        if not contract_text or not contract_text.strip():
            raise ValueError("Cannot summarize empty text")

        # Prepare text for summarization
        prepared_text = self._prepare_text(contract_text)

        # Format prompt
        user_prompt = format_summary_prompt(prepared_text)

        logger.info(
            "Generating summary (input: %d chars, ~%d tokens)",
            len(prepared_text),
            count_tokens(prepared_text, self._settings.llm_model),
        )

        # Call LLM
        result = self._llm.generate(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=SummaryResponse,
        )

        # Validate word count
        self._validate_word_count(result.summary)

        return result
