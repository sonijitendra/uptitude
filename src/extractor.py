"""
Legal clause extraction from contracts using LLM.

Handles both short contracts (single-pass extraction) and long
contracts (chunked extraction with result merging) transparently.
"""

import json
from typing import Optional

from config.settings import get_settings
from src.llm import LLMClient
from src.prompts import (
    CLAUSE_EXTRACTION_SYSTEM_PROMPT,
    format_extraction_prompt,
    format_chunk_extraction_prompt,
    format_merge_prompt,
)
from src.schemas import ClauseExtractionResponse
from src.utils import setup_logging, count_tokens, chunk_text

logger = setup_logging(__name__)


class ClauseExtractor:
    """Extracts termination, confidentiality, and liability clauses from contracts.

    Automatically selects the extraction strategy based on contract length:
        - Short contracts: Single-pass extraction with few-shot examples
        - Long contracts: Chunked extraction → merge results

    Example:
        >>> extractor = ClauseExtractor()
        >>> result = extractor.extract(contract_text)
        >>> print(result.termination_clause)
    """

    # Token budget for the prompt template + system message
    PROMPT_OVERHEAD_TOKENS: int = 2000

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_few_shot: bool = True,
    ) -> None:
        """Initialize the clause extractor.

        Args:
            llm_client: Pre-configured LLM client. Creates one if None.
            use_few_shot: Whether to use few-shot examples in prompts.
        """
        self._llm = llm_client or LLMClient()
        self._settings = get_settings()
        self._use_few_shot = use_few_shot

    def _fits_single_pass(self, text: str) -> bool:
        """Check if contract text fits in a single LLM call.

        Accounts for prompt template overhead and max context window.

        Args:
            text: Contract text to check.

        Returns:
            True if text fits in a single pass.
        """
        text_tokens = count_tokens(text, self._settings.llm_model)
        # gpt-4o-mini has 128k context, but we stay well under for cost
        max_input_tokens = 120000 - self.PROMPT_OVERHEAD_TOKENS - self._settings.llm_max_tokens
        fits = text_tokens <= max_input_tokens

        logger.debug(
            "Token check: text=%d, max_input=%d, fits=%s",
            text_tokens, max_input_tokens, fits,
        )
        return fits

    def _extract_single_pass(self, text: str) -> ClauseExtractionResponse:
        """Extract clauses from a contract in a single LLM call.

        Args:
            text: Full contract text.

        Returns:
            Validated clause extraction response.
        """
        user_prompt = format_extraction_prompt(
            contract_text=text,
            use_few_shot=self._use_few_shot,
        )

        return self._llm.generate(
            system_prompt=CLAUSE_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=ClauseExtractionResponse,
        )

    def _extract_chunked(self, text: str) -> ClauseExtractionResponse:
        """Extract clauses from a long contract using chunking.

        Strategy:
            1. Split text into overlapping chunks
            2. Extract clauses from each chunk independently
            3. Merge results using LLM to select best extractions

        Args:
            text: Full contract text (too long for single pass).

        Returns:
            Merged clause extraction response.
        """
        chunks = chunk_text(
            text,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )

        logger.info("Processing %d chunks for clause extraction", len(chunks))

        chunk_results: list[dict] = []

        for i, chunk in enumerate(chunks, 1):
            logger.debug("Processing chunk %d/%d", i, len(chunks))

            user_prompt = format_chunk_extraction_prompt(
                chunk_text=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
            )

            try:
                result = self._llm.generate(
                    system_prompt=CLAUSE_EXTRACTION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    response_model=ClauseExtractionResponse,
                )
                chunk_results.append(result.model_dump())
            except RuntimeError as exc:
                logger.warning("Chunk %d extraction failed: %s", i, exc)
                chunk_results.append({
                    "termination_clause": None,
                    "confidentiality_clause": None,
                    "liability_clause": None,
                })

        # Check if any clauses were found across chunks
        any_found = any(
            any(v is not None for v in result.values())
            for result in chunk_results
        )

        if not any_found:
            logger.info("No clauses found in any chunk")
            return ClauseExtractionResponse()

        # Merge results using LLM
        merge_prompt = format_merge_prompt(
            chunk_results_json=json.dumps(chunk_results, indent=2),
            total_chunks=len(chunks),
        )

        return self._llm.generate(
            system_prompt=CLAUSE_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=merge_prompt,
            response_model=ClauseExtractionResponse,
        )

    def extract(self, contract_text: str) -> ClauseExtractionResponse:
        """Extract clauses from a contract, auto-selecting strategy.

        Automatically determines whether to use single-pass or chunked
        extraction based on contract length.

        Args:
            contract_text: Cleaned contract text.

        Returns:
            Validated clause extraction response with termination,
            confidentiality, and liability clauses (or null).

        Raises:
            RuntimeError: If extraction fails after all retries.
            ValueError: If contract text is empty.
        """
        if not contract_text or not contract_text.strip():
            raise ValueError("Cannot extract clauses from empty text")

        if self._fits_single_pass(contract_text):
            logger.info("Using single-pass extraction")
            return self._extract_single_pass(contract_text)
        else:
            logger.info("Contract too long, using chunked extraction")
            return self._extract_chunked(contract_text)
