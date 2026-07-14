"""
OpenAI LLM client with retry logic, structured JSON parsing, and error handling.

Provides a resilient interface to the OpenAI API with:
    - Automatic retries on transient failures
    - Structured JSON response parsing
    - Token-aware request management
    - Rate limit handling with exponential backoff
"""

import json
import time
from typing import Any, Type, TypeVar

from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from pydantic import BaseModel, ValidationError

from config.settings import get_settings
from src.utils import setup_logging, count_tokens

logger = setup_logging(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Resilient OpenAI API client with structured output parsing.

    Wraps the OpenAI chat completion API with automatic retry logic,
    JSON response parsing, and Pydantic schema validation.

    Example:
        >>> from src.schemas import ClauseExtractionResponse
        >>> client = LLMClient()
        >>> result = client.generate(
        ...     system_prompt="You are a legal analyst.",
        ...     user_prompt="Extract clauses from...",
        ...     response_model=ClauseExtractionResponse,
        ... )
        >>> print(result.termination_clause)
    """

    def __init__(self) -> None:
        """Initialize the LLM client with application settings."""
        self._settings = get_settings()
        self._client = OpenAI(api_key=self._settings.openai_api_key)

        logger.info(
            "LLM client initialized | model=%s | temp=%.1f | max_tokens=%d",
            self._settings.llm_model,
            self._settings.llm_temperature,
            self._settings.llm_max_tokens,
        )

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Make a single API call to OpenAI.

        Args:
            system_prompt: System-level instruction.
            user_prompt: User-level prompt with content.

        Returns:
            Raw response content string.

        Raises:
            APIError: On non-retryable API errors.
        """
        response = self._client.chat.completions.create(
            model=self._settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content is None:
            raise APIError(
                message="Empty response from API",
                request=None,
                body=None,
            )

        # Log token usage
        if response.usage:
            logger.debug(
                "Token usage: prompt=%d, completion=%d, total=%d",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
            )

        return content

    def _parse_json(self, raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues.

        Strips markdown code fences if present, then parses JSON.

        Args:
            raw: Raw response string from LLM.

        Returns:
            Parsed dictionary.

        Raises:
            json.JSONDecodeError: If response is not valid JSON.
        """
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (``` markers)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        return json.loads(text)

    def _validate_response(
        self,
        data: dict[str, Any],
        response_model: Type[T],
    ) -> T:
        """Validate parsed JSON against a Pydantic model.

        Args:
            data: Parsed JSON dictionary.
            response_model: Pydantic model class for validation.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValidationError: If data doesn't match schema.
        """
        return response_model.model_validate(data)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> T:
        """Generate a structured response from the LLM with retry logic.

        Makes API calls with automatic retry on:
            - Rate limit errors (exponential backoff)
            - Connection errors (with delay)
            - Malformed JSON responses (re-prompt)
            - Schema validation failures (re-prompt)

        Args:
            system_prompt: System-level instruction.
            user_prompt: User-level prompt with content.
            response_model: Pydantic model class to validate response against.

        Returns:
            Validated Pydantic model instance.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        max_retries = self._settings.llm_max_retries
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug("LLM call attempt %d/%d", attempt, max_retries)

                # ── API Call ──────────────────────────────────────────
                raw_response = self._call_api(system_prompt, user_prompt)

                # ── Parse JSON ────────────────────────────────────────
                parsed = self._parse_json(raw_response)

                # ── Validate Schema ───────────────────────────────────
                result = self._validate_response(parsed, response_model)

                logger.debug("LLM response validated successfully")
                return result

            except RateLimitError as exc:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    "Rate limited (attempt %d/%d). Waiting %ds: %s",
                    attempt, max_retries, wait_time, exc,
                )
                last_error = exc
                time.sleep(wait_time)

            except APIConnectionError as exc:
                wait_time = min(2 ** attempt, 30)
                logger.warning(
                    "Connection error (attempt %d/%d). Waiting %ds: %s",
                    attempt, max_retries, wait_time, exc,
                )
                last_error = exc
                time.sleep(wait_time)

            except json.JSONDecodeError as exc:
                logger.warning(
                    "Malformed JSON (attempt %d/%d): %s",
                    attempt, max_retries, exc,
                )
                last_error = exc

            except ValidationError as exc:
                logger.warning(
                    "Schema validation failed (attempt %d/%d): %s",
                    attempt, max_retries, exc,
                )
                last_error = exc

            except APIError as exc:
                logger.error(
                    "API error (attempt %d/%d): %s",
                    attempt, max_retries, exc,
                )
                last_error = exc
                # Don't retry on 4xx client errors (except rate limits)
                if hasattr(exc, "status_code") and 400 <= exc.status_code < 500:
                    break
                time.sleep(2)

        raise RuntimeError(
            f"LLM call failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def generate_raw(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a raw string response without schema validation.

        Useful for freeform text generation where structured output
        is not needed.

        Args:
            system_prompt: System-level instruction.
            user_prompt: User-level prompt with content.

        Returns:
            Raw response string.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        max_retries = self._settings.llm_max_retries
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                return self._call_api(system_prompt, user_prompt)

            except (RateLimitError, APIConnectionError) as exc:
                wait_time = 2 ** attempt
                logger.warning(
                    "Retryable error (attempt %d/%d). Waiting %ds: %s",
                    attempt, max_retries, wait_time, exc,
                )
                last_error = exc
                time.sleep(wait_time)

            except APIError as exc:
                logger.error("API error: %s", exc)
                last_error = exc
                if hasattr(exc, "status_code") and 400 <= exc.status_code < 500:
                    break
                time.sleep(2)

        raise RuntimeError(
            f"LLM call failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
