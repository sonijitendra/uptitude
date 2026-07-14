"""
Centralized configuration management using Pydantic Settings.

Loads configuration from environment variables and .env file with
type validation, sensible defaults, and clear documentation.
"""

from pathlib import Path
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file.

    Attributes:
        openai_api_key: OpenAI API key for LLM calls.
        llm_model: Model identifier (e.g. gpt-4o-mini).
        llm_max_tokens: Maximum tokens in LLM response.
        llm_temperature: Sampling temperature (0.0 = deterministic).
        llm_max_retries: Max retry attempts for malformed LLM responses.
        num_contracts: Number of CUAD contracts to process.
        hf_dataset: HuggingFace dataset identifier for CUAD PDFs.
        chunk_size: Max characters per chunk for long contracts.
        chunk_overlap: Overlap between adjacent chunks in characters.
        embedding_model: Sentence-transformer model name.
        search_top_k: Number of similar clauses to retrieve.
        output_dir: Directory path for output files.
        log_level: Logging verbosity level.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model identifier",
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum tokens for LLM response",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max retry attempts for malformed responses",
    )

    # ── Data ──────────────────────────────────────────────────────────────
    num_contracts: int = Field(
        default=50,
        ge=1,
        le=510,
        description="Number of contracts to process from CUAD",
    )
    hf_dataset: str = Field(
        default="dvgodoy/CUAD_v1_Contract_Understanding_PDF",
        description="HuggingFace dataset identifier",
    )

    # ── Processing ────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=12000,
        ge=1000,
        le=100000,
        description="Max characters per chunk for long contracts",
    )
    chunk_overlap: int = Field(
        default=500,
        ge=0,
        le=5000,
        description="Character overlap between adjacent chunks",
    )

    # ── Embeddings (Bonus) ────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model for embeddings",
    )
    search_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of similar clauses to retrieve",
    )

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = Field(
        default="outputs",
        description="Directory for output files",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(
                f"Invalid log level '{v}'. Must be one of: {valid_levels}"
            )
        return upper

    @property
    def output_path(self) -> Path:
        """Resolved output directory path."""
        path = PROJECT_ROOT / self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_path(self) -> Path:
        """Resolved data directory path."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return DATA_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance.

    Uses lru_cache to ensure settings are loaded only once
    and shared across the application.

    Returns:
        Settings: Validated application configuration.

    Raises:
        ValidationError: If required environment variables are missing
            or values fail validation.
    """
    return Settings()
