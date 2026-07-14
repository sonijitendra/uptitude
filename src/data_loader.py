"""
CUAD dataset loader using HuggingFace datasets library.

Loads contract PDFs from the dvgodoy/CUAD_v1_Contract_Understanding_PDF
dataset and provides them as decoded byte streams for PDF parsing.
"""

import base64
import io
from dataclasses import dataclass, field
from typing import List, Iterator

from datasets import load_dataset, Dataset

from config.settings import get_settings
from src.utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class ContractDocument:
    """Represents a single contract loaded from the CUAD dataset.

    Attributes:
        contract_id: Unique identifier (derived from filename).
        file_name: Original PDF filename.
        pdf_bytes: Raw PDF bytes decoded from base64.
        text: Extracted text (populated later by parser).
    """

    contract_id: str
    file_name: str
    pdf_bytes: bytes
    text: str = ""
    metadata: dict = field(default_factory=dict)


class CUADDataLoader:
    """Loads and serves contracts from the CUAD HuggingFace dataset.

    Downloads the dataset on first use (cached by HuggingFace),
    selects the configured number of contracts, and yields
    ContractDocument instances with decoded PDF bytes.

    Example:
        >>> loader = CUADDataLoader()
        >>> contracts = loader.load()
        >>> for contract in contracts:
        ...     print(contract.contract_id, len(contract.pdf_bytes))
    """

    def __init__(self) -> None:
        """Initialize the data loader with application settings."""
        self._settings = get_settings()
        self._dataset: Dataset | None = None

    def _load_dataset(self) -> Dataset:
        """Download and cache the HuggingFace dataset.

        Returns:
            The training split of the CUAD PDF dataset.

        Raises:
            ConnectionError: If HuggingFace Hub is unreachable.
            ValueError: If dataset structure is unexpected.
        """
        if self._dataset is not None:
            return self._dataset

        logger.info(
            "Loading CUAD dataset from HuggingFace: %s",
            self._settings.hf_dataset,
        )

        try:
            ds = load_dataset(
                self._settings.hf_dataset,
                split="train",
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.error("Failed to load dataset: %s", exc)
            raise ConnectionError(
                f"Could not load dataset '{self._settings.hf_dataset}'. "
                f"Check your internet connection and dataset name."
            ) from exc

        # Validate expected columns exist
        expected_columns = {"file_name", "pdf_bytes_base64"}
        actual_columns = set(ds.column_names)
        if not expected_columns.issubset(actual_columns):
            missing = expected_columns - actual_columns
            raise ValueError(
                f"Dataset missing expected columns: {missing}. "
                f"Available columns: {actual_columns}"
            )

        self._dataset = ds
        logger.info("Dataset loaded successfully: %d total contracts", len(ds))
        return ds

    def _decode_pdf(self, base64_string: str) -> bytes:
        """Decode base64-encoded PDF string to raw bytes.

        Args:
            base64_string: Base64-encoded PDF content.

        Returns:
            Decoded PDF bytes.

        Raises:
            ValueError: If base64 decoding fails.
        """
        try:
            return base64.b64decode(base64_string)
        except Exception as exc:
            raise ValueError(f"Failed to decode base64 PDF: {exc}") from exc

    def _make_contract_id(self, file_name: str) -> str:
        """Generate a clean contract ID from the filename.

        Removes file extension and normalizes characters to create
        a readable, unique identifier.

        Args:
            file_name: Original PDF filename.

        Returns:
            Cleaned contract identifier string.
        """
        # Remove .pdf extension and clean up
        contract_id = file_name.replace(".pdf", "").replace(".PDF", "")
        # Replace problematic characters
        contract_id = contract_id.replace(" ", "_").replace("/", "_")
        return contract_id

    def load(self, num_contracts: int | None = None) -> List[ContractDocument]:
        """Load and return a list of contract documents.

        Downloads the dataset (if not cached), selects the configured
        number of contracts, decodes PDFs from base64.

        Args:
            num_contracts: Override for number of contracts to load.
                Defaults to settings value.

        Returns:
            List of ContractDocument instances with decoded PDF bytes.
        """
        n = num_contracts or self._settings.num_contracts
        dataset = self._load_dataset()

        # Cap at available contracts
        n = min(n, len(dataset))
        logger.info("Loading %d contracts from dataset", n)

        contracts: List[ContractDocument] = []

        for i in range(n):
            row = dataset[i]
            file_name = row["file_name"]

            try:
                pdf_bytes = self._decode_pdf(row["pdf_bytes_base64"])
                contract = ContractDocument(
                    contract_id=self._make_contract_id(file_name),
                    file_name=file_name,
                    pdf_bytes=pdf_bytes,
                )
                contracts.append(contract)
                logger.debug(
                    "Loaded contract %d/%d: %s (%d bytes)",
                    i + 1, n, file_name, len(pdf_bytes),
                )
            except Exception as exc:
                logger.warning(
                    "Skipping contract '%s': %s", file_name, exc
                )
                continue

        logger.info(
            "Successfully loaded %d/%d contracts", len(contracts), n
        )
        return contracts

    def iterate(self, num_contracts: int | None = None) -> Iterator[ContractDocument]:
        """Lazily iterate over contract documents.

        Memory-efficient alternative to load() for large batches.

        Args:
            num_contracts: Override for number of contracts.

        Yields:
            ContractDocument instances one at a time.
        """
        n = num_contracts or self._settings.num_contracts
        dataset = self._load_dataset()
        n = min(n, len(dataset))

        for i in range(n):
            row = dataset[i]
            file_name = row["file_name"]

            try:
                pdf_bytes = self._decode_pdf(row["pdf_bytes_base64"])
                yield ContractDocument(
                    contract_id=self._make_contract_id(file_name),
                    file_name=file_name,
                    pdf_bytes=pdf_bytes,
                )
            except Exception as exc:
                logger.warning("Skipping contract '%s': %s", file_name, exc)
                continue
