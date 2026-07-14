"""
PDF text extraction with dual-engine fallback.

Primary engine: PyMuPDF (fitz) — fast and reliable.
Fallback engine: pdfplumber — better for complex layouts.

Both engines extract text from in-memory PDF bytes without
requiring files on disk.
"""

import io
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

from src.data_loader import ContractDocument
from src.utils import setup_logging

logger = setup_logging(__name__)


class PDFParser:
    """Extracts text from PDF documents using a dual-engine strategy.

    Tries PyMuPDF first for speed and reliability. Falls back to
    pdfplumber if PyMuPDF produces insufficient text (< min_chars)
    or raises an error.

    Example:
        >>> parser = PDFParser()
        >>> text = parser.extract_text(contract.pdf_bytes)
        >>> print(text[:200])
    """

    # Minimum characters to consider extraction successful
    MIN_CHARS_THRESHOLD: int = 100

    def __init__(self, min_chars: int = MIN_CHARS_THRESHOLD) -> None:
        """Initialize the PDF parser.

        Args:
            min_chars: Minimum character count to accept PyMuPDF output
                before falling back to pdfplumber.
        """
        self._min_chars = min_chars

    def extract_text_pymupdf(self, pdf_bytes: bytes) -> str:
        """Extract text using PyMuPDF (fitz).

        Args:
            pdf_bytes: Raw PDF file bytes.

        Returns:
            Extracted text from all pages, concatenated.

        Raises:
            RuntimeError: If PyMuPDF cannot open or parse the PDF.
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages: list[str] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    pages.append(text)

            doc.close()
            full_text = "\n\n".join(pages)

            logger.debug(
                "PyMuPDF extracted %d chars from %d pages",
                len(full_text), len(pages),
            )
            return full_text

        except Exception as exc:
            raise RuntimeError(f"PyMuPDF extraction failed: {exc}") from exc

    def extract_text_pdfplumber(self, pdf_bytes: bytes) -> str:
        """Extract text using pdfplumber (fallback engine).

        Better at handling complex layouts, tables, and
        multi-column PDFs.

        Args:
            pdf_bytes: Raw PDF file bytes.

        Returns:
            Extracted text from all pages, concatenated.

        Raises:
            RuntimeError: If pdfplumber cannot open or parse the PDF.
        """
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            pages: list[str] = []

            with pdfplumber.open(pdf_stream) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append(text)

            full_text = "\n\n".join(pages)

            logger.debug(
                "pdfplumber extracted %d chars from %d pages",
                len(full_text), len(pages),
            )
            return full_text

        except Exception as exc:
            raise RuntimeError(f"pdfplumber extraction failed: {exc}") from exc

    def extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes with automatic fallback.

        Strategy:
            1. Try PyMuPDF (fast)
            2. If result < min_chars, try pdfplumber
            3. Return the longer/better result

        Args:
            pdf_bytes: Raw PDF file bytes.

        Returns:
            Extracted text. Empty string if both engines fail.
        """
        pymupdf_text: Optional[str] = None
        pdfplumber_text: Optional[str] = None

        # ── Primary: PyMuPDF ──────────────────────────────────────────
        try:
            pymupdf_text = self.extract_text_pymupdf(pdf_bytes)
        except RuntimeError as exc:
            logger.warning("PyMuPDF failed, trying pdfplumber: %s", exc)

        # Check if PyMuPDF result is sufficient
        if pymupdf_text and len(pymupdf_text.strip()) >= self._min_chars:
            return pymupdf_text

        # ── Fallback: pdfplumber ──────────────────────────────────────
        logger.info("PyMuPDF output insufficient, falling back to pdfplumber")
        try:
            pdfplumber_text = self.extract_text_pdfplumber(pdf_bytes)
        except RuntimeError as exc:
            logger.warning("pdfplumber also failed: %s", exc)

        # ── Select best result ────────────────────────────────────────
        if pdfplumber_text and pymupdf_text:
            # Return whichever produced more text
            if len(pdfplumber_text) > len(pymupdf_text):
                logger.info("Using pdfplumber output (longer)")
                return pdfplumber_text
            return pymupdf_text

        if pdfplumber_text:
            return pdfplumber_text

        if pymupdf_text:
            return pymupdf_text

        logger.error("Both PDF engines failed to extract text")
        return ""

    def parse_contract(self, contract: ContractDocument) -> ContractDocument:
        """Extract text from a ContractDocument and store it in-place.

        Convenience method that extracts text from the contract's
        PDF bytes and sets the .text attribute.

        Args:
            contract: ContractDocument with pdf_bytes populated.

        Returns:
            The same ContractDocument with .text populated.
        """
        logger.info("Parsing contract: %s", contract.contract_id)
        contract.text = self.extract_text(contract.pdf_bytes)

        if not contract.text:
            logger.warning(
                "No text extracted from contract: %s", contract.contract_id
            )
        else:
            logger.info(
                "Extracted %d characters from %s",
                len(contract.text), contract.contract_id,
            )

        return contract
