"""
Text preprocessing and normalization for legal contracts.

Handles common issues in PDF-extracted text:
    - Excessive whitespace and blank lines
    - Unicode normalization (smart quotes, em-dashes, etc.)
    - Header/footer artifacts
    - Page number removal
    - Paragraph boundary preservation
"""

import re
import unicodedata

from src.utils import setup_logging

logger = setup_logging(__name__)


class TextPreprocessor:
    """Cleans and normalizes raw text extracted from legal contract PDFs.

    Applies a pipeline of normalization steps designed specifically
    for legal document text, preserving paragraph structure while
    removing extraction artifacts.

    Example:
        >>> preprocessor = TextPreprocessor()
        >>> clean = preprocessor.normalize("  Some\\x00 messy   text\\n\\n\\n\\n")
        >>> print(clean)
        Some messy text
    """

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to their canonical forms.

        Converts:
            - Smart quotes → straight quotes
            - Em/en dashes → standard hyphens
            - Non-breaking spaces → regular spaces
            - Ligatures → expanded form (fi → fi)

        Args:
            text: Raw input text.

        Returns:
            Unicode-normalized text.
        """
        # NFKD decomposition then NFC recomposition
        text = unicodedata.normalize("NFKC", text)

        # Smart quotes → straight quotes
        replacements = {
            "\u2018": "'",   # Left single quote
            "\u2019": "'",   # Right single quote
            "\u201c": '"',   # Left double quote
            "\u201d": '"',   # Right double quote
            "\u2013": "-",   # En dash
            "\u2014": "-",   # Em dash
            "\u2015": "-",   # Horizontal bar
            "\u2026": "...", # Ellipsis
            "\u00a0": " ",   # Non-breaking space
            "\u200b": "",    # Zero-width space
            "\u200c": "",    # Zero-width non-joiner
            "\u200d": "",    # Zero-width joiner
            "\ufeff": "",    # BOM
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text

    def remove_null_bytes(self, text: str) -> str:
        """Remove null bytes and other control characters.

        Preserves newlines and tabs, which have structural meaning.

        Args:
            text: Input text potentially containing control chars.

        Returns:
            Cleaned text.
        """
        # Remove all control characters except \n, \r, \t
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph breaks.

        - Collapses multiple spaces/tabs into single spaces
        - Preserves double newlines as paragraph separators
        - Reduces 3+ consecutive newlines to double newlines

        Args:
            text: Input text with irregular whitespace.

        Returns:
            Text with normalized whitespace.
        """
        # Replace tabs with spaces
        text = text.replace("\t", " ")

        # Collapse multiple spaces (within lines) to single space
        text = re.sub(r"[^\S\n]+", " ", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Reduce 3+ consecutive newlines to paragraph break
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip trailing whitespace from each line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text

    def remove_page_artifacts(self, text: str) -> str:
        """Remove common PDF extraction artifacts.

        Removes:
            - Standalone page numbers
            - Page header/footer patterns
            - Form feed characters

        Args:
            text: Text with potential page artifacts.

        Returns:
            Cleaned text.
        """
        # Remove form feed characters
        text = text.replace("\f", "\n\n")

        # Remove standalone page numbers (e.g., lines that are just "12" or "Page 12")
        text = re.sub(
            r"^\s*(?:Page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$",
            "",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # Remove common header/footer patterns
        text = re.sub(
            r"^\s*-\s*\d+\s*-\s*$",
            "",
            text,
            flags=re.MULTILINE,
        )

        return text

    def remove_excessive_line_breaks(self, text: str) -> str:
        """Join lines that were split by PDF column/page boundaries.

        Heuristic: if a line ends without sentence-ending punctuation
        and the next line starts with a lowercase letter, join them.

        Args:
            text: Text with potential mid-sentence line breaks.

        Returns:
            Text with rejoined sentences.
        """
        # Join lines where previous line doesn't end with
        # sentence-ending punctuation and next starts with lowercase
        text = re.sub(
            r"(?<=[a-z,;])\n(?=[a-z])",
            " ",
            text,
        )

        return text

    def normalize(self, text: str) -> str:
        """Apply the full normalization pipeline.

        Runs all preprocessing steps in the correct order:
            1. Remove null/control bytes
            2. Normalize Unicode
            3. Remove page artifacts
            4. Normalize whitespace
            5. Rejoin broken sentences
            6. Final trim

        Args:
            text: Raw extracted text from PDF.

        Returns:
            Clean, normalized text ready for LLM processing.
        """
        if not text or not text.strip():
            logger.warning("Empty text received for normalization")
            return ""

        original_len = len(text)

        text = self.remove_null_bytes(text)
        text = self.normalize_unicode(text)
        text = self.remove_page_artifacts(text)
        text = self.normalize_whitespace(text)
        text = self.remove_excessive_line_breaks(text)
        text = text.strip()

        logger.debug(
            "Text normalized: %d → %d chars (%.1f%% reduction)",
            original_len,
            len(text),
            (1 - len(text) / max(original_len, 1)) * 100,
        )

        return text
