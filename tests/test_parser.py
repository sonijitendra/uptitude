"""Unit tests for PDF parser module."""

import pytest

from src.parser import PDFParser


class TestPDFParser:
    """Tests for the dual-engine PDF parser."""

    def setup_method(self):
        """Initialize parser for each test."""
        self.parser = PDFParser()

    def test_extract_text_pymupdf(self, sample_pdf_bytes, clear_settings_cache):
        """PyMuPDF should extract text from valid PDF bytes."""
        text = self.parser.extract_text_pymupdf(sample_pdf_bytes)
        assert len(text) > 0
        assert "test contract" in text.lower()

    def test_extract_text_pdfplumber(self, sample_pdf_bytes, clear_settings_cache):
        """pdfplumber should extract text from valid PDF bytes."""
        text = self.parser.extract_text_pdfplumber(sample_pdf_bytes)
        assert len(text) > 0
        assert "test contract" in text.lower()

    def test_extract_text_with_fallback(self, sample_pdf_bytes, clear_settings_cache):
        """Main extract_text should return text from either engine."""
        text = self.parser.extract_text(sample_pdf_bytes)
        assert len(text) > 0
        assert "test contract" in text.lower()

    def test_extract_text_invalid_bytes(self, clear_settings_cache):
        """Should return empty string for invalid PDF bytes."""
        text = self.parser.extract_text(b"not a pdf")
        assert text == ""

    def test_extract_text_empty_bytes(self, clear_settings_cache):
        """Should return empty string for empty bytes."""
        text = self.parser.extract_text(b"")
        assert text == ""

    def test_pymupdf_raises_on_invalid(self, clear_settings_cache):
        """PyMuPDF engine should raise RuntimeError on invalid input."""
        with pytest.raises(RuntimeError):
            self.parser.extract_text_pymupdf(b"not a pdf")

    def test_min_chars_threshold(self, clear_settings_cache):
        """Parser with high threshold should fall back to pdfplumber."""
        # Create parser with impossibly high threshold
        parser = PDFParser(min_chars=999999)
        # Shouldn't crash — should try fallback
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Short text.")
        pdf_bytes = doc.tobytes()
        doc.close()

        text = parser.extract_text(pdf_bytes)
        assert isinstance(text, str)
