"""Unit tests for text preprocessing module."""

import pytest

from src.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Tests for the text normalization pipeline."""

    def setup_method(self):
        """Initialize preprocessor for each test."""
        self.preprocessor = TextPreprocessor()

    # ── Unicode Normalization ─────────────────────────────────────

    def test_smart_quotes_replaced(self):
        """Smart quotes should be converted to straight quotes."""
        text = "\u201cHello\u201d and \u2018world\u2019"
        result = self.preprocessor.normalize_unicode(text)
        assert '"Hello"' in result
        assert "'world'" in result

    def test_em_dash_replaced(self):
        """Em and en dashes should be replaced with hyphens."""
        text = "clause \u2013 section \u2014 paragraph"
        result = self.preprocessor.normalize_unicode(text)
        assert "\u2013" not in result
        assert "\u2014" not in result
        assert "-" in result

    def test_non_breaking_space_replaced(self):
        """Non-breaking spaces should become regular spaces."""
        text = "word\u00a0word"
        result = self.preprocessor.normalize_unicode(text)
        assert "\u00a0" not in result
        assert "word word" in result

    def test_zero_width_chars_removed(self):
        """Zero-width characters should be stripped."""
        text = "te\u200bst\u200cwo\u200drd"
        result = self.preprocessor.normalize_unicode(text)
        assert result == "testword"

    # ── Null Byte Removal ─────────────────────────────────────────

    def test_null_bytes_removed(self):
        """Null bytes and control characters should be removed."""
        text = "hello\x00world\x01test"
        result = self.preprocessor.remove_null_bytes(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "helloworld" in result

    def test_newlines_preserved(self):
        """Newlines should NOT be removed as control characters."""
        text = "line1\nline2\tindented"
        result = self.preprocessor.remove_null_bytes(text)
        assert "\n" in result
        assert "\t" in result

    # ── Whitespace Normalization ──────────────────────────────────

    def test_multiple_spaces_collapsed(self):
        """Multiple consecutive spaces should collapse to one."""
        text = "too    many     spaces"
        result = self.preprocessor.normalize_whitespace(text)
        assert "too many spaces" in result

    def test_paragraph_breaks_preserved(self):
        """Double newlines (paragraph breaks) should be preserved."""
        text = "paragraph one\n\nparagraph two"
        result = self.preprocessor.normalize_whitespace(text)
        assert "\n\n" in result

    def test_excessive_newlines_reduced(self):
        """3+ consecutive newlines should be reduced to double."""
        text = "section one\n\n\n\n\nsection two"
        result = self.preprocessor.normalize_whitespace(text)
        # Should have exactly one paragraph break, not more
        assert "\n\n\n" not in result
        assert "\n\n" in result

    # ── Page Artifacts ────────────────────────────────────────────

    def test_page_numbers_removed(self):
        """Standalone page numbers should be removed."""
        text = "Some text\n42\nMore text"
        result = self.preprocessor.remove_page_artifacts(text)
        assert "\n42\n" not in result

    def test_page_x_of_y_removed(self):
        """'Page X of Y' patterns should be removed."""
        text = "Content here\nPage 5 of 20\nMore content"
        result = self.preprocessor.remove_page_artifacts(text)
        assert "Page 5 of 20" not in result

    def test_dash_page_numbers_removed(self):
        """'- N -' page markers should be removed."""
        text = "Content\n- 3 -\nMore content"
        result = self.preprocessor.remove_page_artifacts(text)
        assert "- 3 -" not in result

    def test_form_feeds_replaced(self):
        """Form feed characters should be replaced with paragraph breaks."""
        text = "page one\fpage two"
        result = self.preprocessor.remove_page_artifacts(text)
        assert "\f" not in result
        assert "\n\n" in result

    # ── Full Pipeline ─────────────────────────────────────────────

    def test_full_normalize(self, clear_settings_cache):
        """Full normalization pipeline should handle all issues."""
        messy = (
            "\x00Smart \u201cquotes\u201d and\x00\n"
            "   extra   spaces\n\n\n\n\n"
            "Page 1\n"
            "Normal text continues."
        )
        result = self.preprocessor.normalize(messy)

        assert "\x00" not in result
        assert "\u201c" not in result
        assert "  " not in result  # No double spaces
        assert "\n\n\n" not in result
        assert "Normal text continues" in result

    def test_empty_text_returns_empty(self, clear_settings_cache):
        """Empty input should return empty string."""
        assert self.preprocessor.normalize("") == ""
        assert self.preprocessor.normalize("   ") == ""

    def test_preserves_legal_content(self, clear_settings_cache):
        """Should preserve important legal text and structure."""
        text = (
            "Section 5. TERMINATION\n\n"
            "Either party may terminate this Agreement upon thirty (30) "
            "days' prior written notice.\n\n"
            "Section 6. GOVERNING LAW"
        )
        result = self.preprocessor.normalize(text)
        assert "TERMINATION" in result
        assert "thirty (30)" in result
        assert "GOVERNING LAW" in result
