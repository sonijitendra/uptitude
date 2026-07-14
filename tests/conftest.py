"""
Shared test fixtures and configuration for pytest.

Provides reusable sample data, mock objects, and test utilities
used across the test suite.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Environment Setup — ensure tests don't need a real API key
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Set mock environment variables for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-testing-only")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("NUM_CONTRACTS", "5")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")


@pytest.fixture
def clear_settings_cache():
    """Clear the settings LRU cache between tests."""
    from config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SAMPLE_CONTRACT_TEXT = """
MASTER SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is entered into as of January 1, 2024
("Effective Date") by and between Acme Corporation, a Delaware corporation ("Client"),
and TechServ Inc., a California corporation ("Provider").

1. SERVICES
Provider shall provide the software development services described in each Statement
of Work ("SOW") executed by both parties. Each SOW shall be incorporated into and
governed by this Agreement.

2. TERM AND TERMINATION
2.1 Term. This Agreement shall commence on the Effective Date and continue for a
period of three (3) years, unless earlier terminated as provided herein.

2.2 Termination for Convenience. Either party may terminate this Agreement upon
sixty (60) days' prior written notice to the other party.

2.3 Termination for Cause. Either party may terminate this Agreement immediately
upon written notice if the other party: (a) materially breaches this Agreement and
fails to cure such breach within thirty (30) days of receiving written notice; or
(b) becomes insolvent, files for bankruptcy, or ceases operations.

3. CONFIDENTIALITY
3.1 Definition. "Confidential Information" means any non-public information disclosed
by one party to the other, including but not limited to trade secrets, business plans,
technical data, and financial information.

3.2 Obligations. The receiving party shall: (a) hold all Confidential Information in
strict confidence; (b) not disclose Confidential Information to any third party without
prior written consent; (c) use Confidential Information only for the purposes of this
Agreement; and (d) protect Confidential Information with at least the same degree of
care used for its own confidential information.

3.3 Exceptions. Confidential Information does not include information that: (a) is or
becomes publicly available without breach; (b) was known prior to disclosure; or
(c) is independently developed without reference to Confidential Information.

4. LIMITATION OF LIABILITY
4.1 IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL,
CONSEQUENTIAL OR PUNITIVE DAMAGES ARISING OUT OF OR RELATED TO THIS AGREEMENT.

4.2 THE TOTAL AGGREGATE LIABILITY OF EITHER PARTY UNDER THIS AGREEMENT SHALL NOT
EXCEED THE TOTAL AMOUNTS PAID OR PAYABLE BY CLIENT TO PROVIDER IN THE TWELVE (12)
MONTHS PRECEDING THE EVENT GIVING RISE TO THE CLAIM.

5. GOVERNING LAW
This Agreement shall be governed by the laws of the State of California.
"""

SAMPLE_MESSY_TEXT = """
\x00Some text with null bytes\x00
   Excessive    whitespace   here
\u201cSmart quotes\u201d and \u2013 em dashes
Page 1
Page 2 of 10

- 3 -

Multiple


blank


lines

A sentence that breaks
mid-line because of pdf extraction.
"""

SAMPLE_CLAUSE_RESPONSE = {
    "termination_clause": "Either party may terminate this Agreement upon sixty (60) days' prior written notice to the other party.",
    "confidentiality_clause": "The receiving party shall hold all Confidential Information in strict confidence.",
    "liability_clause": "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL OR PUNITIVE DAMAGES.",
}

SAMPLE_SUMMARY_RESPONSE = {
    "summary": (
        "This Master Services Agreement between Acme Corporation (Client) and "
        "TechServ Inc. (Provider) governs software development services over a "
        "three-year term. The Provider delivers services per executed Statements "
        "of Work. Either party may terminate with 60 days' notice or immediately "
        "for material breach uncured within 30 days. Both parties must protect "
        "confidential information including trade secrets and business plans, "
        "restricting disclosure to third parties. Liability is capped at amounts "
        "paid in the preceding 12 months, with indirect and consequential damages "
        "excluded. The agreement is governed by California law. Key risks include "
        "liability limitations that cap recoverable damages and automatic "
        "termination upon insolvency or bankruptcy of either party."
    ),
}


@pytest.fixture
def sample_contract_text() -> str:
    """Return sample contract text for testing."""
    return SAMPLE_CONTRACT_TEXT


@pytest.fixture
def sample_messy_text() -> str:
    """Return messy PDF-extracted text for testing."""
    return SAMPLE_MESSY_TEXT


@pytest.fixture
def sample_clause_response() -> dict:
    """Return sample clause extraction response."""
    return SAMPLE_CLAUSE_RESPONSE.copy()


@pytest.fixture
def sample_summary_response() -> dict:
    """Return sample summary response."""
    return SAMPLE_SUMMARY_RESPONSE.copy()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Return minimal valid PDF bytes for parser testing.

    Creates a minimal PDF with a single page containing test text.
    """
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a test contract.\n\nSection 1: Terms and Conditions.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
