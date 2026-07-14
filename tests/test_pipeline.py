"""Integration tests for the end-to-end pipeline."""

from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import json
import csv
import tempfile

import pytest

from src.schemas import ClauseExtractionResponse, SummaryResponse, ContractResult
from src.pipeline import Pipeline


class TestPipelineIntegration:
    """Integration tests for the pipeline orchestrator."""

    @pytest.fixture
    def mock_pipeline(self, clear_settings_cache, tmp_path, monkeypatch):
        """Create a pipeline with all components mocked."""
        # Mock settings to use tmp directory for output
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

        with patch("src.pipeline.CUADDataLoader") as MockLoader, \
             patch("src.pipeline.PDFParser") as MockParser, \
             patch("src.pipeline.LLMClient") as MockLLM:

            from src.data_loader import ContractDocument

            # Setup mock data loader
            mock_contracts = [
                ContractDocument(
                    contract_id="test_contract_1",
                    file_name="test_1.pdf",
                    pdf_bytes=b"fake pdf bytes",
                    text="",
                ),
                ContractDocument(
                    contract_id="test_contract_2",
                    file_name="test_2.pdf",
                    pdf_bytes=b"fake pdf bytes",
                    text="",
                ),
            ]
            MockLoader.return_value.load.return_value = mock_contracts

            # Setup mock parser — sets text on contracts
            def mock_parse(contract):
                contract.text = "This is a test contract with terms and conditions. " * 50
                return contract
            MockParser.return_value.parse_contract.side_effect = mock_parse

            # Setup mock LLM responses
            mock_llm_instance = MockLLM.return_value

            def mock_generate(system_prompt, user_prompt, response_model):
                if response_model == ClauseExtractionResponse:
                    return ClauseExtractionResponse(
                        termination_clause="Terminable with 30 days notice.",
                        confidentiality_clause="All info is confidential.",
                        liability_clause=None,
                    )
                elif response_model == SummaryResponse:
                    return SummaryResponse(
                        summary="This contract governs the relationship between parties. "
                               "It establishes terms for service delivery and payment obligations. "
                               "Key provisions include termination rights with 30 days notice, "
                               "confidentiality protections for proprietary information, "
                               "and standard liability limitations. The agreement runs for three years "
                               "with automatic renewal unless terminated. Both parties agree to resolve "
                               "disputes through binding arbitration in the state of California."
                    )
                return None

            mock_llm_instance.generate.side_effect = mock_generate

            # Need to clear settings cache again after monkeypatch
            from config.settings import get_settings
            get_settings.cache_clear()

            pipeline = Pipeline(num_contracts=2)
            yield pipeline, tmp_path

    def test_pipeline_produces_csv_and_json(self, mock_pipeline):
        """Pipeline should produce both CSV and JSON output files."""
        pipeline, tmp_path = mock_pipeline

        output = pipeline.run()

        assert output.successful == 2
        assert output.failed == 0

        # Check CSV exists and has correct content
        csv_path = tmp_path / "results.csv"
        assert csv_path.exists()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert "contract_id" in rows[0]
            assert "summary" in rows[0]
            assert "termination_clause" in rows[0]

        # Check JSON exists and has correct content
        json_path = tmp_path / "results.json"
        assert json_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]["contract_id"] == "test_contract_1"

    def test_pipeline_output_fields(self, mock_pipeline):
        """Each result should have all required fields."""
        pipeline, _ = mock_pipeline

        output = pipeline.run()

        for result in output.results:
            assert result.contract_id
            assert result.summary
            assert isinstance(result.termination_clause, (str, type(None)))
            assert isinstance(result.confidentiality_clause, (str, type(None)))
            assert isinstance(result.liability_clause, (str, type(None)))

    def test_pipeline_handles_failure_gracefully(self, clear_settings_cache, tmp_path, monkeypatch):
        """Pipeline should continue processing when one contract fails."""
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

        with patch("src.pipeline.CUADDataLoader") as MockLoader, \
             patch("src.pipeline.PDFParser") as MockParser, \
             patch("src.pipeline.LLMClient") as MockLLM:

            from src.data_loader import ContractDocument
            from config.settings import get_settings
            get_settings.cache_clear()

            mock_contracts = [
                ContractDocument(
                    contract_id="good_contract",
                    file_name="good.pdf",
                    pdf_bytes=b"good",
                    text="",
                ),
                ContractDocument(
                    contract_id="bad_contract",
                    file_name="bad.pdf",
                    pdf_bytes=b"bad",
                    text="",
                ),
            ]
            MockLoader.return_value.load.return_value = mock_contracts

            call_count = 0
            def mock_parse(contract):
                nonlocal call_count
                call_count += 1
                if contract.contract_id == "bad_contract":
                    contract.text = ""  # Empty text = skip
                else:
                    contract.text = "Good contract text. " * 50
                return contract

            MockParser.return_value.parse_contract.side_effect = mock_parse

            mock_llm = MockLLM.return_value
            mock_llm.generate.side_effect = lambda sp, up, rm: (
                ClauseExtractionResponse(termination_clause="clause")
                if rm == ClauseExtractionResponse
                else SummaryResponse(summary="Summary of the contract covering all key aspects " * 4)
            )

            pipeline = Pipeline(num_contracts=2)
            output = pipeline.run()

            assert output.successful == 1
            assert output.failed == 1
