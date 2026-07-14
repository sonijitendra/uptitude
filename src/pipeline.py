"""
End-to-end pipeline orchestrator for legal contract analysis.

Coordinates all pipeline stages:
    1. Data loading from CUAD dataset
    2. PDF text extraction
    3. Text preprocessing
    4. LLM-based clause extraction
    5. LLM-based summarization
    6. Output generation (CSV + JSON)
"""

import csv
import json
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from config.settings import get_settings
from src.data_loader import CUADDataLoader, ContractDocument
from src.extractor import ClauseExtractor
from src.llm import LLMClient
from src.parser import PDFParser
from src.preprocessing import TextPreprocessor
from src.schemas import ContractResult, PipelineOutput
from src.summarizer import ContractSummarizer
from src.utils import setup_logging

logger = setup_logging(__name__)


class Pipeline:
    """End-to-end orchestrator for the contract analysis pipeline.

    Manages the complete lifecycle: loading contracts, extracting text,
    preprocessing, running LLM extraction and summarization, and
    writing structured outputs.

    Example:
        >>> pipeline = Pipeline()
        >>> output = pipeline.run()
        >>> print(f"Processed {output.successful} contracts")
    """

    def __init__(
        self,
        num_contracts: Optional[int] = None,
        use_few_shot: bool = True,
    ) -> None:
        """Initialize pipeline with all components.

        Args:
            num_contracts: Override for number of contracts to process.
            use_few_shot: Whether to use few-shot examples in extraction prompts.
        """
        self._settings = get_settings()
        self._num_contracts = num_contracts or self._settings.num_contracts

        # Initialize components
        self._loader = CUADDataLoader()
        self._parser = PDFParser()
        self._preprocessor = TextPreprocessor()

        # Share a single LLM client across extractor and summarizer
        self._llm_client = LLMClient()
        self._extractor = ClauseExtractor(
            llm_client=self._llm_client,
            use_few_shot=use_few_shot,
        )
        self._summarizer = ContractSummarizer(
            llm_client=self._llm_client,
        )

        logger.info(
            "Pipeline initialized | contracts=%d | model=%s | few_shot=%s",
            self._num_contracts,
            self._settings.llm_model,
            use_few_shot,
        )

    def _process_single_contract(
        self,
        contract: ContractDocument,
    ) -> Optional[ContractResult]:
        """Process a single contract through the full pipeline.

        Args:
            contract: Contract document with PDF bytes.

        Returns:
            ContractResult if successful, None if processing failed.
        """
        try:
            # ── Step 1: Parse PDF ─────────────────────────────────────
            self._parser.parse_contract(contract)
            if not contract.text:
                logger.warning(
                    "Skipping %s: no text extracted", contract.contract_id
                )
                return None

            # ── Step 2: Preprocess Text ───────────────────────────────
            clean_text = self._preprocessor.normalize(contract.text)
            if not clean_text:
                logger.warning(
                    "Skipping %s: text empty after preprocessing",
                    contract.contract_id,
                )
                return None

            logger.info(
                "Processing %s (%d chars)",
                contract.contract_id, len(clean_text),
            )

            # ── Step 3: Extract Clauses ───────────────────────────────
            clauses = self._extractor.extract(clean_text)

            # ── Step 4: Generate Summary ──────────────────────────────
            summary_response = self._summarizer.summarize(clean_text)

            # ── Step 5: Build Result ──────────────────────────────────
            result = ContractResult(
                contract_id=contract.contract_id,
                summary=summary_response.summary,
                termination_clause=clauses.termination_clause,
                confidentiality_clause=clauses.confidentiality_clause,
                liability_clause=clauses.liability_clause,
            )

            logger.info("✓ Completed: %s", contract.contract_id)
            return result

        except Exception as exc:
            logger.error(
                "✗ Failed processing %s: %s",
                contract.contract_id, exc,
            )
            return None

    def _write_csv(self, results: List[ContractResult], path: Path) -> None:
        """Write results to CSV file.

        Args:
            results: List of contract results to write.
            path: Output file path.
        """
        fieldnames = [
            "contract_id",
            "summary",
            "termination_clause",
            "confidentiality_clause",
            "liability_clause",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())

        logger.info("CSV output written to: %s", path)

    def _write_json(self, results: List[ContractResult], path: Path) -> None:
        """Write results to JSON file.

        Args:
            results: List of contract results to write.
            path: Output file path.
        """
        output = [result.to_dict() for result in results]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info("JSON output written to: %s", path)

    def run(self) -> PipelineOutput:
        """Execute the complete pipeline.

        Returns:
            PipelineOutput with aggregated results and statistics.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Starting Contract Analysis Pipeline")
        logger.info("=" * 60)

        # ── Load Contracts ────────────────────────────────────────────
        contracts = self._loader.load(self._num_contracts)

        if not contracts:
            logger.error("No contracts loaded. Aborting pipeline.")
            return PipelineOutput(
                total_contracts=0, successful=0, failed=0, results=[]
            )

        # ── Process Each Contract ─────────────────────────────────────
        results: List[ContractResult] = []
        failed_count = 0

        for contract in tqdm(contracts, desc="Processing contracts"):
            result = self._process_single_contract(contract)
            if result:
                results.append(result)
            else:
                failed_count += 1

        # ── Write Outputs ─────────────────────────────────────────────
        output_dir = self._settings.output_path

        if results:
            csv_path = output_dir / "results.csv"
            json_path = output_dir / "results.json"
            self._write_csv(results, csv_path)
            self._write_json(results, json_path)
        else:
            logger.warning("No results to write — all contracts failed")

        # ── Summary Statistics ────────────────────────────────────────
        elapsed = time.time() - start_time
        output = PipelineOutput(
            total_contracts=len(contracts),
            successful=len(results),
            failed=failed_count,
            results=results,
        )

        logger.info("=" * 60)
        logger.info("Pipeline Complete")
        logger.info("  Total:      %d contracts", output.total_contracts)
        logger.info("  Successful: %d", output.successful)
        logger.info("  Failed:     %d", output.failed)
        logger.info("  Duration:   %.1f seconds", elapsed)
        logger.info("  Output:     %s", output_dir)
        logger.info("=" * 60)

        return output
