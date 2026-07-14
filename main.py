#!/usr/bin/env python3
"""
Legal Contract Analysis Pipeline — CLI Entry Point.

Analyzes legal contracts from the CUAD dataset using LLMs to:
    - Extract termination, confidentiality, and liability clauses
    - Generate concise contract summaries
    - Output structured results in CSV and JSON format
    - (Optional) Build semantic search index over clauses

Usage:
    python main.py                          # Run full pipeline (50 contracts)
    python main.py --num-contracts 10       # Process 10 contracts
    python main.py --no-few-shot            # Disable few-shot examples
    python main.py --search "termination"   # Semantic search after pipeline
"""

import argparse
import sys

from config.settings import get_settings
from src.pipeline import Pipeline
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Legal Contract Analysis Pipeline — "
                    "Extract clauses and generate summaries from CUAD contracts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        Process 50 contracts (default)
  python main.py -n 5                   Process 5 contracts (quick test)
  python main.py --no-few-shot          Disable few-shot examples
  python main.py --search "liability"   Run semantic search after pipeline
        """,
    )

    parser.add_argument(
        "-n", "--num-contracts",
        type=int,
        default=None,
        help="Number of contracts to process (default: from .env or 50)",
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        default=False,
        help="Disable few-shot examples in extraction prompts",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Run semantic search with this query after pipeline completes",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        default=False,
        help="Skip pipeline, only run semantic search on existing results",
    )

    return parser.parse_args()


def run_semantic_search(query: str) -> None:
    """Run semantic search over previously extracted clauses.

    Args:
        query: Search query text.
    """
    try:
        from src.semantic_search import SemanticSearchEngine

        logger = setup_logging("search")
        logger.info("Running semantic search: '%s'", query)

        engine = SemanticSearchEngine()
        engine.load_index()
        response = engine.search(query)

        print(f"\n{'='*60}")
        print(f"Semantic Search Results for: \"{query}\"")
        print(f"{'='*60}\n")

        if not response.results:
            print("No matching clauses found.")
            return

        for i, result in enumerate(response.results, 1):
            print(f"--- Result {i} (score: {result.similarity_score:.4f}) ---")
            print(f"  Contract: {result.contract_id}")
            print(f"  Type:     {result.clause_type}")
            print(f"  Text:     {result.clause_text[:300]}...")
            print()

    except FileNotFoundError:
        print("ERROR: No search index found. Run the pipeline first.")
        sys.exit(1)
    except ImportError as exc:
        print(f"ERROR: Semantic search dependencies missing: {exc}")
        print("Install with: pip install sentence-transformers faiss-cpu")
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()
    logger = setup_logging("main")

    # Handle search-only mode
    if args.search_only:
        if not args.search:
            print("ERROR: --search-only requires --search <query>")
            sys.exit(1)
        run_semantic_search(args.search)
        return

    # Validate settings
    try:
        settings = get_settings()
    except Exception as exc:
        logger.error(
            "Configuration error: %s\n"
            "Ensure .env file exists with required variables. "
            "See .env.example for reference.",
            exc,
        )
        sys.exit(1)

    # Run pipeline
    try:
        pipeline = Pipeline(
            num_contracts=args.num_contracts,
            use_few_shot=not args.no_few_shot,
        )
        output = pipeline.run()

        # Run semantic search if requested
        if args.search and output.results:
            try:
                from src.semantic_search import SemanticSearchEngine

                logger.info("Building semantic search index...")
                engine = SemanticSearchEngine()
                engine.build_index(output.results)
                run_semantic_search(args.search)
            except ImportError:
                logger.warning(
                    "Semantic search dependencies not installed. Skipping."
                )

        # Exit with appropriate code
        if output.failed > 0 and output.successful == 0:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Pipeline failed with unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
