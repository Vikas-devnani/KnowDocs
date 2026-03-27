"""
scripts/run_cli.py
─────────────────────────────────────────────────────────────────
Command-line interface for the RAG system.
Useful for testing without Streamlit.

Usage examples:
  python scripts/run_cli.py --ingest data/sample_pdfs/manual.pdf
  python scripts/run_cli.py --query "What are the system requirements?"
  python scripts/run_cli.py --status
─────────────────────────────────────────────────────────────────
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import ingest_documents, query, get_pipeline_status
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="RAG Tech Docs — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ingest", nargs="+", metavar="PDF_PATH",
                        help="Ingest one or more PDF files")
    parser.add_argument("--query", "-q", type=str, metavar="QUESTION",
                        help="Ask a question against indexed documents")
    parser.add_argument("--top-k", type=int, default=4,
                        help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset the FAISS index before ingesting")
    parser.add_argument("--status", action="store_true",
                        help="Print current index status")
    parser.add_argument("--json", action="store_true",
                        help="Output answers as JSON")

    args = parser.parse_args()

    # ── Status ─────────────────────────────────────────────────
    if args.status:
        status = get_pipeline_status()
        print(json.dumps(status, indent=2))
        return

    # ── Ingest ─────────────────────────────────────────────────
    if args.ingest:
        result = ingest_documents(
            sources=args.ingest,
            reset_index=args.reset,
        )
        print(json.dumps(result, indent=2))

    # ── Query ──────────────────────────────────────────────────
    if args.query:
        result = query(args.query, top_k=args.top_k)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "═" * 60)
            print("ANSWER:")
            print("─" * 60)
            print(result["answer"])
            print("\nSOURCES:")
            print("─" * 60)
            for src in result["sources"]:
                print(f"  [{src['index']}] {src['source']} — Page {src['page']} (score: {src['score']})")
                print(f"      {src['content'][:150]}…")
            print("═" * 60)


if __name__ == "__main__":
    main()
