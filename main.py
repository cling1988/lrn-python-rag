"""
main.py – Command-line interface for the RAG pipeline.

Usage
-----
Ingest a document::

    python main.py ingest my_document.pdf --persist-dir chroma_db

Query a persisted vector store::

    python main.py query "What is the document about?" --persist-dir chroma_db

Ingest and query in one step (useful for quick experiments)::

    python main.py query "What is discussed?" --source my_document.pdf

Environment
-----------
Set GOOGLE_API_KEY in your shell or in a .env file before running.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG (Retrieval-Augmented Generation) CLI – powered by Google GenAI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- ingest subcommand ----
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load documents into the vector store",
    )
    ingest_parser.add_argument(
        "source",
        help="File path (PDF or text), directory path, or URL to ingest",
    )
    ingest_parser.add_argument(
        "--persist-dir",
        default="chroma_db",
        help="Directory to persist the vector store (default: chroma_db)",
    )

    # ---- query subcommand ----
    query_parser = subparsers.add_parser(
        "query",
        help="Query the vector store and generate an answer with Gemini",
    )
    query_parser.add_argument("question", help="Question to answer")
    query_parser.add_argument(
        "--source",
        help="Document source to ingest before querying (skips if already ingested)",
    )
    query_parser.add_argument(
        "--persist-dir",
        default=None,
        help="Directory of a persisted vector store to load",
    )
    query_parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="Gemini model name (default: gemini-2.5-flash-lite)",
    )
    query_parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of chunks to retrieve (default: 4)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Validate that a Google API key is available before doing any work
    if not os.getenv("GOOGLE_API_KEY"):
        print(
            "Error: GOOGLE_API_KEY environment variable is not set.\n"
            "Create a .env file with GOOGLE_API_KEY=<your-key> or export it.",
            file=sys.stderr,
        )
        sys.exit(1)

    from rag.pipeline import RAGPipeline

    if args.command == "ingest":
        pipeline = RAGPipeline(persist_directory=args.persist_dir)
        n = pipeline.ingest(args.source)
        print(f"Ingested {n} chunk(s) from '{args.source}' → '{args.persist_dir}'")

    elif args.command == "query":
        pipeline = RAGPipeline(model=args.model, k=args.k)
        print(args.source)
        if args.source:
            # Ingest on-the-fly then immediately query
            pipeline.ingest(args.source)
        elif args.persist_dir:
            # Load a previously built vector store from disk
            pipeline.load(args.persist_dir)
        else:
            print(
                "Error: provide --source to ingest documents or "
                "--persist-dir to load a pre-built vector store.",
                file=sys.stderr,
            )
            sys.exit(1)

        answer = pipeline.query(args.question)
        print(answer)


if __name__ == "__main__":
    main()
