"""Command-line entry point for the RAG pipeline."""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG (Retrieval-Augmented Generation) CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- ingest ----
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

    # ---- query ----
    query_parser = subparsers.add_parser(
        "query",
        help="Query the vector store and generate an answer",
    )
    query_parser.add_argument("question", help="Question to answer")
    query_parser.add_argument(
        "--source",
        help="Document source to ingest before querying (optional if already ingested)",
    )
    query_parser.add_argument(
        "--persist-dir",
        default=None,
        help="Directory of a persisted vector store to load",
    )
    query_parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat model name (default: gpt-4o-mini)",
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

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "Create a .env file with OPENAI_API_KEY=<your-key> or export it.",
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

        if args.source:
            pipeline.ingest(args.source)
        elif args.persist_dir:
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
