import argparse

from .rag_pipeline import generate_response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PDF RAG pipeline.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    answer = generate_response(query=args.query, pdf_path=args.pdf, k=args.k)
    print("\n=== RAG Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
