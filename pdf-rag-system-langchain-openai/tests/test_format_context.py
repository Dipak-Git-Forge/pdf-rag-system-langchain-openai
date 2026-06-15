from langchain.docstore.document import Document

from src.rag_pipeline import format_context


def test_format_context_includes_page_numbers_and_text():
    docs = [
        Document(page_content="Recipe one", metadata={"page": 1}),
        Document(page_content="Recipe two", metadata={"page": 2}),
    ]
    output = format_context(docs)
    assert "[Page 1] Recipe one" in output
    assert "[Page 2] Recipe two" in output
