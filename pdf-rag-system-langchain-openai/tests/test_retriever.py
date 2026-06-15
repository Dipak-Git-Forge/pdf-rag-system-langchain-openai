from src.retriever import retrieve_docs


class MockVectorStore:
    def similarity_search(self, query, k=5):
        return [f"Mock result for query: {query}" for _ in range(k)]


def test_retrieve_docs_returns_requested_number_of_results():
    mock_store = MockVectorStore()
    results = retrieve_docs(mock_store, "bread recipe", k=3)
    assert len(results) == 3


def test_retrieve_docs_uses_query_text():
    mock_store = MockVectorStore()
    results = retrieve_docs(mock_store, "wheat flour", k=2)
    assert all("wheat flour" in item for item in results)
