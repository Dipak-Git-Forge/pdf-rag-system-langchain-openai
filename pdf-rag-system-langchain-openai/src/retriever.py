from typing import List, Protocol, Any


class SimilaritySearchable(Protocol):
    def similarity_search(self, query: str, k: int = 5) -> List[Any]:
        ...


def retrieve_docs(vector_store: SimilaritySearchable, query: str, k: int = 5):
    """Return the top-k most relevant chunks for a user query."""
    return vector_store.similarity_search(query, k=k)
