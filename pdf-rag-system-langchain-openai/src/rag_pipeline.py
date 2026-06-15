from typing import List

from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K
from .embed_store import create_vector_store
from .load_pdf import load_pdf
from .retriever import retrieve_docs


def format_context(context_docs: List[Document]) -> str:
    return "\n\n".join(
        [
            f"[Page {doc.metadata.get('page', 'N/A')}] {doc.page_content}"
            for doc in context_docs
        ]
    )


def generate_response(query: str, pdf_path: str, k: int = TOP_K) -> str:
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Add it to your .env file or environment variables."
        )

    documents = load_pdf(pdf_path)
    vector_store, _ = create_vector_store(documents)
    context_docs = retrieve_docs(vector_store, query, k=k)
    context_text = format_context(context_docs)

    system_message = f"""
You are a helpful document assistant.
Answer the user's question only from the provided context.
If the answer is not available in the context, clearly say that the information is not present.
Keep the answer practical, concise, and grounded.

Context:
{context_text}
"""

    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    response = llm.invoke([("system", system_message), ("human", query)])
    return response.content
