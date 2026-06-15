from typing import List, Tuple

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME


def create_vector_store(documents: List[Document]) -> Tuple[FAISS, List[Document]]:
    """Split documents into chunks, generate embeddings, and store them in FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunked_documents = splitter.split_documents(documents)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(chunked_documents, embedding_model)
    return vector_store, chunked_documents
