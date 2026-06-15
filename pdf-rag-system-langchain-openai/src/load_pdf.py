from pathlib import Path
from typing import List

import PyPDF2
from langchain.docstore.document import Document


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF and return one LangChain Document per page."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    documents = []
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "page": page_number},
                    )
                )
    return documents
