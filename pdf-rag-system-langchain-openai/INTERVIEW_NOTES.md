# Viva / Interview Notes

## 1) What is RAG?
RAG stands for Retrieval-Augmented Generation. It improves LLM answers by first retrieving relevant context from an external knowledge source and then using that context to generate the answer.

## 2) Why did you use RAG in this project?
LLMs can hallucinate or answer from general knowledge. I used RAG so the model answers from the PDF content instead of relying only on pretrained knowledge.

## 3) How does your system work?
- I load the PDF with PyPDF2.
- I convert extracted text into LangChain Documents.
- I split documents into chunks.
- I create embeddings for each chunk using a Hugging Face model.
- I store the embeddings in FAISS.
- For a user query, I run similarity search to retrieve the most relevant chunks.
- I send the retrieved context plus the user query to ChatOpenAI.
- The model generates a final answer grounded in retrieved text.

## 4) Why did you use chunking?
Full pages may be too large and may contain mixed information. Chunking improves retrieval precision because the retriever can return smaller, more focused pieces of text.

## 5) Why FAISS?
FAISS is fast and widely used for similarity search over dense vector embeddings. It makes semantic retrieval efficient even with many text chunks.

## 6) Why all-MiniLM-L6-v2?
It is a lightweight and effective embedding model for semantic similarity tasks. It is a good choice for fast local experimentation.

## 7) What are the limitations of your project?
- PDF extraction quality depends on the document format.
- The system currently processes one PDF at a time.
- I am not persisting the vector store yet.
- Responses do not yet show source citations in the final output.

## 8) What improvements would you add next?
- multi-PDF support
- persistent FAISS storage
- source references in final answers
- better prompt templates
- reranking or hybrid retrieval
- deployment through Streamlit Cloud or Docker

## 9) Difference between fine-tuning and RAG?
Fine-tuning changes the model’s weights. RAG does not retrain the model; it augments the prompt with retrieved external context at runtime.

## 10) One strong final answer in interview
This project demonstrates how to build a practical LLM application that can answer questions from a document in a grounded way using embeddings, vector search, and prompt-based generation.
