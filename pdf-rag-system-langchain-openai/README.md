# PDF RAG System with LangChain, FAISS, and OpenAI

A complete **Retrieval-Augmented Generation (RAG)** project for PDF-based question answering using **LangChain**, **Hugging Face embeddings**, **FAISS**, and **OpenAI**.

This project loads a PDF, extracts text, splits the content into chunks, generates embeddings, stores them in a vector database, retrieves the most relevant chunks for a user query, and produces a grounded answer using an OpenAI chat model.

---

## 🚀 Key Features

- Extracts text from PDF files with **PyPDF2**
- Converts PDF pages into **LangChain Documents**
- Splits text into chunks for better retrieval quality
- Creates embeddings with **sentence-transformers/all-MiniLM-L6-v2**
- Stores vectors in a **FAISS** index
- Retrieves top-k relevant chunks using similarity search
- Generates answers with **OpenAI** through LangChain
- Includes a **Streamlit UI**
- Includes **unit tests** and sample project structure

---

## 🧰 Tech Stack

- Python
- LangChain
- LangChain OpenAI
- LangChain HuggingFace
- FAISS
- PyPDF2
- Sentence Transformers
- OpenAI API
- Streamlit
- Pytest

---

## 📁 Project Structure

```bash
pdf-rag-system-langchain-openai-v2/
│
├── data/
│   └── .gitkeep
├── notebooks/
│   └── rag_pdf_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── load_pdf.py
│   ├── embed_store.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── main.py
├── tests/
│   ├── test_retriever.py
│   └── test_format_context.py
├── assets/
│   └── .gitkeep
├── streamlit_app.py
├── PROJECT_DESCRIPTION.md
├── LINKEDIN_POST.md
├── INTERVIEW_NOTES.md
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 🧠 Project Workflow

1. Load the PDF from the `data/` folder.
2. Extract text page by page.
3. Convert text into LangChain `Document` objects.
4. Split documents into smaller chunks.
5. Generate vector embeddings using Hugging Face.
6. Store embeddings in a FAISS vector database.
7. Retrieve relevant context using similarity search.
8. Pass the retrieved context and query to OpenAI.
9. Generate an answer grounded in the PDF content.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/pdf-rag-system-langchain-openai-v2.git
cd pdf-rag-system-langchain-openai-v2
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

#### Windows
```bash
venv\Scriptsctivate
```

#### macOS / Linux
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Copy `.env.example` to `.env` and update the values.

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 5. Add your PDF file
Put your PDF inside `data/`.

Example:
```bash
data/Famous old receipts - bread.pdf
```

---

## ▶️ Usage

### Run from terminal
```bash
python -m src.main --pdf "data/Famous old receipts - bread.pdf" --query "a bread that uses wheat flour and is suitable for a dinner party"
```

### Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

### Run the notebook
Open the file below in Jupyter Notebook or VS Code:
```bash
notebooks/rag_pdf_demo.ipynb
```

---

## 🧪 Testing

Run all tests:

```bash
pytest tests/
```

---

## 🖼️ Screenshots

Add your actual screenshots to the `assets/` folder and uncomment the lines below.

```markdown
![Project Workflow](assets/flow.png)
![Notebook Output](assets/screenshot.png)
![Streamlit UI](assets/streamlit-ui.png)
```

---

## 💡 Sample Query

```python
query = "a bread that uses wheat flour and is suitable for a dinner party"
```

---

## 📌 Why This Project Matters

This project shows practical understanding of:

- RAG architecture
- document ingestion
- text chunking
- vector embeddings
- semantic search
- LLM prompting
- end-to-end application structure

It is a strong portfolio project for **Generative AI**, **LLM application development**, and **NLP engineering** roles.

---

## 🔮 Future Improvements

- Support multiple PDFs
- Save and reuse FAISS indexes
- Add citations/source references in responses
- Add metadata-aware retrieval
- Add hybrid search and reranking
- Add deployment with Docker or cloud hosting

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Dipak Thakre**  
Consultant  
Pune, Maharashtra

If you found this project useful, consider starring the repository.
