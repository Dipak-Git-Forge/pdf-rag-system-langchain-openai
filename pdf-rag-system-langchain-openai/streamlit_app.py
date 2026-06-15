import streamlit as st

from src.rag_pipeline import generate_response

st.set_page_config(page_title="PDF RAG System", page_icon="📄", layout="centered")

st.title("📄 PDF RAG System")
st.write("Ask questions about a PDF using LangChain, FAISS, embeddings, and OpenAI.")

pdf_path = st.text_input("PDF path", value="data/Famous old receipts - bread.pdf")
query = st.text_area("Your question", value="a bread that uses wheat flour and is suitable for a dinner party")
k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)

if st.button("Generate Answer"):
    try:
        with st.spinner("Running RAG pipeline..."):
            answer = generate_response(query=query, pdf_path=pdf_path, k=k)
        st.subheader("Answer")
        st.write(answer)
    except Exception as exc:
        st.error(f"Error: {exc}")
