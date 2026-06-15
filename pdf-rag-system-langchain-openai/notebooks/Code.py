--Step 1: Loading and Extracting Text Data from a PDF Document
!pip install sentence-transformers==3.3.1
!pip install langchain-huggingface==0.1.2
!pip install langchain-community==0.3.7
!pip install langchain-openai==0.2.9

# Import the necessary libraries for PDF processing and document handling
import PyPDF2
from langchain.docstore.document import Document
from IPython.display import display, Markdown

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI

# Open and read the PDF file
pdf_text = []
with open('Famous old receipts - bread.pdf', "rb") as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        pdf_text.append(page.extract_text())

# Confirm successful extraction by printing the total number of pages
print(f"Text extracted from {len(pdf_text)} pages.")

# Create document objects from the extracted text
documents = []
for i in range(len(pdf_text)):
    documents.append(Document(page_content=pdf_text[i]))

# Print the number of documents created
print(f"Number of documents created: {len(documents)}")
--------------------------------------------------------------------------------------------------------------------

--Step 2: Embedding and Storing PDF Data in FAISS
# Load Hugging Face's Sentence Transformers embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Confirm successful model loading
print("Embedding model loaded successfully.")

# Embed the document chunks and store them in a FAISS vector database
db_faiss = FAISS.from_documents(documents, embedding_model)

# Confirm that the embeddings have been successfully stored
print("Document chunks embedded and stored in FAISS vector database.")
--------------------------------------------------------------------------------------------------------------------
--Step 3: Creating a Function to Retrieve Relevant Context from Embedded Data

# Define a function to retrieve relevant documents based on a query
def retrieve_docs(query, k):
    # Perform similarity search on the FAISS database
    docs_faiss = db_faiss.similarity_search(query, k=k)
    
    # Return the most relevant document chunks
    return docs_faiss
    
# Test the function by retrieving context based on a sample query
context = retrieve_docs("Most unique bread", 5)

# Display the first retrieved chunk to verify correct retrieval
print(context[0])    
-----------------------------------------------------------------------------------------------------------------------

Step 4: Build RAG System using Langchain and OpenAI
# Initialize the Langchain ChatOpenAI client
llm = ChatOpenAI()

# Define the user query
query = "a bread that uses wheat flour and is suitable for a dinner party"

# Define the system prompt for the assistant
system_message = f"""
    You are an assistant chef. 
    Your role is to recommend the most suitable recipe based on the context provided and the specific requirements given by the user, 
    such as available ingredients, dietary preferences, skill level, and desired baking time. 
    Make sure your recommendations are clear, practical, and tailored to meet the user's needs.
    You answer the {query} with the {context}
"""
# Structure the messages for the assistant
messages = [("system", system_message), ("human", query)]

# Retrieve relevant context from the FAISS database
context = retrieve_docs(query, 10)
# Generate and display the response from the assistant
response = llm.invoke(messages)  # Call the API with the messages
display(Markdown(response.content))  # Display the response in markdown format