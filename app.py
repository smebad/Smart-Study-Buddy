import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile
from langchain_groq import ChatGroq
from langchain_community.chains import RetrievalQA

# Load our API key from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart Study Buddy",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Smart Study Buddy")
st.subheader("Upload your notes or textbook and ask me anything!")


# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # Load and split the PDF into chunks
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    st.info(f"PDF split into {len(chunks)} chunks for processing")

    with st.spinner("Processing your PDF... please wait"):
        # Create embeddings using HuggingFace (free, runs locally)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Store chunks in ChromaDB vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

    st.success("Ready! Ask me anything about your PDF below.")

    # Setup Groq LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.2
    )

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    # Question input
    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": question})
            st.markdown("### Answer:")
            st.write(result["result"])