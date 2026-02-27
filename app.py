import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile

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