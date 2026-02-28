import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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

    # Setup Groq LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.2
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the provided context below.
    If you don't know the answer, say "I couldn't find that in the document."
    
    Context: {context}
    
    Question: {input}
    """)

    # Create retriever from vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Question input
    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(question)
            st.markdown("### Answer:")
            st.write(answer)