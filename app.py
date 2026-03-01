import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import tempfile

load_dotenv()

# Page Config
st.set_page_config(
    page_title="Smart Study Buddy",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #ffffff; }
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3250;
    }
    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 15px;
    }
    .ai-bubble {
        background-color: #1e2235;
        color: #e0e0e0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        border: 1px solid #2e3250;
        font-size: 15px;
    }
    .avatar-user {
        text-align: right;
        font-size: 12px;
        color: #888;
        margin-bottom: 2px;
    }
    .avatar-ai {
        text-align: left;
        font-size: 12px;
        color: #888;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/book.png", width=60)
    st.title("Smart Study Buddy")
    st.markdown("---")
    st.markdown("### Upload Your Document")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Reset everything if a new file is uploaded
    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        st.session_state.pdf_processed = False
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.pdf_name = uploaded_file.name

    if st.session_state.pdf_processed:
        st.success(f"Loaded: {st.session_state.pdf_name}")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Upload a PDF\n2. Ask any question\n3. AI answers from your document")
    st.markdown("---")
    st.caption("Powered by Groq + LangChain + ChromaDB")

# Main Area
st.title("📚 Smart Study Buddy")
st.markdown("*Your AI-powered document assistant*")
st.markdown("---")

# Process PDF only ONCE
if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner("Reading and indexing your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Bigger chunks = more context captured per chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"pdf_{st.session_state.pdf_name}_{len(chunks)}"
        )
        st.session_state.pdf_processed = True

    st.success(f"PDF ready! {len(chunks)} chunks indexed. Ask me anything!")

# Display Chat History
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.markdown('<div class="avatar-user">You</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="user-bubble">{message.content}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="avatar-ai">Study Buddy</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ai-bubble">{message.content}</div>', unsafe_allow_html=True)

# Chat Input
if st.session_state.pdf_processed:
    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.chat_history.append(HumanMessage(content=question))

        # Build full conversation history as text for context
        history_text = ""
        for msg in st.session_state.chat_history[:-1]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_text += f"{role}: {msg.content}\n"

        # Setup LLM
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )

        # Improved prompt - strict, thorough, follow-up aware
        prompt = ChatPromptTemplate.from_template("""
You are a helpful study assistant. Your job is to answer questions based ONLY on the document content provided below.

Rules:
- Answer based strictly on the document context
- If the user asks a follow-up like "and?" or "tell me more", refer back to the conversation history and expand your previous answer
- If something is not in the document, say "I couldn't find that in the document"
- Be thorough — list ALL relevant items you find, do not stop early
- Never make up or assume information not present in the document

Previous conversation:
{history}

Document context:
{context}

Current question: {input}

Answer:
""")

        # Retrieve more chunks (5 instead of 3) for better coverage
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough(),
                "history": lambda _: history_text
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Thinking..."):
            # Enrich follow-up questions with history so retriever finds right chunks
            full_query = question
            if history_text:
                full_query = f"{history_text}\nFollow-up question: {question}"
            answer = rag_chain.invoke(full_query)

        st.session_state.chat_history.append(AIMessage(content=answer))
        st.rerun()

else:
    st.info("Upload a PDF from the sidebar to get started!")