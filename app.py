import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    /* Main background */
    .stApp {
        background-color: #0f1117;
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3250;
    }

    /* User message bubble */
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

    /* AI message bubble */
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

    /* Avatar labels */
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

    /* Input box */
    .stTextInput input {
        background-color: #1e2235;
        color: white;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 12px;
    }

    /* Success/info colors */
    .stSuccess {
        background-color: #1a2e1a;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/book.png", width=60)
    st.title("Smart Study Buddy")
    st.markdown("---")
    st.markdown("### Upload Your Document")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Upload a PDF
    2. Ask any question
    3. AI answers from your document
    """)
    st.markdown("---")
    st.caption("Powered by Groq + LangChain + ChromaDB")


# Main Area
st.title("📚 Smart Study Buddy")
st.markdown("*Your AI-powered document assistant*")
st.markdown("---")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# Process PDF
if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner("Reading and indexing your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        st.session_state.pdf_processed = True
        st.session_state.chat_history = []

    st.success(f"PDF ready! {len(chunks)} chunks indexed. Ask me anything!")


# Display Chat History 
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.markdown(f'<div class="avatar-user">You</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="user-bubble">{message.content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="avatar-ai">Study Buddy</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ai-bubble">{message.content}</div>', unsafe_allow_html=True)


# Chat Input
if st.session_state.pdf_processed:
    question = st.chat_input("Ask a question about your document...")

    if question:
        # Add user message to history
        st.session_state.chat_history.append(HumanMessage(content=question))

        # Build context from history
        history_text = ""
        for msg in st.session_state.chat_history[:-1]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_text += f"{role}: {msg.content}\n"

        # Setup LLM
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )

        # Setup prompt with history
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful study assistant. Answer based on the document context below.
        If the answer is not in the document, say "I couldn't find that in the document."
        
        Previous conversation:
        {history}
        
        Document context:
        {context}
        
        Question: {input}
        """)

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
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
            answer = rag_chain.invoke(question)

        # Add AI response to history
        st.session_state.chat_history.append(AIMessage(content=answer))

        # Rerun to show updated chat
        st.rerun()

else:
    st.info("Upload a PDF from the sidebar to get started!")