import streamlit as st
from dotenv import load_dotenv
import os

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