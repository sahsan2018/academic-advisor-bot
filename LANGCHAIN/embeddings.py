from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from config import GOOGLE_API_KEY
import streamlit as st
import os

# Initialize embeddings and chat model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)