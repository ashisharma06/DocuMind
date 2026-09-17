import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from main import load_and_split_pdf, get_vectorstore, ask_question, PDF_PATH
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

st.set_page_config(page_title="DocuMind", page_icon="📄")
st.title("📄 DocuMind — Chat with your PDF")

@st.cache_resource
def setup():
    chunks = load_and_split_pdf(PDF_PATH)
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_GEMINI_API")
    )
    vectorstore = get_vectorstore(chunks, embeddings_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_GEMINI_API")
    )
    return retriever, llm

retriever, llm = setup()

query = st.text_input("Ask a question about the document:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        answer = ask_question(query, retriever, llm)
    st.write("### Answer")
    st.write(answer)