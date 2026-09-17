import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

PDF_PATH = "Ashish_Resume (2).pdf"
PERSIST_DIR = "./chroma_db"

PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not found in the context, say "I don't have enough information in the document to answer that."
Do not use any outside knowledge.

Context:
{context}

Question:
{question}

Answer:
""")


def load_and_split_pdf(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def get_vectorstore(chunks, embeddings_model):
    if os.path.exists(PERSIST_DIR):
        print("Loading existing vector store...")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings_model)
    print("Creating new vector store...")
    return Chroma.from_documents(documents=chunks, embedding=embeddings_model, persist_directory=PERSIST_DIR)


def ask_question(query, retriever, llm):
    retrieved_docs = retriever.invoke(query)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = PROMPT_TEMPLATE.format(context=context_text, question=query)
    response = llm.invoke(final_prompt)
    return response.content


def main():
    chunks = load_and_split_pdf(PDF_PATH)

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key = os.getenv("GOOGLE_GEMINI_API"))
    vectorstore = get_vectorstore(chunks, embeddings_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_GEMINI_API"))

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = ask_question(query, retriever, llm)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()