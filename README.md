# DocuMind 📄🤖

A "Chat with your PDF" RAG (Retrieval-Augmented Generation) system that lets you ask questions about a local PDF document and get accurate, grounded answers — no hallucinations, no made-up facts.

## How it works

1. **Ingestion** — The PDF is loaded (`PyPDFLoader`) and split into small overlapping text chunks (`RecursiveCharacterTextSplitter`).
2. **Indexing** — Each chunk is converted into a vector embedding (`GoogleGenerativeAIEmbeddings`) and stored in a local vector database (`Chroma`).
3. **Retrieval** — When you ask a question, it's embedded too, and the most semantically similar chunks are retrieved from the vector store.
4. **Generation** — The retrieved chunks + your question are passed to Gemini (`ChatGoogleGenerativeAI`) via a strict prompt template that instructs it to answer **only** from the provided context — if the answer isn't in the document, it says so instead of guessing.

```
PDF → Split into chunks → Embed → Store in Chroma
                                        ↓
Question → Embed → Retrieve similar chunks → LLM (Gemini) → Answer
```

## Tech Stack

- `langchain` / `langchain-community` / `langchain-text-splitters`
- `langchain-google-genai` (Gemini embeddings + chat model)
- `chromadb` / `langchain-chroma` (vector database)
- `pypdf` (PDF parsing)
- `python-dotenv` (environment variable management)

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/DocuMind.git
   cd DocuMind
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Gemini API key (get one at [Google AI Studio](https://aistudio.google.com/apikey)):
   ```
   GOOGLE_API_KEY=your_key_here
   ```

4. Place your PDF in the project folder and update the `PDF_PATH` variable in `main.py` to match its filename.

5. Run it:
   ```bash
   python main.py
   ```

6. Ask questions in the terminal. Type `exit` to quit.

## Example

```
Ask a question (or type 'exit'): What hackathons did he participate in?
Answer: He participated in Smart India Hackathon, VIBE-A-THON 2026, 
AI Verse Hackathon, and NMIT Hacks 2026.

Ask a question (or type 'exit'): Rate the resume out of 10
Answer: I don't have enough information in the document to answer that.
```

Notice the second example — the system correctly refuses to answer questions that require opinion/judgment rather than facts from the document, avoiding hallucination.

## Web UI (optional)

A simple Streamlit interface is also available:

```bash
streamlit run app.py
```

This opens a browser tab where you can ask questions through a text box instead of the terminal.

## Key Design Choices

- **Persistent vector store**: Embeddings are cached to disk (`chroma_db/`) so the document isn't re-embedded on every run, saving API calls.
- **Grounded prompting**: The prompt explicitly instructs the model to answer only from retrieved context, preventing fabricated answers.
- **`temperature=0`**: Ensures deterministic, factual responses rather than creative ones.

## Future Improvements

- Source citation (show which page an answer came from)
- Support for multiple documents
