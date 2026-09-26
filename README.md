# 📚 Intelligent RAG Document Assistant

A high-performance Retrieval-Augmented Generation (RAG) assistant built with **Streamlit**, **LangChain**, and **Groq Cloud**. This app allows users to upload multiple PDF documents simultaneously and chat with them in real-time, retrieving factual answers directly from the document context.

---

## 🚀 Features

- **Multi-Document Ingestion & Chunking:** Parses and chunks multiple PDF documents at once into dense semantic segments.
- **Local Dense Embeddings:** Generates embeddings locally using `all-MiniLM-L6-v2` with ChromaDB vector store.
- **Ultra-Fast LLM Inference:** Powered by Groq's high-speed inference engine with dynamic model verification.
- **Clean Chat Interface:** Interactive conversational UI with persistent session history via Streamlit.
- **Post-Processing Filtering:** Automated sanitization to strip internal reasoning tokens (`<think>` blocks) for direct answers.
- **Graceful Fallback:** Seamlessly answers general knowledge questions when queries go beyond the document scope.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Framework:** LangChain (LCEL)
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **LLM Provider:** Groq API
- **Parser:** PyPDF Loader

---

## 📦 Getting Started & Setup

### Step 1: Clone the repository
```bash
git clone https://github.com/Gehad-11/rag-document-assistant.git
cd rag-document-assistant

### Step 2: Create and activate virtual environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
###Step 3: Create project dependencies (requirements.txt)
Create a file named requirements.txt in the root folder with:

Plaintext
streamlit
langchain
langchain-community
langchain-core
langchain-groq
langchain-text-splitters
chromadb
sentence-transformers
pypdf
httpx
python-dotenv
groq
###Step 4: Set up environment variables (.env)
Create a .env file in the root directory to store your Groq API key securely:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
Step 5: Configure git ignore (.gitignore)
Create a .gitignore file to prevent sensitive files and cache from being committed:

Plaintext
.env
venv/
__pycache__/
*.pyc
.chroma/
###Step 6: Install dependencies
Bash
pip install -r requirements.txt
Step 7: Run the application
Bash
python -m streamlit run app.py