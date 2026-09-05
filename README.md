# 📚 Intelligent RAG Document Assistant

A high-performance Retrieval-Augmented Generation (RAG) assistant built with **Streamlit**, **LangChain**, and **Groq Cloud**. This app allows users to upload PDF documents and chat with them in real-time, retrieving factual answers directly from the document context.

---

## 🚀 Features

- **Document Ingestion & Chunking:** Parses and chunks PDF documents into dense semantic segments.
- **Local Dense Embeddings:** Generates embeddings locally using `all-MiniLM-L6-v2` with ChromaDB vector store.
- **Ultra-Fast LLM Inference:** Powered by Groq's high-speed inference engine using `openai/gpt-oss-120b`.
- **Clean Chat Interface:** Interactive conversational UI with persistent session history via Streamlit.
- **Post-Processing Filtering:** Automated sanitization to strip internal reasoning tokens (`<think>` blocks) for direct answers.
- **Graceful Fallback:** Seamlessly answers general knowledge questions when queries go beyond the document scope.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Framework:** LangChain (LCEL)
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **LLM Provider:** Groq API (`openai/gpt-oss-120b`)
- **Parser:** PyPDF

---

## 📦 Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/Gehad-11/rag-document-assistant.git](https://github.com/Gehad-11/rag-document-assistant.git)
cd rag-document-assistant