import os
import re
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

st.set_page_config(page_title="RAG Knowledge Base Assistant", layout="wide")
st.title("Intelligent RAG Document Assistant")
st.write("Upload a PDF document and chat with it in real-time.")

def clean_response(text: str) -> str:
    cleaned = re.sub(r"(?is)<think>.*?(?:</think>|$)", "", text)
    cleaned = re.sub(r"(?i)</?think>", "", cleaned)
    return cleaned.strip()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])

    if uploaded_file is not None and st.session_state.rag_chain is None:
        with st.spinner("Processing document embeddings..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

            llm = ChatGroq(
                model_name="openai/gpt-oss-120b",
                temperature=0.1,
                max_tokens=300
            )

            system_prompt = (
                "You are a helpful and intelligent assistant. "
                "Prioritize using the provided context to answer questions about the document. "
                "If the question is unrelated to the context, answer it using your general knowledge directly and concisely.\n\n"
                "Context:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])

            st.session_state.rag_chain = (
                {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            st.success("Document ready for chat!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_query := st.chat_input("Ask a question about your document..."):
    if st.session_state.rag_chain is None:
        st.warning("Please upload a PDF document first from the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    raw_answer = st.session_state.rag_chain.invoke(user_query)
                    final_answer = clean_response(raw_answer)
                    st.write(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                except Exception as e:
                    st.error(f"Error: {e}")