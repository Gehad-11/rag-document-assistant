import os
import re
import ssl
import tempfile
import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from groq import Groq

load_dotenv()

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
ssl._create_default_https_context = ssl._create_unverified_context

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RAG Knowledge Base Assistant", layout="wide")
st.title("Intelligent RAG Document Assistant")
st.write("Upload PDF document(s) and chat with them in real-time.")

def get_verified_groq_model(api_key):
    try:
        client = Groq(api_key=api_key)
        models_list = client.models.list()
        
        banned_terms = ["orpheus", "vision", "guard", "whisper", "audio", "embed", "classifier"]
        
        for m in models_list.data:
            m_id = m.id.lower()
            if not any(term in m_id for term in banned_terms):
                if any(valid in m_id for valid in ["llama", "mixtral", "gemma", "deepseek"]):
                    return m.id
        
        for m in models_list.data:
            m_id = m.id.lower()
            if not any(term in m_id for term in banned_terms):
                return m.id
                
        return models_list.data[0].id
    except Exception:
        return "llama-3.1-8b-instant"

def clean_response(text: str) -> str:
    cleaned = re.sub(r"(?is)<think>.*?(?:</think>|$)", "", text)
    cleaned = re.sub(r"(?i)</?think>", "", cleaned)
    return cleaned.strip()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.header("Document Setup")
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.session_state.rag_chain is None:
        with st.spinner("Processing all documents embeddings..."):
            all_docs = []
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    # استخدام PDFPlumberLoader لقراءة العربي بشكل أصح وأدق
                    loader = PDFPlumberLoader(tmp_path)
                    all_docs.extend(loader.load())
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            try:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = text_splitter.split_documents(all_docs)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                vectorstore = Chroma.from_documents(chunks, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                custom_http_client = httpx.Client(verify=False)
                chosen_model = get_verified_groq_model(GROQ_API_KEY)

                llm = ChatGroq(
                    model_name=chosen_model,
                    temperature=0.0,
                    api_key=GROQ_API_KEY,
                    http_client=custom_http_client
                )

                # برومبت صارم يمنع التكرار والرغي
                system_prompt = (
                    "You are a concise, factual assistant.\n"
                    "Answer the question strictly based ONLY on the context below.\n"
                    "Do NOT repeat yourself. Be direct and brief.\n"
                    "Always answer in the same language as the user's question.\n"
                    "If the answer is NOT in the context, reply with ONLY ONE single sentence: "
                    "\"عذراً، هذه المعلومة غير موجودة في المستند المرفق.\"\n\n"
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
                st.success(f"Loaded {len(uploaded_files)} document(s)! (Model: {chosen_model})")
            except Exception as e:
                st.error(f"Error processing documents: {e}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_query := st.chat_input("Ask a question about your documents..."):
    if st.session_state.rag_chain is None:
        st.warning("Please upload PDF document(s) first from the sidebar.")
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