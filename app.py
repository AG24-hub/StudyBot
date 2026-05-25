import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
 
load_dotenv()
 
st.set_page_config(page_title="StudyBot", layout="wide", initial_sidebar_state="expanded")
 
st.markdown("""
    <style>
    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
 
    /* Chat container */
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding-bottom: 1rem;
    }
 
    /* Individual message bubbles */
    .msg-user {
        background: #0284c7;
        color: #fff;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 75%;
        align-self: flex-end;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .msg-ai {
        background: #f1f5f9;
        color: #1e293b;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 75%;
        align-self: flex-start;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .msg-label {
        font-size: 0.72rem;
        color: #94a3b8;
        margin-bottom: 2px;
    }
    .msg-label-right {
        text-align: right;
        font-size: 0.72rem;
        color: #94a3b8;
        margin-bottom: 2px;
    }
 
    /* Main title */
    .main-title { font-size: 2rem; font-weight: 700; color: #0284c7; margin-bottom: 0.1rem; }
    .sub-title { font-size: 1rem; color: #64748b; margin-bottom: 1.5rem; }
 
    div.stButton > button:first-child { width: 100%; border-radius: 8px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)
 
# --- Session State Init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"|"ai", "content": str}
 
# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='margin-top:0;'>📂 Control Center</h2>", unsafe_allow_html=True)
    st.write("Upload and index your PDF to start chatting.")
 
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")
 
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
 
        st.toast("PDF uploaded!", icon="📄")
 
        if st.button("⚡ Build Knowledge Base", type="primary"):
            with st.spinner("Processing & indexing..."):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
 
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
 
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db"
                )
            st.success("Knowledge base ready!")
            st.session_state.chat_history = []  # reset chat on new doc
 
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
 
# --- Main Area ---
st.markdown("<div class='main-title'>📚 StudyBot</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Chat with your document — ask anything, anytime.</div>", unsafe_allow_html=True)
 
if not os.path.exists("chroma_db"):
    st.info("💡 Upload a PDF and click **Build Knowledge Base** in the sidebar to get started.")
else:
    # Load retriever + LLM (cached implicitly via session)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    llm = ChatMistralAI(model="mistral-small-2506")
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant.\n\n"
            "Use ONLY the provided context to answer the question.\n\n"
            "If the answer is not present in the context, "
            "say: 'I could not find the answer in the document.'"
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])
 
    # --- Chat history display ---
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                "<p style='color:#94a3b8; font-style:italic;'>No messages yet. Ask something below!</p>",
                unsafe_allow_html=True
            )
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown("<div class='msg-label-right'>You</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='msg-user'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='msg-label'>StudyBot</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='msg-ai'>{msg['content']}</div>", unsafe_allow_html=True)
 
    st.divider()
 
    # --- Input row ---
    col1, col2 = st.columns([9, 1])
    with col1:
        user_input = st.chat_input("Ask something about your document…")
 
    if user_input:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
 
        # Generate response
        with st.spinner("Thinking…"):
            docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])
            final_prompt = prompt.invoke({"context": context, "question": user_input})
            response = llm.invoke(final_prompt)
 
        # Append AI message
        st.session_state.chat_history.append({"role": "ai", "content": response.content})
        st.rerun()
 