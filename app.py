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
#MainMenu, footer, header { visibility: hidden; }

/* ── WhatsApp-style bubble colours ── */
.bubble-user {
    background: #dcf8c6;
    color: #111;
    padding: 10px 14px;
    border-radius: 10px 0px 10px 10px;
    max-width: 72%;
    margin-left: auto;
    font-size: 0.93rem;
    line-height: 1.55;
    box-shadow: 0 1px 2px rgba(0,0,0,0.12);
    word-wrap: break-word;
}
.bubble-ai {
    background: #ffffff;
    color: #111;
    padding: 10px 14px;
    border-radius: 0px 10px 10px 10px;
    max-width: 72%;
    margin-right: auto;
    font-size: 0.93rem;
    line-height: 1.55;
    box-shadow: 0 1px 2px rgba(0,0,0,0.12);
    word-wrap: break-word;
}
.bubble-wrap-user {
    display: flex;
    justify-content: flex-end;
    margin: 4px 0;
}
.bubble-wrap-ai {
    display: flex;
    justify-content: flex-start;
    margin: 4px 0;
}
.chat-bg {
    background: #e5ddd5;
    border-radius: 12px;
    padding: 16px 12px;
    min-height: 300px;
}

/* Header bar */
.chat-header {
    background: #075e54;
    color: #fff;
    padding: 10px 16px;
    border-radius: 12px 12px 0 0;
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 600;
    font-size: 1rem;
}
.avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: #128c7e;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 14px; color: #fff;
    flex-shrink: 0;
}

div.stButton > button:first-child { width: 100%; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='margin-top:0;'>📂 Control Center</h2>", unsafe_allow_html=True)
    st.write("Upload and index your PDF to start chatting.")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        st.toast("PDF uploaded!", icon="📄")

        if st.button("⚡ Build Knowledge Base", type="primary"):
            with st.spinner("Processing & indexing…"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                ).split_documents(docs)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db"
                )
            st.success("Knowledge base ready!")
            st.session_state.chat_history = []

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Main area ──────────────────────────────────────────────────
if not os.path.exists("chroma_db"):
    st.info("💡 Upload a PDF and click **Build Knowledge Base** in the sidebar to get started.")
    st.stop()

# Load retriever + LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)
llm = ChatMistralAI(model="mistral-small-2506")
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant.\n\n"
     "Use ONLY the provided context to answer the question.\n\n"
     "If the answer is not present in the context, "
     "say: 'I could not find the answer in the document.'"),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# ── Chat header ────────────────────────────────────────────────
st.markdown("""
<div class='chat-header'>
  <div class='avatar'>S</div>
  <div>
    <div>StudyBot</div>
    <div style='font-size:11px;opacity:0.75;font-weight:400;'>Ask anything about your document</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Chat messages (scrollable container) ───────────────────────
with st.container():
    st.markdown("<div class='chat-bg'>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown(
            "<p style='text-align:center;color:#888;font-size:0.85rem;padding-top:1rem;'>"
            "No messages yet. Say something below!</p>",
            unsafe_allow_html=True
        )

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='bubble-wrap-user'>"
                f"<div class='bubble-user'>{msg['content']}</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='bubble-wrap-ai'>"
                f"<div class='bubble-ai'>{msg['content']}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ── Sticky input (Streamlit pins st.chat_input to the bottom) ──
user_input = st.chat_input("Ask something about your document…")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Thinking…"):
        retrieved = retriever.invoke(user_input)
        context = "\n\n".join([d.page_content for d in retrieved])
        final_prompt = prompt_template.invoke({"context": context, "question": user_input})
        response = llm.invoke(final_prompt)

    st.session_state.chat_history.append({"role": "ai", "content": response.content})
    st.rerun()
