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

# Set up a wide layout with a dark/light responsive title
st.set_page_config(page_title="StudyBot", layout="wide", initial_sidebar_state="expanded")

# Inject minimal, clean custom CSS for structural refinement
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #0284c7; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748b; margin-bottom: 2rem; }
    div.stButton > button:first-child { width: 100%; border-radius: 8px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: Document Management Control ---
with st.sidebar:
    st.markdown("<h2 style='margin-top:0;'>Control Center</h2>", unsafe_allow_html=True)
    st.write("Upload and process your document here.")
    
    uploaded_file = st.file_uploader("Upload a PDF book", type="pdf", label_visibility="collapsed")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
        
        st.toast("PDF uploaded successfully!", icon="📄")
        
        if st.button("⚡ Create VectorDB", type="primary"):
            with st.spinner("Processing text & building vector store..."):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db"
                )
            st.success("Knowledge base created successfully!")

# --- Main Workspace: Q&A Engine ---
st.markdown("<div class='main-title'>📚 BookLens</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Ask deep questions and extract immediate intelligence from your literature.</div>", unsafe_allow_html=True)

if os.path.exists("chroma_db"):
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
            "You are a helpful AI assistant.\n\nUse ONLY the provided context to answer the question.\n\nIf the answer is not present in the context,\nsay: 'I could not find the answer in the document.'"
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ])
    
    query = st.text_input("💬 Query the document:", placeholder="e.g., What are the core themes discussed in chapter 3?")
    
    if query:
        with st.spinner("Scanning vector indices..."):
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            final_prompt = prompt.invoke({"context": context, "question": query})
            response = llm.invoke(final_prompt)
        
        # Display response inside a visually clean bordered panel
        with st.container(border=True):
            st.markdown("### ✨ Response")
            st.markdown(response.content)
else:
    st.info("💡 Get started by uploading a PDF and clicking **Index Document** inside the sidebar panel.")