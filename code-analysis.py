import streamlit as st
from pathlib import Path
import os
import shutil
from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.chroma import ChromaDb
from phi.knowledge.text import TextKnowledgeBase
import chromadb
from chromadb.config import Settings
import stat

# SECURITY: Disable external connections
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['NO_PROXY'] = '*'  

# Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gpt-oss:20b"
VECTOR_DB_PATH = "./chroma_db"

# Initialize embedder
embedder = OllamaEmbedder(
    model="nomic-embed-text",
    host=OLLAMA_HOST,
)

# Initialize vector DB
chroma_client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH,
    settings=Settings(anonymized_telemetry=False, allow_reset=True)
)

vector_db = ChromaDb(
    collection="code_review_docs",
    path=VECTOR_DB_PATH,
    embedder=embedder,
    # client=chroma_client
)

# Knowledge Base for code only
code_knowledge = TextKnowledgeBase(
    path="uploaded_code",
    vector_db=vector_db,
)

def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Code Analysis Agent (No tools)
code_analyzer = Agent(
    name="Code Analyzer",
    model=Ollama(id=MODEL_NAME, allow_tools=False, host=OLLAMA_HOST),
    description="Expert in C language syntax, embedded systems patterns, and code structure analysis",
    instructions=[
        "You are a senior embedded C developer specializing in code analysis",
        "Analyze C code for syntax correctness, coding standards, and structural issues",
        "Focus on memory management, pointer usage, and embedded-specific concerns",
        "Identify potential bugs, memory leaks, buffer overflows, and unsafe operations",
        "Check for proper use of volatile, static, const keywords in embedded context",
        "Evaluate function complexity and suggest refactoring opportunities",
        "Provide specific line numbers and code snippets in your analysis"
    ],
    knowledge_base=code_knowledge,
    search_knowledge=False,
    markdown=True,
    show_tool_calls=False
)

# Streamlit UI
st.set_page_config(
    page_title="Embedded C Code Analysis",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Embedded C Code Analysis")
st.markdown("Focus: Syntax, memory management, and embedded best practices")

# Session state for chat history
if 'review_history' not in st.session_state:
    st.session_state.review_history = []

# Sidebar for code upload
with st.sidebar:
    st.header("📁 Upload Code Files")
    
    code_files = st.file_uploader(
        "Upload C source/header files",
        type=["c", "h"],
        accept_multiple_files=True
    )
    




    if st.button("🔄 Clear Knowledge Base"):
        if os.path.exists("uploaded_code"):
            shutil.rmtree("uploaded_code", onexc=handle_remove_readonly)
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH, onexc=handle_remove_readonly)
        st.success("Knowledge base cleared!")
        st.experimental_rerun()

# Create upload directory
os.makedirs("uploaded_code", exist_ok=True)

# Save uploaded code
if code_files:
    for code_file in code_files:
        file_path = os.path.join("uploaded_code", code_file.name)
        with open(file_path, "wb") as f:
            f.write(code_file.getbuffer())
    st.sidebar.success(f"✅ Uploaded {len(code_files)} code file(s)")
    # Load knowledge base
    try:
        code_knowledge.load(recreate=False, upsert=True)
        st.sidebar.success("✅ Knowledge base updated!")
    except Exception as e:
        st.sidebar.error(f"Error loading knowledge base: {str(e)}")

# Main analysis arease
analysis_query = st.text_area(
    "Enter your code analysis request:",
    placeholder="e.g., 'Check for memory leaks and buffer overflows in the driver code'",
    height=150
)

if st.button("Analyze Code"):
    code_count = len(list(Path("uploaded_code").glob("*.[ch]"))) if os.path.exists("uploaded_code") else 0
    if code_count == 0:
        st.warning("⚠️ Please upload code files first!")
    elif not analysis_query:
        st.warning("Please enter an analysis request")
    else:
        with st.spinner("Analyzing code..."):
            try:
                response = code_analyzer.run(analysis_query)
                st.markdown(response.content)
                # Save to history
                st.session_state.review_history.append({
                    "query": analysis_query,
                    "response": response.content
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Review History
if st.session_state.review_history:
    with st.expander("📜 Analysis History", expanded=False):
        for i, review in enumerate(reversed(st.session_state.review_history)):
            st.markdown(f"**Analysis #{len(st.session_state.review_history) - i}**")
            st.markdown(f"*Query:* {review['query']}")
            st.markdown(review['response'])
            st.markdown("---")
