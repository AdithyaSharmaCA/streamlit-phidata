import streamlit as st
from pathlib import Path
import tempfile
import shutil
from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.chroma import ChromaDb
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.knowledge.text import TextKnowledgeBase
from phi.knowledge.combined import CombinedKnowledgeBase
import os
import json
import chromadb
from chromadb.config import Settings
import stat
import gc

# SECURITY: Disable any external connections
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['NO_PROXY'] = '*'  # Disable proxy for all connections

# Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "phi3:mini"
VECTOR_DB_PATH = "./chroma_db"

# Initialize embedder - LOCAL ONLY
embedder = OllamaEmbedder(
    model="nomic-embed-text",
    host=OLLAMA_HOST,
)

# Initialize vector database with ChromaDB - LOCAL ONLY
vector_db = ChromaDb(
    collection="code_review_docs",
    path=VECTOR_DB_PATH,
    embedder=embedder,
    persistent_client=True
)

# Initialize Knowledge Bases
code_knowledge = TextKnowledgeBase(
    path="uploaded_code",
    vector_db=vector_db,
)

docs_knowledge = PDFKnowledgeBase(
    path="uploaded_docs",
    vector_db=vector_db,
)

# Combine knowledge bases
combined_knowledge = CombinedKnowledgeBase(
    sources=[code_knowledge, docs_knowledge]
)

# Agent 1: Code Analyzer
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
    knowledge_base=combined_knowledge,
    search_knowledge=False,
    markdown=True,
    show_tool_calls=False
)

# Agent 2: Architecture Reviewer
architecture_reviewer = Agent(
    name="Architecture Reviewer",
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    description="Expert in embedded systems architecture and design patterns",
    instructions=[
        "You are a principal embedded systems architect",
        "Review code from an architectural perspective",
        "Evaluate modularity, coupling, cohesion, and separation of concerns",
        "Assess hardware abstraction layers and driver implementations",
        "Check for proper interrupt handling and real-time considerations",
        "Evaluate state machines, concurrency patterns, and synchronization",
        "Suggest architectural improvements for scalability and maintainability",
        "Consider power consumption, performance, and resource constraints"
    ],
    knowledge_base=combined_knowledge,
    search_knowledge=False,
    markdown=True,
    show_tool_calls=False
)

# Agent 3: Safety & Standards Validator
safety_validator = Agent(
    name="Safety & Standards Validator",
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    description="Expert in embedded safety standards and best practices",
    instructions=[
        "You are a safety-critical systems expert",
        "Validate code against MISRA-C standards and embedded best practices",
        "Check for safety-critical issues: race conditions, deadlocks, priority inversion",
        "Evaluate error handling, boundary checks, and defensive programming",
        "Assess compliance with coding standards and documentation requirements",
        "Review watchdog implementation, fail-safe mechanisms, and error recovery",
        "Check for proper use of assert, error codes, and logging",
        "Flag any undefined behavior or implementation-defined behavior concerns"
    ],
    knowledge_base=combined_knowledge,
    search_knowledge=False,
    markdown=True,
    show_tool_calls=False
)

# Team of agents
review_team = Agent(
    name="Code Review Team Lead",
    team=[code_analyzer, architecture_reviewer, safety_validator],
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    instructions=[
        "You are a lead reviewer coordinating a team of specialists",
        "Delegate tasks to the appropriate specialist agents",
        "Synthesize their findings into a comprehensive review",
        "Prioritize issues by severity and impact",
        "Provide actionable recommendations"
    ],
    knowledge_base=combined_knowledge,
    search_knowledge=False,
    markdown=True,
    show_tool_calls=False
)

# Streamlit UI Setup
st.set_page_config(
    page_title="Embedded C Code Review System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Embedded C Code Review System")
st.markdown("### Multi-Agent Architecture Review for Embedded Development")

# Initialize session state
if 'review_history' not in st.session_state:
    st.session_state.review_history = []
if 'kb_cleared' not in st.session_state:
    st.session_state.kb_cleared = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Define safe delete handler
def remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass  # ignore stubborn files safely

# Sidebar for uploads
with st.sidebar:
    st.header("📁 Upload Files")

    st.subheader("Code Files (.c, .h)")
    code_files = st.file_uploader(
        "Upload C source/header files",
        type=["c", "h"],
        accept_multiple_files=True,
        key=f"code_files_{st.session_state.uploader_key}"
    )

    st.subheader("Documentation (.pdf)")
    doc_files = st.file_uploader(
        "Upload reference docs, datasheets, specs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"doc_files_{st.session_state.uploader_key}"
    )

    if st.button("🔄 Clear Knowledge Base", type="secondary"):
        try:
            # First, delete the ChromaDB collection
            try:
                vector_db.client.delete_collection("code_review_docs")
                st.info("🗑️ ChromaDB collection deleted")
            except Exception as e:
                st.warning(f"⚠️ Could not delete collection: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Then delete the uploaded files and vector DB directory
            for folder in ["uploaded_code", "uploaded_docs", VECTOR_DB_PATH]:
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder, onerror=remove_readonly)
                        st.info(f"🗑️ Deleted {folder}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not fully delete {folder}: {e}")
            
            # Increment uploader key to reset file uploaders
            st.session_state.uploader_key += 1
            st.session_state.kb_cleared = True
            
            st.success("✅ Knowledge base cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error clearing knowledge base: {str(e)}")
            st.exception(e)
    
    st.markdown("---")
    st.subheader("⚙️ Configuration")
    st.text(f"Server: {OLLAMA_HOST}")
    st.text(f"Model: {MODEL_NAME}")

# Create directories
os.makedirs("uploaded_code", exist_ok=True)
os.makedirs("uploaded_docs", exist_ok=True)

# Reset kb_cleared flag after rerun
if st.session_state.kb_cleared:
    st.session_state.kb_cleared = False

# Handle file uploads - only process if files are actually uploaded
files_updated = False

if code_files and len(code_files) > 0:
    for code_file in code_files:
        file_path = os.path.join("uploaded_code", code_file.name)
        # Only write if file doesn't exist or is different
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(code_file.getbuffer())
            files_updated = True
    if files_updated:
        st.sidebar.success(f"✅ Uploaded {len(code_files)} code file(s)")

if doc_files and len(doc_files) > 0:
    doc_updated = False
    for doc_file in doc_files:
        file_path = os.path.join("uploaded_docs", doc_file.name)
        # Only write if file doesn't exist or is different
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(doc_file.getbuffer())
            doc_updated = True
    if doc_updated:
        files_updated = True
        st.sidebar.success(f"✅ Uploaded {len(doc_files)} document(s)")

# Load into knowledge base only if files were actually updated
if files_updated:
    with st.sidebar:
        with st.spinner("Loading files into knowledge base..."):
            try:
                if code_files and len(code_files) > 0:
                    code_knowledge.load(recreate=False, upsert=True)
                if doc_files and len(doc_files) > 0:
                    docs_knowledge.load(recreate=False, upsert=True)
                st.success("✅ Knowledge base updated!")
            except Exception as e:
                st.error(f"Error loading knowledge base: {str(e)}")

# Tabs for review modes
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Comprehensive Review",
    "🔬 Code Analysis",
    "🏗️ Architecture Review",
    "⚠️ Safety Validation"
])

# Comprehensive Review Tab
with tab1:
    st.header("Comprehensive Multi-Agent Review")
    st.markdown("*All three agents collaborate to provide a complete code review*")

    review_query = st.text_area(
        "Enter your review request:",
        placeholder="e.g., 'Review the entire codebase for embedded best practices and safety issues'",
        height=100,
        key="comprehensive_query"
    )

    if st.button("🚀 Start Comprehensive Review", type="primary"):
        if not review_query:
            st.warning("Please enter a review request")
        else:
            code_count = len(list(Path("uploaded_code").glob("*.[ch]"))) if os.path.exists("uploaded_code") else 0
            if code_count == 0:
                st.warning("⚠️ Please upload code files first!")
            else:
                with st.spinner("Agents are reviewing your code..."):
                    try:
                        response = review_team.run(review_query)
                        st.success("✅ Review Complete!")
                        st.markdown(response.content)

                        st.session_state.review_history.append({
                            "type": "Comprehensive",
                            "query": review_query,
                            "response": response.content
                        })
                    except Exception as e:
                        st.error(f"❌ Error during review: {str(e)}")
                        st.exception(e)

# Code Analysis Tab
with tab2:
    st.header("Code Analysis")
    st.markdown("*Focus: Syntax, structure, memory management, and code quality*")

    analysis_query = st.text_area(
        "Specific analysis request:",
        placeholder="e.g., 'Check for memory leaks and buffer overflows in the driver code'",
        height=100,
        key="analysis"
    )

    if st.button("Analyze Code", type="primary", key="analyze_btn"):
        if not analysis_query:
            st.warning("Please enter an analysis request")
        else:
            code_count = len(list(Path("uploaded_code").glob("*.[ch]"))) if os.path.exists("uploaded_code") else 0
            if code_count == 0:
                st.warning("⚠️ Please upload code files first!")
            else:
                with st.spinner("Analyzing code..."):
                    try:
                        response = code_analyzer.run(analysis_query)
                        st.markdown(response.content)
                        st.session_state.review_history.append({
                            "type": "Code Analysis",
                            "query": analysis_query,
                            "response": response.content
                        })
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)

# Architecture Review Tab
with tab3:
    st.header("Architecture Review")
    st.markdown("*Focus: Design patterns, modularity, and system architecture*")

    arch_query = st.text_area(
        "Architecture review request:",
        placeholder="e.g., 'Evaluate the HAL design and suggest improvements for modularity'",
        height=100,
        key="architecture"
    )

    if st.button("Review Architecture", type="primary", key="arch_btn"):
        if not arch_query:
            st.warning("Please enter an architecture review request")
        else:
            code_count = len(list(Path("uploaded_code").glob("*.[ch]"))) if os.path.exists("uploaded_code") else 0
            if code_count == 0:
                st.warning("⚠️ Please upload code files first!")
            else:
                with st.spinner("Reviewing architecture..."):
                    try:
                        response = architecture_reviewer.run(arch_query)
                        st.markdown(response.content)
                        st.session_state.review_history.append({
                            "type": "Architecture",
                            "query": arch_query,
                            "response": response.content
                        })
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)

# Safety Validation Tab
with tab4:
    st.header("Safety & Standards Validation")
    st.markdown("*Focus: MISRA-C compliance, safety-critical issues, and best practices*")

    safety_query = st.text_area(
        "Safety validation request:",
        placeholder="e.g., 'Check for MISRA-C violations and safety-critical issues'",
        height=100,
        key="safety"
    )

    if st.button("Validate Safety", type="primary", key="safety_btn"):
        if not safety_query:
            st.warning("Please enter a safety validation request")
        else:
            code_count = len(list(Path("uploaded_code").glob("*.[ch]"))) if os.path.exists("uploaded_code") else 0
            if code_count == 0:
                st.warning("⚠️ Please upload code files first!")
            else:
                with st.spinner("Validating safety standards..."):
                    try:
                        response = safety_validator.run(safety_query)
                        st.markdown(response.content)
                        st.session_state.review_history.append({
                            "type": "Safety",
                            "query": safety_query,
                            "response": response.content
                        })
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)

# Review History
if st.session_state.review_history:
    with st.expander("📜 Review History", expanded=False):
        for i, review in enumerate(reversed(st.session_state.review_history)):
            st.markdown(f"**{review['type']} Review #{len(st.session_state.review_history) - i}**")
            st.markdown(f"*Query:* {review['query']}")
            with st.container():
                st.markdown(review['response'])
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("**System Status:**")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Ollama Server", "localhost:11434")
with col2:
    st.metric("Model", MODEL_NAME)
with col3:
    code_count = len(list(Path("uploaded_code").glob("*.[ch]"))) if os.path.exists("uploaded_code") else 0
    docs_count = len(list(Path("uploaded_docs").glob("*.pdf"))) if os.path.exists("uploaded_docs") else 0
    st.metric("Files Loaded", f"{code_count} code, {docs_count} docs")