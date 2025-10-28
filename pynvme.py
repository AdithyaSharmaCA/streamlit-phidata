import streamlit as st
from pathlib import Path
import tempfile
import shutil
from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.chroma import ChromaDb
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.knowledge.text import TextKnowledgeBase
from phi.knowledge.combined import CombinedKnowledgeBase
from phi.tools.function import Function
import os
import json
import stat
import gc
from typing import Dict, List, Optional
from datetime import datetime

# SECURITY: Disable any external connections
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['NO_PROXY'] = '*'

# Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "phi3:mini"
VECTOR_DB_PATH = "./pynvme_chroma_db"

# Initialize embedder
embedder = OllamaEmbedder(
    model="nomic-embed-text",
    host=OLLAMA_HOST,
)

# Initialize vector database
vector_db = ChromaDb(
    collection="pynvme_docs",
    path=VECTOR_DB_PATH,
    embedder=embedder,
    persistent_client=True
)

# Initialize Knowledge Base for PyNVMe documentation
pynvme_knowledge = CombinedKnowledgeBase(
    sources=[
        PDFKnowledgeBase(
            path="pynvme_docs",
            vector_db=vector_db,
        ),
        TextKnowledgeBase(
            path="pynvme_docs",
            vector_db=vector_db,
        )
    ]
)

# ============================================================================
# TOOL FUNCTIONS - These will be used by agents to interact with each other
# ============================================================================

def validate_test_syntax(test_code: str) -> Dict[str, any]:
    """
    Validate the syntax and structure of a PyNVMe test case.
    
    Args:
        test_code: The Python test code to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": []
    }
    
    # Check for required imports
    required_imports = ["pytest", "pynvme"]
    for imp in required_imports:
        if imp not in test_code:
            validation_results["errors"].append(f"Missing import: {imp}")
            validation_results["is_valid"] = False
    
    # Check for test function
    if "def test_" not in test_code:
        validation_results["errors"].append("No test function found (should start with 'def test_')")
        validation_results["is_valid"] = False
    
    # Check for PyNVMe fixture usage
    if "nvme" not in test_code and "subsystem" not in test_code:
        validation_results["warnings"].append("No PyNVMe fixture (nvme/subsystem) detected")
    
    # Check for assertions
    if "assert" not in test_code:
        validation_results["warnings"].append("No assertions found in test")
    
    return validation_results


def check_pynvme_best_practices(test_code: str) -> Dict[str, List[str]]:
    """
    Check if the test follows PyNVMe best practices.
    
    Args:
        test_code: The test code to check
        
    Returns:
        Dictionary with best practice recommendations
    """
    recommendations = {
        "good_practices": [],
        "improvements_needed": []
    }
    
    # Check for proper cleanup
    if "nvme.close()" in test_code or "with " in test_code:
        recommendations["good_practices"].append("Proper resource cleanup detected")
    else:
        recommendations["improvements_needed"].append(
            "Add proper cleanup: use context managers or explicit nvme.close()"
        )
    
    # Check for error handling
    if "try:" in test_code or "pytest.raises" in test_code:
        recommendations["good_practices"].append("Error handling present")
    else:
        recommendations["improvements_needed"].append(
            "Consider adding error handling for robustness"
        )
    
    # Check for timeouts
    if "timeout" in test_code.lower():
        recommendations["good_practices"].append("Timeout handling present")
    else:
        recommendations["improvements_needed"].append(
            "Consider adding timeout parameters for long operations"
        )
    
    # Check for docstrings
    if '"""' in test_code or "'''" in test_code:
        recommendations["good_practices"].append("Test documentation present")
    else:
        recommendations["improvements_needed"].append(
            "Add docstring to explain test purpose and expected behavior"
        )
    
    return recommendations


def suggest_test_improvements(test_code: str, validation_results: Dict) -> str:
    """
    Generate specific improvement suggestions for a test case.
    
    Args:
        test_code: The test code
        validation_results: Results from validation
        
    Returns:
        Formatted string with improvement suggestions
    """
    suggestions = []
    
    if not validation_results["is_valid"]:
        suggestions.append("CRITICAL: Fix validation errors first")
        suggestions.extend([f"  - {err}" for err in validation_results["errors"]])
    
    if validation_results["warnings"]:
        suggestions.append("\nWARNINGS:")
        suggestions.extend([f"  - {warn}" for warn in validation_results["warnings"]])
    
    # Additional suggestions based on code analysis
    if "pytest.mark" not in test_code:
        suggestions.append("\nConsider adding pytest markers (@pytest.mark.parametrize, etc.)")
    
    if "logging" not in test_code:
        suggestions.append("\nAdd logging for better debugging")
    
    return "\n".join(suggestions) if suggestions else "Test looks good!"


def save_test_case(test_name: str, test_code: str, category: str = "general") -> Dict[str, str]:
    """
    Save a generated test case to disk.
    
    Args:
        test_name: Name of the test
        test_code: The test code
        category: Test category (performance, functional, stress, etc.)
        
    Returns:
        Dictionary with save status
    """
    try:
        # Create directory structure
        test_dir = Path(f"generated_tests/{category}")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.py"
        filepath = test_dir / filename
        
        # Save file
        with open(filepath, "w") as f:
            f.write(test_code)
        
        return {
            "status": "success",
            "filepath": str(filepath),
            "message": f"Test saved successfully to {filepath}"
        }
    except Exception as e:
        return {
            "status": "error",
            "filepath": "",
            "message": f"Error saving test: {str(e)}"
        }


def get_test_template(test_type: str) -> str:
    """
    Get a test template based on test type.
    
    Args:
        test_type: Type of test (read, write, trim, admin, etc.)
        
    Returns:
        Template code string
    """
    templates = {
        "read": '''
import pytest
import pynvme as nvme

def test_sequential_read(nvme0):
    """Test sequential read operations."""
    # Test implementation here
    pass
''',
        "write": '''
import pytest
import pynvme as nvme

def test_sequential_write(nvme0):
    """Test sequential write operations."""
    # Test implementation here
    pass
''',
        "admin": '''
import pytest
import pynvme as nvme

def test_admin_command(nvme0):
    """Test NVMe admin commands."""
    # Test implementation here
    pass
'''
    }
    return templates.get(test_type, templates["read"])


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

# Agent 1: PyNVMe Expert - Knowledge Base Agent
pynvme_expert = Agent(
    name="PyNVMe Expert",
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    description="Expert in PyNVMe library, NVMe protocol, and test automation",
    instructions=[
        "You are an expert in PyNVMe library and NVMe testing",
        "Provide accurate information about PyNVMe APIs, fixtures, and best practices",
        "Reference the documentation from the knowledge base",
        "Explain NVMe concepts clearly and provide code examples",
        "Help with understanding PyNVMe test patterns and common use cases",
        "Search the knowledge base before answering questions"
    ],
    knowledge_base=pynvme_knowledge,
    search_knowledge=True,
    markdown=True,
    show_tool_calls=True
)

# Agent 2: Test Case Generator
test_generator = Agent(
    name="Test Case Generator",
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    description="Generates PyNVMe test cases based on requirements",
    instructions=[
        "You are a test automation expert specializing in PyNVMe test generation",
        "Generate complete, working PyNVMe test cases based on user requirements",
        "Follow pytest conventions and PyNVMe best practices",
        "Include proper fixtures, assertions, and error handling",
        "Add comprehensive docstrings explaining test purpose",
        "Use the knowledge base to ensure API correctness",
        "Generate tests that are maintainable and well-structured",
        "When you generate a test, use the save_test_case tool to save it",
        "Always use get_test_template to start with a proper template"
    ],
    knowledge_base=pynvme_knowledge,
    search_knowledge=True,
    tools=[
        Function(function=get_test_template, name="get_test_template", description="Get a test template based on test type"),
        Function(function=save_test_case, name="save_test_case", description="Save a generated test case to disk")
    ],
    markdown=True,
    show_tool_calls=True
)

# Agent 3: Test Validator
test_validator = Agent(
    name="Test Validator",
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    description="Validates and improves PyNVMe test cases",
    instructions=[
        "You are a quality assurance expert for PyNVMe tests",
        "Review generated test cases for correctness and best practices",
        "Use validate_test_syntax tool to check test structure",
        "Use check_pynvme_best_practices tool to verify best practices",
        "Use suggest_test_improvements tool to provide actionable feedback",
        "Be thorough but constructive in your feedback",
        "Verify that tests follow PyNVMe API patterns correctly",
        "Check for proper resource management and cleanup",
        "Ensure tests are deterministic and reproducible"
    ],
    knowledge_base=pynvme_knowledge,
    search_knowledge=True,
    tools=[
        Function(function=validate_test_syntax, name="validate_test_syntax", description="Validate the syntax and structure of a PyNVMe test case"),
        Function(function=check_pynvme_best_practices, name="check_pynvme_best_practices", description="Check if the test follows PyNVMe best practices"),
        Function(function=suggest_test_improvements, name="suggest_test_improvements", description="Generate specific improvement suggestions for a test case")
    ],
    markdown=True,
    show_tool_calls=True
)

# Agent 4: Orchestrator - Coordinates Generator and Validator
test_orchestrator = Agent(
    name="Test Orchestrator",
    team=[test_generator, test_validator, pynvme_expert],
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    description="Orchestrates test generation and validation workflow",
    instructions=[
        "You coordinate the test generation and validation process",
        "First, delegate to PyNVMe Expert if user needs clarification on requirements",
        "Then delegate to Test Generator to create the test case",
        "After generation, delegate to Test Validator to review the test",
        "If validation fails, work with Generator to fix issues",
        "Iterate until the test passes validation",
        "Provide a final summary of the generated and validated test",
        "Be efficient - don't repeat work unnecessarily"
    ],
    knowledge_base=pynvme_knowledge,
    search_knowledge=True,
    markdown=True,
    show_tool_calls=True
)

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="PyNVMe Test Generation System",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 PyNVMe Test Case Generation System")
st.markdown("### AI-Powered Test Automation with Multi-Agent Validation")

# Initialize session state
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'kb_loaded' not in st.session_state:
    st.session_state.kb_loaded = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Sidebar
with st.sidebar:
    st.header("📚 PyNVMe Documentation")
    
    st.subheader("Upload Documentation")
    doc_files = st.file_uploader(
        "Upload PyNVMe docs (.pdf, .txt, .md)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=f"pynvme_docs_{st.session_state.uploader_key}"
    )
    
    # Create documentation directory
    os.makedirs("pynvme_docs", exist_ok=True)
    
    # Handle file uploads
    if doc_files and len(doc_files) > 0:
        for doc_file in doc_files:
            file_path = os.path.join("pynvme_docs", doc_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(doc_file.getbuffer())
        
        if st.button("📥 Load Documentation", type="primary"):
            with st.spinner("Loading documentation into knowledge base..."):
                try:
                    pynvme_knowledge.load(recreate=False, upsert=True)
                    st.session_state.kb_loaded = True
                    st.success("✅ Documentation loaded!")
                except Exception as e:
                    st.error(f"Error loading documentation: {str(e)}")
    
    if st.button("🔄 Clear Knowledge Base", type="secondary"):
        try:
            vector_db.client.delete_collection("pynvme_docs")
            gc.collect()
            
            for folder in ["pynvme_docs", VECTOR_DB_PATH, "generated_tests"]:
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder, ignore_errors=True)
                        st.info(f"🗑️ Deleted {folder}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not fully delete {folder}: {e}")
            
            st.session_state.uploader_key += 1
            st.session_state.kb_loaded = False
            st.success("✅ Knowledge base cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error clearing knowledge base: {str(e)}")
    
    st.markdown("---")
    st.subheader("⚙️ Configuration")
    st.text(f"Server: {OLLAMA_HOST}")
    st.text(f"Model: {MODEL_NAME}")
    
    # Show loaded files
    doc_count = len(list(Path("pynvme_docs").glob("*"))) if os.path.exists("pynvme_docs") else 0
    st.metric("Docs Loaded", doc_count)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Orchestrated Generation",
    "✍️ Generate Tests",
    "✅ Validate Tests",
    "📖 PyNVMe Expert",
    "📊 Generated Tests"
])

# Tab 1: Orchestrated Generation (Generator + Validator)
with tab1:
    st.header("Orchestrated Test Generation & Validation")
    st.markdown("*The orchestrator coordinates generation and validation automatically*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_requirement = st.text_area(
            "Describe the test you need:",
            placeholder="e.g., 'Generate a test for sequential 4K read performance with QD=32'",
            height=150,
            key="orchestrator_req"
        )
    
    with col2:
        test_category = st.selectbox(
            "Test Category",
            ["performance", "functional", "stress", "admin", "error_handling", "endurance"],
            key="orch_category"
        )
        
        validation_level = st.select_slider(
            "Validation Strictness",
            options=["Basic", "Standard", "Strict"],
            value="Standard"
        )
    
    if st.button("🚀 Generate & Validate Test", type="primary", key="orchestrate_btn"):
        if not test_requirement:
            st.warning("Please describe the test requirement")
        else:
            with st.spinner("Orchestrating test generation and validation..."):
                try:
                    # Orchestrator handles the entire workflow
                    enhanced_query = f"""
                    Generate and validate a PyNVMe test with these requirements:
                    {test_requirement}
                    
                    Category: {test_category}
                    Validation Level: {validation_level}
                    
                    Workflow:
                    1. Generate a complete test case following PyNVMe best practices
                    2. Validate the generated test thoroughly
                    3. If issues found, regenerate and validate again
                    4. Save the final validated test
                    5. Provide a summary of the generation and validation process
                    """
                    
                    response = test_orchestrator.run(enhanced_query)
                    
                    st.success("✅ Test Generated and Validated!")
                    st.markdown(response.content)
                    
                    st.session_state.generation_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "Orchestrated",
                        "requirement": test_requirement,
                        "category": test_category,
                        "response": response.content
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)

# Tab 2: Generate Tests
with tab2:
    st.header("Test Case Generator")
    st.markdown("*Generate PyNVMe test cases based on specifications*")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        gen_query = st.text_area(
            "Test specification:",
            placeholder="e.g., 'Create a test for random write with various block sizes (512B to 128KB)'",
            height=120,
            key="gen_query"
        )
    
    with col2:
        gen_category = st.selectbox(
            "Category",
            ["performance", "functional", "stress", "admin", "error_handling"],
            key="gen_cat"
        )
    
    if st.button("Generate Test", type="primary", key="gen_btn"):
        if not gen_query:
            st.warning("Please enter test specification")
        else:
            with st.spinner("Generating test case..."):
                try:
                    response = test_generator.run(f"{gen_query}\nCategory: {gen_category}")
                    st.success("✅ Test Generated!")
                    st.markdown(response.content)
                    
                    st.session_state.generation_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "Generation",
                        "requirement": gen_query,
                        "category": gen_category,
                        "response": response.content
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Tab 3: Validate Tests
with tab3:
    st.header("Test Case Validator")
    st.markdown("*Validate and improve existing test cases*")
    
    test_code_input = st.text_area(
        "Paste test code to validate:",
        height=300,
        key="validate_input",
        placeholder="""import pytest
import pynvme as nvme

def test_example(nvme0):
    # Your test code here
    pass
"""
    )
    
    if st.button("Validate Test", type="primary", key="val_btn"):
        if not test_code_input:
            st.warning("Please paste test code")
        else:
            with st.spinner("Validating test case..."):
                try:
                    response = test_validator.run(
                        f"Validate this PyNVMe test and provide improvement suggestions:\n\n{test_code_input}"
                    )
                    st.markdown(response.content)
                    
                    st.session_state.generation_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "Validation",
                        "requirement": "Code validation",
                        "category": "validation",
                        "response": response.content
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Tab 4: PyNVMe Expert
with tab4:
    st.header("PyNVMe Expert")
    st.markdown("*Ask questions about PyNVMe library and NVMe testing*")
    
    expert_query = st.text_area(
        "Ask the PyNVMe expert:",
        placeholder="e.g., 'How do I use the qpair fixture for parallel I/O?'",
        height=100,
        key="expert_query"
    )
    
    if st.button("Ask Expert", type="primary", key="expert_btn"):
        if not expert_query:
            st.warning("Please enter a question")
        else:
            with st.spinner("Consulting PyNVMe expert..."):
                try:
                    response = pynvme_expert.run(expert_query)
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Tab 5: Generated Tests
with tab5:
    st.header("Generated Test Cases")
    
    if os.path.exists("generated_tests"):
        categories = [d for d in os.listdir("generated_tests") if os.path.isdir(os.path.join("generated_tests", d))]
        
        if categories:
            selected_category = st.selectbox("Select Category", ["All"] + categories)
            
            all_tests = []
            if selected_category == "All":
                for cat in categories:
                    cat_path = Path("generated_tests") / cat
                    all_tests.extend([(cat, f) for f in cat_path.glob("*.py")])
            else:
                cat_path = Path("generated_tests") / selected_category
                all_tests.extend([(selected_category, f) for f in cat_path.glob("*.py")])
            
            if all_tests:
                st.metric("Total Tests", len(all_tests))
                
                for category, test_file in all_tests:
                    with st.expander(f"📄 {category}/{test_file.name}"):
                        with open(test_file, "r") as f:
                            st.code(f.read(), language="python")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📋 Copy", key=f"copy_{test_file}"):
                                st.info("Use the copy button in the code block above")
                        with col2:
                            if st.button("🗑️ Delete", key=f"del_{test_file}"):
                                os.remove(test_file)
                                st.success("Deleted!")
                                st.rerun()
            else:
                st.info("No tests generated yet")
        else:
            st.info("No test categories found")
    else:
        st.info("No tests generated yet")

# History sidebar
if st.session_state.generation_history:
    with st.expander("📜 Generation History", expanded=False):
        for i, entry in enumerate(reversed(st.session_state.generation_history)):
            st.markdown(f"**{entry['type']} #{len(st.session_state.generation_history) - i}**")
            st.markdown(f"*Time:* {entry['timestamp']}")
            st.markdown(f"*Category:* {entry['category']}")
            st.markdown(f"*Requirement:* {entry['requirement']}")
            with st.container():
                st.markdown(entry['response'][:500] + "..." if len(entry['response']) > 500 else entry['response'])
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("**System Status**")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", MODEL_NAME)
with col2:
    doc_count = len(list(Path("pynvme_docs").glob("*"))) if os.path.exists("pynvme_docs") else 0
    st.metric("Docs", doc_count)
with col3:
    test_count = len(list(Path("generated_tests").rglob("*.py"))) if os.path.exists("generated_tests") else 0
    st.metric("Generated Tests", test_count)