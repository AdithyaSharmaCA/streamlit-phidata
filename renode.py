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
import re

# SECURITY: Disable any external connections
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['NO_PROXY'] = '*'

# Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "phi3:mini"
VECTOR_DB_PATH = "./renode_chroma_db"

# Initialize embedder
embedder = OllamaEmbedder(
    model="nomic-embed-text",
    host=OLLAMA_HOST,
)

# Initialize vector database
vector_db = ChromaDb(
    collection="renode_docs",
    path=VECTOR_DB_PATH,
    embedder=embedder,
    persistent_client=True
)

# Initialize Knowledge Base for RENODE documentation
renode_knowledge = CombinedKnowledgeBase(
    sources=[
        PDFKnowledgeBase(
            path="renode_docs",
            vector_db=vector_db,
        ),
        TextKnowledgeBase(
            path="renode_docs",
            vector_db=vector_db,
        )
    ]
)

# ============================================================================
# TOOL FUNCTIONS - These will be used by agents to interact with each other
# ============================================================================

def validate_renode_syntax(code: str, code_type: str) -> Dict[str, any]:
    """
    Validate the syntax and structure of RENODE code.
    
    Args:
        code: The RENODE code to validate
        code_type: Type of code (resc, repl, cs, platform)
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": [],
        "code_type": code_type
    }
    
    if code_type == "resc":
        # Validate RENODE script (.resc)
        if "mach create" not in code and "using sysbus" not in code:
            validation_results["warnings"].append("No machine creation or sysbus usage found")
        
        # Check for common commands
        common_commands = ["sysbus LoadELF", "machine StartGdbServer", "start"]
        found_commands = [cmd for cmd in common_commands if cmd in code]
        if not found_commands:
            validation_results["suggestions"].append("Consider adding common RENODE commands like LoadELF or start")
    
    elif code_type == "repl":
        # Validate platform description (.repl)
        if not re.search(r'\w+:\s*([\w\.]+)', code):
            validation_results["errors"].append("Invalid REPL syntax - missing peripheral definitions")
            validation_results["is_valid"] = False
        
        # Check for required fields
        if "@" not in code:
            validation_results["warnings"].append("No memory addresses (@) specified")
        
        if "size:" not in code.lower():
            validation_results["suggestions"].append("Consider specifying sizes for memory regions")
    
    elif code_type == "cs":
        # Validate C# peripheral model
        if "using Antmicro.Renode" not in code:
            validation_results["errors"].append("Missing RENODE namespace imports")
            validation_results["is_valid"] = False
        
        if "class" not in code:
            validation_results["errors"].append("No class definition found")
            validation_results["is_valid"] = False
        
        if ": IDoubleWordPeripheral" not in code and ": IBytePeripheral" not in code:
            validation_results["warnings"].append("Consider implementing IDoubleWordPeripheral or IBytePeripheral")
    
    elif code_type == "platform":
        # Validate platform definition
        if "cpu:" not in code.lower() and "nvic:" not in code.lower():
            validation_results["warnings"].append("No CPU or NVIC defined")
        
        if "uart" not in code.lower() and "usart" not in code.lower():
            validation_results["suggestions"].append("Consider adding UART for debugging")
    
    # General checks
    if len(code.strip()) < 10:
        validation_results["errors"].append("Code is too short or empty")
        validation_results["is_valid"] = False
    
    return validation_results


def check_renode_best_practices(code: str, code_type: str) -> Dict[str, List[str]]:
    """
    Check if the code follows RENODE best practices.
    
    Args:
        code: The code to check
        code_type: Type of code (resc, repl, cs, platform)
        
    Returns:
        Dictionary with best practice recommendations
    """
    recommendations = {
        "good_practices": [],
        "improvements_needed": []
    }
    
    if code_type == "resc":
        # RESC script best practices
        if "# " in code or "//" in code:
            recommendations["good_practices"].append("Code includes comments")
        else:
            recommendations["improvements_needed"].append("Add comments to explain script purpose")
        
        if "logLevel" in code:
            recommendations["good_practices"].append("Logging configuration present")
        else:
            recommendations["improvements_needed"].append("Consider setting appropriate logLevel")
        
        if "showAnalyzer" in code or "CreateTerminalTester" in code:
            recommendations["good_practices"].append("Output/analysis tools configured")
        else:
            recommendations["improvements_needed"].append("Add analyzer or terminal tester for debugging")
    
    elif code_type == "repl":
        # REPL best practices
        if "///" in code:
            recommendations["good_practices"].append("Documentation comments present")
        else:
            recommendations["improvements_needed"].append("Add /// documentation comments for peripherals")
        
        if re.search(r'<\s*0x[0-9A-Fa-f]+\s*,\s*\+0x[0-9A-Fa-f]+\s*>', code):
            recommendations["good_practices"].append("Proper address range syntax used")
        
        if "IRQ" in code or "-> " in code:
            recommendations["good_practices"].append("Interrupt connections defined")
        else:
            recommendations["improvements_needed"].append("Consider defining interrupt connections")
    
    elif code_type == "cs":
        # C# peripheral best practices
        if "[Constructor]" in code:
            recommendations["good_practices"].append("Constructor attribute used correctly")
        
        if "this.Log" in code or "this.DebugLog" in code:
            recommendations["good_practices"].append("Logging implemented")
        else:
            recommendations["improvements_needed"].append("Add logging for debugging (this.Log)")
        
        if "Reset()" in code:
            recommendations["good_practices"].append("Reset method implemented")
        else:
            recommendations["improvements_needed"].append("Implement Reset() method")
        
        if "private" in code and "public" in code:
            recommendations["good_practices"].append("Proper access modifiers used")
    
    # Common best practices
    lines = code.split('\n')
    if len(lines) > 100 and code_type in ["resc", "cs"]:
        recommendations["improvements_needed"].append("Consider breaking large files into smaller modules")
    
    return recommendations


def suggest_code_improvements(code: str, code_type: str, validation_results: Dict) -> str:
    """
    Generate specific improvement suggestions for RENODE code.
    
    Args:
        code: The code
        code_type: Type of code
        validation_results: Results from validation
        
    Returns:
        Formatted string with improvement suggestions
    """
    suggestions = []
    
    suggestions.append(f"=== Improvements for {code_type.upper()} Code ===\n")
    
    if not validation_results["is_valid"]:
        suggestions.append("🔴 CRITICAL ERRORS - Must fix:")
        suggestions.extend([f"  ❌ {err}" for err in validation_results["errors"]])
        suggestions.append("")
    
    if validation_results["warnings"]:
        suggestions.append("⚠️  WARNINGS:")
        suggestions.extend([f"  ⚠️  {warn}" for warn in validation_results["warnings"]])
        suggestions.append("")
    
    if validation_results["suggestions"]:
        suggestions.append("💡 SUGGESTIONS:")
        suggestions.extend([f"  💡 {sugg}" for sugg in validation_results["suggestions"]])
        suggestions.append("")
    
    # Type-specific suggestions
    if code_type == "resc":
        suggestions.append("📋 RESC Script Tips:")
        suggestions.append("  • Use 'include @path/to/platform.repl' to load platforms")
        suggestions.append("  • Add 'emulation CreateUartPtyTerminal \"term\" \"/tmp/uart\"' for serial")
        suggestions.append("  • Use 'machine StartGdbServer 3333' for debugging")
    
    elif code_type == "repl":
        suggestions.append("📋 REPL Platform Tips:")
        suggestions.append("  • Use consistent indentation (4 spaces)")
        suggestions.append("  • Define memory regions with @ and size:")
        suggestions.append("  • Connect interrupts with '-> cpu@IRQ_NUMBER'")
    
    elif code_type == "cs":
        suggestions.append("📋 C# Peripheral Tips:")
        suggestions.append("  • Inherit from BasicDoubleWordPeripheral for simplicity")
        suggestions.append("  • Use [Constructor] attribute for dependency injection")
        suggestions.append("  • Implement IDisposable if managing resources")
    
    return "\n".join(suggestions) if suggestions else "✅ Code looks good!"


def save_renode_code(filename: str, code: str, code_type: str, category: str = "general") -> Dict[str, str]:
    """
    Save generated RENODE code to disk.
    
    Args:
        filename: Name of the file (without extension)
        code: The code content
        code_type: Type of code (resc, repl, cs, platform)
        category: Code category
        
    Returns:
        Dictionary with save status
    """
    try:
        # Determine file extension
        extensions = {
            "resc": ".resc",
            "repl": ".repl",
            "cs": ".cs",
            "platform": ".repl"
        }
        ext = extensions.get(code_type, ".txt")
        
        # Create directory structure
        code_dir = Path(f"generated_renode/{category}/{code_type}")
        code_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = re.sub(r'[^\w\-_]', '_', filename)
        full_filename = f"{safe_filename}_{timestamp}{ext}"
        filepath = code_dir / full_filename
        
        # Save file
        with open(filepath, "w") as f:
            f.write(code)
        
        return {
            "status": "success",
            "filepath": str(filepath),
            "message": f"Code saved successfully to {filepath}"
        }
    except Exception as e:
        return {
            "status": "error",
            "filepath": "",
            "message": f"Error saving code: {str(e)}"
        }


def get_renode_template(template_type: str) -> str:
    """
    Get a RENODE code template based on type.
    
    Args:
        template_type: Type of template (resc_basic, repl_platform, cs_peripheral, etc.)
        
    Returns:
        Template code string
    """
    templates = {
        "resc_basic": '''# RENODE Script Template
# Create and configure a basic machine

mach create "machine_name"
machine LoadPlatformDescription @platforms/boards/your_board.repl

# Load binary
sysbus LoadELF @path/to/firmware.elf

# Setup logging
logLevel -1

# Start simulation
start
''',
        "resc_with_uart": '''# RENODE Script with UART
mach create
machine LoadPlatformDescription @platforms/boards/stm32f4_discovery.repl

# Create UART terminal
emulation CreateUartPtyTerminal "term" "/tmp/uart"
connector Connect sysbus.usart1 term

# Load and start
sysbus LoadELF @firmware.elf
start
''',
        "repl_platform": '''// Platform Description Template
// Define your custom platform here

cpu: CPU.CortexM @ sysbus
    cpuType: "cortex-m4"
    nvic: nvic

nvic: IRQControllers.NVIC @ sysbus 0xE000E000
    systickFrequency: 72000000
    IRQ -> cpu@0

sram: Memory.MappedMemory @ sysbus 0x20000000
    size: 0x20000

flash: Memory.MappedMemory @ sysbus 0x08000000
    size: 0x100000
''',
        "repl_uart": '''// UART Peripheral Definition

uart0: UART.STM32_UART @ sysbus <0x40011000, +0x400>
    frequency: 200000000
    IRQ -> nvic@37

uart1: UART.STM32_UART @ sysbus <0x40004400, +0x400>
    frequency: 200000000
    IRQ -> nvic@38
''',
        "cs_peripheral": '''using System;
using Antmicro.Renode.Core;
using Antmicro.Renode.Peripherals.Bus;
using Antmicro.Renode.Logging;

namespace Antmicro.Renode.Peripherals.CustomPeripherals
{
    public class CustomPeripheral : BasicDoubleWordPeripheral, IKnownSize
    {
        public CustomPeripheral(Machine machine) : base(machine)
        {
            DefineRegisters();
        }

        public long Size => 0x1000;

        public override void Reset()
        {
            base.Reset();
            // Reset peripheral state
        }

        private void DefineRegisters()
        {
            Registers.Control.Define(this)
                .WithFlag(0, out enabled, name: "EN")
                .WithReservedBits(1, 31);

            Registers.Status.Define(this)
                .WithFlag(0, FieldMode.Read, name: "READY")
                .WithReservedBits(1, 31);
        }

        private IFlagRegisterField enabled;

        private enum Registers : long
        {
            Control = 0x00,
            Status = 0x04
        }
    }
}
''',
        "robot_test": '''*** Settings ***
Suite Setup                   Setup
Suite Teardown                Teardown
Test Setup                    Reset Emulation
Test Teardown                 Test Teardown
Resource                      ${RENODEKEYWORDS}

*** Variables ***
${UART}                       sysbus.uart0
${URI}                        @https://dl.antmicro.com/projects/renode

*** Test Cases ***
Should Boot And Print
    Execute Command          mach create
    Execute Command          machine LoadPlatformDescription @platforms/boards/stm32f4_discovery.repl
    Execute Command          sysbus LoadELF ${URI}/stm32f4_discovery--zephyr-shell.elf-s_184340-4d614eb2b1906ea4a2a05b313c025c3c4ba7c2e0
    
    Create Terminal Tester   ${UART}
    
    Start Emulation
    Wait For Line On Uart    Booting Zephyr
'''
    }
    
    return templates.get(template_type, templates["resc_basic"])


def analyze_platform_compatibility(code: str, target_arch: str = "ARM") -> Dict[str, any]:
    """
    Analyze platform code for compatibility with target architecture.
    
    Args:
        code: Platform code to analyze
        target_arch: Target architecture (ARM, RISC-V, etc.)
        
    Returns:
        Dictionary with compatibility analysis
    """
    analysis = {
        "compatible": True,
        "architecture": target_arch,
        "detected_peripherals": [],
        "issues": [],
        "recommendations": []
    }
    
    # Detect peripherals
    peripheral_patterns = {
        "uart": r'UART[.\w]*',
        "gpio": r'GPIO[.\w]*',
        "timer": r'Timer[.\w]*',
        "spi": r'SPI[.\w]*',
        "i2c": r'I2C[.\w]*',
    }
    
    for periph_type, pattern in peripheral_patterns.items():
        if re.search(pattern, code, re.IGNORECASE):
            analysis["detected_peripherals"].append(periph_type.upper())
    
    # Check architecture-specific elements
    if target_arch == "ARM":
        if "CortexM" not in code and "CortexA" not in code:
            analysis["issues"].append("No ARM Cortex CPU defined")
            analysis["compatible"] = False
        
        if "NVIC" not in code and "GIC" not in code:
            analysis["recommendations"].append("Consider adding interrupt controller (NVIC/GIC)")
    
    elif target_arch == "RISC-V":
        if "RiscV" not in code:
            analysis["issues"].append("No RISC-V CPU defined")
            analysis["compatible"] = False
    
    return analysis


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

# Agent 1: RENODE Expert - Knowledge Base Agent
renode_expert = Agent(
    name="RENODE Expert",
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    description="Expert in RENODE emulation framework, platform descriptions, and scripting",
    instructions=[
        "You are an expert in RENODE emulation framework",
        "Provide accurate information about RENODE scripts (.resc), platform descriptions (.repl), and C# peripherals",
        "Reference the documentation from the knowledge base",
        "Explain RENODE concepts clearly with examples",
        "Help with platform modeling, peripheral development, and test automation",
        "Search the knowledge base before answering questions",
        "Provide practical examples for common use cases"
    ],
    knowledge_base=renode_knowledge,
    search_knowledge=True,
    markdown=True,
    show_tool_calls=True
)

# Agent 2: Code Generator
code_generator = Agent(
    name="RENODE Code Generator",
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    description="Generates RENODE scripts, platform descriptions, and peripherals",
    instructions=[
        "You are a RENODE code generation expert",
        "Generate complete, working RENODE code based on user requirements",
        "Support multiple code types: .resc scripts, .repl platforms, C# peripherals, Robot tests",
        "Follow RENODE conventions and best practices",
        "Include comprehensive comments explaining the code",
        "Use the knowledge base to ensure API correctness",
        "Always use get_renode_template to start with proper templates",
        "After generating code, use save_renode_code to save it",
        "Generate production-ready, well-structured code"
    ],
    knowledge_base=renode_knowledge,
    search_knowledge=True,
    tools=[
        Function(function=get_renode_template, name="get_renode_template", description="Get a RENODE code template based on type"),
        Function(function=save_renode_code, name="save_renode_code", description="Save generated RENODE code to disk"),
        Function(function=analyze_platform_compatibility, name="analyze_platform_compatibility", description="Analyze platform code for architecture compatibility")
    ],
    markdown=True,
    show_tool_calls=True
)

# Agent 3: Code Validator
code_validator = Agent(
    name="RENODE Code Validator",
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    description="Validates and improves RENODE code",
    instructions=[
        "You are a quality assurance expert for RENODE code",
        "Review generated code for correctness and best practices",
        "Use validate_renode_syntax to check code structure",
        "Use check_renode_best_practices to verify conventions",
        "Use suggest_code_improvements to provide actionable feedback",
        "Be thorough but constructive in your reviews",
        "Verify platform descriptions are syntactically correct",
        "Check that peripherals follow RENODE API patterns",
        "Ensure scripts are executable and well-documented"
    ],
    knowledge_base=renode_knowledge,
    search_knowledge=True,
    tools=[
        Function(function=validate_renode_syntax, name="validate_renode_syntax", description="Validate RENODE code syntax and structure"),
        Function(function=check_renode_best_practices, name="check_renode_best_practices", description="Check if code follows RENODE best practices"),
        Function(function=suggest_code_improvements, name="suggest_code_improvements", description="Generate improvement suggestions for code"),
        Function(function=analyze_platform_compatibility, name="analyze_platform_compatibility", description="Analyze platform compatibility")
    ],
    markdown=True,
    show_tool_calls=True
)

# Agent 4: Orchestrator
code_orchestrator = Agent(
    name="RENODE Code Orchestrator",
    team=[code_generator, code_validator, renode_expert],
    model=Ollama(id=MODEL_NAME, host=OLLAMA_HOST),
    description="Orchestrates code generation and validation workflow",
    instructions=[
        "You coordinate the RENODE code generation and validation process",
        "First, delegate to RENODE Expert if requirements need clarification",
        "Then delegate to Code Generator to create the code",
        "After generation, delegate to Code Validator to review",
        "If validation fails, work with Generator to fix issues",
        "Iterate until code passes validation",
        "Provide a comprehensive summary of the process",
        "Be efficient - avoid unnecessary iterations"
    ],
    knowledge_base=renode_knowledge,
    search_knowledge=True,
    markdown=True,
    show_tool_calls=True
)

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="RENODE Code Generation System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 RENODE Code Generation System")
st.markdown("### AI-Powered Emulation Framework Code Generation")

# Initialize session state
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'kb_loaded' not in st.session_state:
    st.session_state.kb_loaded = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Sidebar
with st.sidebar:
    st.header("📚 RENODE Documentation")
    
    st.subheader("Upload Documentation")
    doc_files = st.file_uploader(
        "Upload RENODE docs (.pdf, .txt, .md, .rst)",
        type=["pdf", "txt", "md", "rst"],
        accept_multiple_files=True,
        key=f"renode_docs_{st.session_state.uploader_key}"
    )
    
    # Create documentation directory
    os.makedirs("renode_docs", exist_ok=True)
    
    # Handle file uploads
    if doc_files and len(doc_files) > 0:
        for doc_file in doc_files:
            file_path = os.path.join("renode_docs", doc_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(doc_file.getbuffer())
        
        if st.button("📥 Load Documentation", type="primary"):
            with st.spinner("Loading documentation into knowledge base..."):
                try:
                    renode_knowledge.load(recreate=False, upsert=True)
                    st.session_state.kb_loaded = True
                    st.success("✅ Documentation loaded!")
                except Exception as e:
                    st.error(f"Error loading documentation: {str(e)}")
    
    if st.button("🔄 Clear Knowledge Base", type="secondary"):
        try:
            vector_db.client.delete_collection("renode_docs")
            gc.collect()
            
            for folder in ["renode_docs", VECTOR_DB_PATH, "generated_renode"]:
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
    doc_count = len(list(Path("renode_docs").glob("*"))) if os.path.exists("renode_docs") else 0
    st.metric("Docs Loaded", doc_count)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Orchestrated Generation",
    "✍️ Generate Code",
    "✅ Validate Code",
    "📖 RENODE Expert",
    "📊 Generated Code"
])

# Tab 1: Orchestrated Generation
with tab1:
    st.header("Orchestrated Code Generation & Validation")
    st.markdown("*Automated generation and validation workflow*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        code_requirement = st.text_area(
            "Describe what you need:",
            placeholder="e.g., 'Create a platform description for STM32F4 with UART, GPIO, and SPI'",
            height=150,
            key="orchestrator_req"
        )
    
    with col2:
        code_type = st.selectbox(
            "Code Type",
            ["resc", "repl", "cs", "robot"],
            format_func=lambda x: {
                "resc": "RESC Script",
                "repl": "Platform Description",
                "cs": "C# Peripheral",
                "robot": "Robot Test"
            }[x],
            key="orch_type"
        )
        
        category = st.selectbox(
            "Category",
            ["platforms", "peripherals", "tests", "examples", "custom"],
            key="orch_category"
        )
        
        architecture = st.selectbox(
            "Target Architecture",
            ["ARM Cortex-M", "ARM Cortex-A", "RISC-V", "x86", "Other"],
            key="orch_arch"
        )
    
    if st.button("🚀 Generate & Validate", type="primary", key="orchestrate_btn"):
        if not code_requirement:
            st.warning("Please describe your requirements")
        else:
            with st.spinner("Orchestrating code generation and validation..."):
                try:
                    enhanced_query = f"""
                    Generate and validate RENODE code with these requirements:
                    {code_requirement}
                    
                    Code Type: {code_type}
                    Category: {category}
                    Target Architecture: {architecture}
                    
                    Workflow:
                    1. Generate complete {code_type} code following RENODE best practices
                    2. Validate the generated code thoroughly
                    3. Check architecture compatibility
                    4. If issues found, regenerate and validate again
                    5. Save the final validated code
                    6. Provide a summary with usage instructions
                    """
                    
                    response = code_orchestrator.run(enhanced_query)
                    
                    st.success("✅ Code Generated and Validated!")
                    st.markdown(response.content)
                    
                    st.session_state.generation_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "Orchestrated",
                        "code_type": code_type,
                        "requirement": code_requirement,
                        "category": category,
                        "response": response.content
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)

# Tab 2: Generate Code
with tab2:
    st.header("RENODE Code Generator")
    st.markdown("*Generate scripts, platforms, peripherals, and tests*")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        gen_query = st.text_area(
            "Code specification:",
            placeholder="e.g., 'Create a UART peripheral model with FIFO support'",
            height=120,
            key="gen_query"
        )
    
    with col2:
        gen_code_type = st.selectbox(
            "Type",
            ["resc", "repl", "cs", "robot"],
            format_func=lambda x: {
                "resc": "RESC Script",
                "repl": "Platform",
                "cs": "C# Peripheral",
                "robot": "Robot Test"
            }[x],
            key="gen_type"
        )
        
        gen_category = st.selectbox(
            "Category",
            ["platforms", "peripherals", "tests", "examples"],
            key="gen_cat"
        )
    
    # Show available templates
    with st.expander("📋 Available Templates"):
        st.markdown("""
        - **resc_basic**: Basic RENODE script
        - **resc_with_uart**: Script with UART terminal
        - **repl_platform**: Platform description template
        - **repl_uart**: UART peripheral definition
        - **cs_peripheral**: C# peripheral model
        - **robot_test**: Robot Framework test
        """)
    
    if st.button("Generate Code", type="primary", key="gen_btn"):
        if not gen_query:
            st.warning("Please enter code specification")
        else:
            with st.spinner("Generating code..."):
                try:
                    response = code_generator.run(
                        f"{gen_query}\nCode Type: {gen_code_type}\nCategory: {gen_category}"
                    )
                    st.success("✅ Code Generated!")
                    st.markdown(response.content)
                    
                    st.session_state.generation_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "Generation",
                        "code_type": gen_code_type,
                        "requirement": gen_query,
                        "category": gen_category,
                        "response": response.content
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Tab 3: Validate Code
with tab3:
    st.header("RENODE Code Validator")
    st.markdown("*Validate and improve existing code*")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        code_input = st.text_area(
            "Paste code to validate:",
            height=300,
            key="validate_input",
            placeholder="""# Example RESC script
mach create
machine LoadPlatformDescription @platforms/boards/stm32f4.repl
sysbus LoadELF @firmware.elf
start
"""
        )
    
    with col2:
        val_code_type = st.selectbox(
            "Code Type",
            ["resc", "repl", "cs", "platform"],
            format_func=lambda x: {
                "resc": "RESC Script",
                "repl": "Platform",
                "cs": "C# Peripheral",
                "platform": "Platform Desc"
            }[x],
            key="val_type"
        )
        
        val_arch = st.selectbox(
            "Target Arch",
            ["ARM", "RISC-V", "x86", "Other"],
            key="val_arch"
        )
    
    if st.button("Validate Code", type="primary", key="val_btn"):
        if not code_input:
            st.warning("Please paste code to validate")
        else:
            with st.spinner("Validating code..."):
                try:
                    response = code_validator.run(
                        f"Validate this {val_code_type} code for {val_arch} architecture and provide improvement suggestions:\n\n{code_input}"
                    )
                    st.markdown(response.content)
                    
                    st.session_state.generation_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "Validation",
                        "code_type": val_code_type,
                        "requirement": "Code validation",
                        "category": "validation",
                        "response": response.content
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Tab 4: RENODE Expert
with tab4:
    st.header("RENODE Expert Assistant")
    st.markdown("*Ask questions about RENODE framework*")
    
    # Quick questions
    with st.expander("💡 Common Questions"):
        quick_questions = [
            "How do I create a custom peripheral in C#?",
            "What's the syntax for platform descriptions?",
            "How do I connect UART to a terminal?",
            "How do I debug with GDB in RENODE?",
            "What are the different memory types in RENODE?",
            "How do I handle interrupts in platform descriptions?"
        ]
        
        for i, question in enumerate(quick_questions):
            if st.button(question, key=f"quick_{i}"):
                st.session_state.expert_query_preset = question
    
    expert_query = st.text_area(
        "Ask the RENODE expert:",
        placeholder="e.g., 'How do I implement a DMA controller peripheral?'",
        height=100,
        key="expert_query",
        value=st.session_state.get('expert_query_preset', '')
    )
    
    if st.button("Ask Expert", type="primary", key="expert_btn"):
        if not expert_query:
            st.warning("Please enter a question")
        else:
            with st.spinner("Consulting RENODE expert..."):
                try:
                    response = renode_expert.run(expert_query)
                    st.markdown(response.content)
                    
                    # Clear preset after use
                    if 'expert_query_preset' in st.session_state:
                        del st.session_state.expert_query_preset
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Tab 5: Generated Code Browser
with tab5:
    st.header("Generated Code Browser")
    
    if os.path.exists("generated_renode"):
        # Get all categories
        categories = []
        for cat in os.listdir("generated_renode"):
            cat_path = os.path.join("generated_renode", cat)
            if os.path.isdir(cat_path):
                categories.append(cat)
        
        if categories:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                selected_category = st.selectbox("Category", ["All"] + categories)
            
            with col2:
                selected_type = st.selectbox(
                    "Code Type",
                    ["All", "resc", "repl", "cs", "robot"],
                    format_func=lambda x: x if x == "All" else {
                        "resc": "RESC Scripts",
                        "repl": "Platforms",
                        "cs": "C# Peripherals",
                        "robot": "Robot Tests"
                    }[x]
                )
            
            # Collect all files
            all_files = []
            search_categories = categories if selected_category == "All" else [selected_category]
            
            for cat in search_categories:
                cat_path = Path("generated_renode") / cat
                if cat_path.exists():
                    for type_dir in cat_path.iterdir():
                        if type_dir.is_dir():
                            code_type = type_dir.name
                            if selected_type == "All" or code_type == selected_type:
                                for code_file in type_dir.glob("*.*"):
                                    all_files.append((cat, code_type, code_file))
            
            if all_files:
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Files", len(all_files))
                with col2:
                    resc_count = len([f for f in all_files if f[1] == "resc"])
                    st.metric("RESC Scripts", resc_count)
                with col3:
                    repl_count = len([f for f in all_files if f[1] == "repl"])
                    st.metric("Platforms", repl_count)
                with col4:
                    cs_count = len([f for f in all_files if f[1] == "cs"])
                    st.metric("Peripherals", cs_count)
                
                st.markdown("---")
                
                # Display files
                for category, code_type, code_file in sorted(all_files, key=lambda x: x[2].stat().st_mtime, reverse=True):
                    icon = {
                        "resc": "📜",
                        "repl": "🗺️",
                        "cs": "⚙️",
                        "robot": "🤖"
                    }.get(code_type, "📄")
                    
                    with st.expander(f"{icon} {category}/{code_type}/{code_file.name}"):
                        try:
                            with open(code_file, "r") as f:
                                content = f.read()
                            
                            # Show file info
                            file_stat = code_file.stat()
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.text(f"Size: {file_stat.st_size} bytes")
                            with col2:
                                mod_time = datetime.fromtimestamp(file_stat.st_mtime)
                                st.text(f"Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
                            with col3:
                                st.text(f"Type: {code_type.upper()}")
                            
                            # Display code
                            language_map = {
                                "resc": "bash",
                                "repl": "yaml",
                                "cs": "csharp",
                                "robot": "robotframework"
                            }
                            st.code(content, language=language_map.get(code_type, "text"))
                            
                            # Action buttons
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("📋 Copy Path", key=f"copy_{code_file}"):
                                    st.code(str(code_file), language="bash")
                            with col2:
                                if st.button("🔍 Validate", key=f"validate_{code_file}"):
                                    with st.spinner("Validating..."):
                                        val_response = code_validator.run(
                                            f"Validate this {code_type} code:\n\n{content}"
                                        )
                                        st.info(val_response.content)
                            with col3:
                                if st.button("🗑️ Delete", key=f"del_{code_file}"):
                                    os.remove(code_file)
                                    st.success("Deleted!")
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
            else:
                st.info("No files match the selected filters")
        else:
            st.info("No categories found")
    else:
        st.info("No code generated yet. Start by using the generation tabs!")

# Generation History
if st.session_state.generation_history:
    with st.expander("📜 Generation History", expanded=False):
        for i, entry in enumerate(reversed(st.session_state.generation_history)):
            st.markdown(f"**{entry['type']} #{len(st.session_state.generation_history) - i}**")
            st.markdown(f"*Time:* {entry['timestamp']}")
            st.markdown(f"*Type:* {entry['code_type']} | *Category:* {entry['category']}")
            st.markdown(f"*Requirement:* {entry['requirement']}")
            
            # Show preview of response
            preview = entry['response'][:400] + "..." if len(entry['response']) > 400 else entry['response']
            with st.container():
                st.markdown(preview)
            
            if st.button("Show Full Response", key=f"show_{i}"):
                st.markdown(entry['response'])
            
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("**System Status**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", MODEL_NAME)

with col2:
    doc_count = len(list(Path("renode_docs").glob("*"))) if os.path.exists("renode_docs") else 0
    st.metric("Docs", doc_count)

with col3:
    code_count = len(list(Path("generated_renode").rglob("*.*"))) if os.path.exists("generated_renode") else 0
    st.metric("Generated Files", code_count)

with col4:
    kb_status = "✅ Loaded" if st.session_state.kb_loaded else "⚠️ Not Loaded"
    st.metric("Knowledge Base", kb_status)

# Help section
with st.expander("ℹ️ Help & Documentation"):
    st.markdown("""
    ### Quick Start Guide
    
    1. **Upload Documentation**: Add RENODE docs in the sidebar
    2. **Load Knowledge Base**: Click "Load Documentation" button
    3. **Generate Code**: Use the orchestrated generation or individual tabs
    4. **Validate**: All generated code is automatically validated
    
    ### Code Types Supported
    
    - **RESC Scripts** (`.resc`): RENODE execution scripts for running simulations
    - **Platform Descriptions** (`.repl`): Hardware platform definitions
    - **C# Peripherals** (`.cs`): Custom peripheral models
    - **Robot Tests** (`.robot`): Automated test cases
    
    ### Architecture Support
    
    - ARM Cortex-M (STM32, NRF, etc.)
    - ARM Cortex-A (Raspberry Pi, Zynq, etc.)
    - RISC-V (SiFive, ESP32-C3, etc.)
    - x86 and others
    
    ### Best Practices
    
    - Always validate generated code before use
    - Include comments for complex configurations
    - Use appropriate memory addresses for your target
    - Test with small examples first
    
    ### Resources
    
    - [RENODE Documentation](https://renode.readthedocs.io/)
    - [Platform Descriptions Guide](https://renode.readthedocs.io/en/latest/basic/describing_platforms.html)
    - [Creating Peripherals](https://renode.readthedocs.io/en/latest/advanced/writing_peripherals.html)
    """)