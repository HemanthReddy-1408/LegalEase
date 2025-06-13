import streamlit as st
import uuid
import time
from datetime import datetime
from agent.rag_chain import generate_answer, get_model_info, reset_conversation
from agent.memory_config import save_memory, clear_session_memory, get_session_list
import os

# Page configuration
st.set_page_config(
    page_title="LegalEase GPT", 
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
    }
    .user-message {
        background-color: #f0f8ff;
        border-left-color: #4CAF50;
    }
    .bot-message {
        background-color: #f9f9f9;
        border-left-color: #1f4e79;
    }
    .source-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
        font-size: 0.9rem;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, #1f4e79, #2d5aa0);
        color: white;
        font-weight: bold;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    
    if "loading_error" not in st.session_state:
        st.session_state.loading_error = None

def format_confidence(confidence):
    """Format confidence score with appropriate styling"""
    if confidence >= 0.7:
        return f'<span class="confidence-high">High ({confidence:.1%})</span>'
    elif confidence >= 0.4:
        return f'<span class="confidence-medium">Medium ({confidence:.1%})</span>'
    else:
        return f'<span class="confidence-low">Low ({confidence:.1%})</span>'

def display_sources(sources):
    """Display source documents in an organized way"""
    if not sources:
        return
    
    st.markdown("**📚 Sources:**")
    for i, source in enumerate(sources, 1):
        section = source.get("section", "Unknown")
        title = source.get("title", "")
        content = source.get("content", "")
        
        source_text = f"**Section {section}**"
        if title:
            source_text += f": {title}"
        
        with st.expander(f"Source {i}: Section {section}", expanded=False):
            st.markdown(source_text)
            if content:
                st.markdown(f"*{content}*")

def display_chat_message(role, message, metadata=None):
    """Display a chat message with proper formatting"""
    if role == "user":
        st.markdown(f'<div class="chat-message user-message"><strong>🧑 You:</strong> {message}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message"><strong>⚖️ LegalEase GPT:</strong> {message}</div>', 
                   unsafe_allow_html=True)
        
        # Display metadata if available
        if metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                if "confidence" in metadata:
                    confidence_html = format_confidence(metadata["confidence"])
                    st.markdown(f"**Confidence:** {confidence_html}", unsafe_allow_html=True)
            
            with col2:
                if "timestamp" in metadata:
                    timestamp = datetime.fromisoformat(metadata["timestamp"])
                    st.markdown(f"**Time:** {timestamp.strftime('%H:%M:%S')}")
            
            # Display sources
            if "sources" in metadata and metadata["sources"]:
                display_sources(metadata["sources"])

def check_prerequisites():
    """Check if all prerequisites are met"""
    vector_store_path = "rag/vector_store"
    
    if not os.path.exists(vector_store_path):
        st.error("❌ Vector database not found!")
        st.markdown("""
        **Setup Required:**
        1. Create a `data` folder in your project directory
        2. Add your legal PDF files (IPC, CrPC, Constitution) to the `data` folder
        3. Run: `python embed_pdf.py` to create the vector database
        4. Then restart this Streamlit app
        """)
        return False
    
    required_files = ["index.faiss", "index.pkl"]
    for file in required_files:
        if not os.path.exists(os.path.join(vector_store_path, file)):
            st.error(f"❌ Required file missing: {file}")
            return False
    
    return True

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ LegalEase GPT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions related to IPC, CrPC, or Constitution of India 📚</p>', 
               unsafe_allow_html=True)
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Sidebar for additional options
    with st.sidebar:
        st.header("⚙️ Options")
        
        # Model status
        try:
            model_info = get_model_info()
            st.success("✅ Model Ready")
            
            with st.expander("📊 Model Info", expanded=False):
                st.json(model_info)
        except Exception as e:
            st.error(f"❌ Model Error: {str(e)}")
        
        # Session management
        st.subheader("💾 Session")
        st.text(f"ID: {st.session_state.session_id[:8]}...")
        
        if st.button("🔄 New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()
        
        # Clear conversation
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            clear_session_memory(st.session_state.session_id)
            reset_conversation()
            st.rerun()
        
        # Display chat statistics
        if st.session_state.chat_history:
            st.subheader("📈 Stats")
            st.metric("Messages", len(st.session_state.chat_history))
            
            # Average confidence
            confidences = [msg.get("metadata", {}).get("confidence", 0) 
                          for role, msg, metadata in st.session_state.chat_history 
                          if role == "bot" and metadata]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Main chat interface
    st.subheader("💬 Chat")
    
    # Input field
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Your Legal Question:", 
                placeholder="e.g., What is Section 420 of IPC?",
                key="user_question"
            )
        
        with col2:
            submit_button = st.form_submit_button("Ask 📤")
    
    # Process user input
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append(("user", user_input, None))
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                # Get response with metadata
                response_data = generate_answer(
                    user_input, 
                    session_id=st.session_state.session_id,
                    debug=False
                )
                
                answer = response_data.get("answer", "I'm sorry, I couldn't generate a response.")
                metadata = {
                    "confidence": response_data.get("confidence", 0),
                    "sources": response_data.get("sources", []),
                    "timestamp": response_data.get("timestamp", datetime.now().isoformat())
                }
                
                # Add bot response to chat history
                st.session_state.chat_history.append(("bot", answer, metadata))
                
            except Exception as e:
                error_message = f"I encountered an error: {str(e)}"
                st.session_state.chat_history.append(("bot", error_message, None))
                st.error(f"Error: {e}")
        
        # Rerun to display new messages
        st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        
        # Display messages in reverse order (newest first)
        for role, message, metadata in reversed(st.session_state.chat_history):
            display_chat_message(role, message, metadata)
            st.markdown("<br>", unsafe_allow_html=True)
    
    else:
        # Show example questions when chat is empty
        st.markdown("### 💡 Example Questions:")
        
        example_questions = [
            "What is Section 420 of the IPC?",
            "Explain Article 21 of the Constitution",
            "What are the provisions for bail in CrPC?",
            "Define theft under Indian Penal Code",
            "What is the right to equality?"
        ]
        
        for i, question in enumerate(example_questions, 1):
            if st.button(f"{i}. {question}", key=f"example_{i}"):
                # Set the question in the input and trigger processing
                st.session_state.user_question = question
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
        '⚖️ LegalEase GPT - AI Legal Assistant for Indian Law<br>'
        '<em>Disclaimer: This is an AI assistant. Always consult legal professionals for official advice.</em>'
        '</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()