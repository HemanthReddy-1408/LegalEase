from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from peft import PeftModel
from agent.memory_config import get_memory, save_memory
import torch
import os
import gc
import warnings
from contextlib import contextmanager
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
MODEL_NAME = "tiiuae/falcon-rw-1b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "rag/vector_store"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
RETRIEVAL_K = 4

@contextmanager
def gpu_memory_cleanup():
    """Context manager for GPU memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def initialize_models():
    """
    Initialize all models and components with proper error handling
    
    Returns:
        tuple: (retriever, llm, memory) or (None, None, None) if failed
    """
    try:
        print("🚀 Initializing LegalEase GPT...")
        
        # Load embedding model and vector DB
        print("📚 Loading vector database...")
        if not os.path.exists(VECTOR_STORE_PATH):
            raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}. Please run embed_pdf.py first.")
        
        embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        db = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embed_model, 
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        print("✅ Vector database loaded successfully")
        
        # Load tokenizer and model
        print(f"🤖 Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            cache_dir="./model_cache"
        )
        
        # Handle padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load base model with optimizations
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="./model_cache",
            trust_remote_code=True  # Added this for Falcon models
        )
        
        # Load fine-tuned LoRA adapter if available
        adapter_path = "finetune/qlora-legalease/checkpoint-1818"
        if os.path.exists(adapter_path):
            print("🔁 LoRA adapter found — loading fine-tuned model...")
            try:
                model = PeftModel.from_pretrained(base_model, adapter_path)
                print("✅ LoRA adapter loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load LoRA adapter: {e}")
                print("📌 Using base model instead")
                model = base_model
        else:
            print("⚠️ LoRA adapter not found. Using base model.")
            model = base_model
        
        # Create text generation pipeline
        print("⚙️ Setting up generation pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            trust_remote_code=True  # Added this
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ Model loaded successfully")
        
        # Initialize memory
        memory = get_memory()
        print("📋 Memory initialized")
        
        return retriever, llm, memory
        
    except Exception as e:
        print(f"❌ Failed to initialize models: {e}")
        return None, None, None

# Global variables for model components
_retriever = None
_llm = None
_memory = None
_qa_chain = None

def get_or_initialize_components():
    """Get initialized components or initialize them if not already done"""
    global _retriever, _llm, _memory, _qa_chain
    
    if _retriever is None or _llm is None or _memory is None:
        _retriever, _llm, _memory = initialize_models()
        
        if _retriever is None or _llm is None or _memory is None:
            raise RuntimeError("Failed to initialize model components")
        
        # Create QA chain with proper configuration
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a legal assistant specializing in Indian law.

Answer the user's question **ONLY using the provided context below**.
If the answer is not in the context, say: "I'm not sure based on the available information."

Be concise, accurate, and cite specific sections when mentioned.

----------------------
Context:
{context}

Question:
{question}

Answer:"""
        )
        
        # Alternative Fix: Create a simple QA chain without conversational memory
        from langchain.chains import RetrievalQA
        
        _qa_chain = RetrievalQA.from_chain_type(
            llm=_llm,
            chain_type="stuff",
            retriever=_retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
    
    return _retriever, _llm, _memory, _qa_chain

def generate_answer(question: str, session_id: str = None, debug: bool = False) -> dict:
    """
    Generate answer for legal question using RAG pipeline
    
    Args:
        question: User's legal question
        session_id: Optional session ID for memory persistence
        debug: Whether to include debug information
    
    Returns:
        dict: Response with answer, confidence, and optional debug info
    """
    if not question or not question.strip():
        return {
            "answer": "Please provide a valid legal question.",
            "confidence": 0.0,
            "sources": [],
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Get or initialize components
        retriever, llm, memory, qa_chain = get_or_initialize_components()
        
        if debug:
            print(f"\n🔍 Processing question: {question}")
        
        # Generate response with memory cleanup
        with gpu_memory_cleanup():
            # Get relevant documents first (for debugging and sources)
            docs = retriever.invoke(question)  # Updated to use invoke instead of deprecated method
            
            if debug and docs:
                print(f"\n📚 Retrieved {len(docs)} relevant documents:")
                for i, doc in enumerate(docs[:2]):  # Show first 2
                    section = doc.metadata.get("section", "Unknown")
                    print(f"[Doc {i+1}] Section {section}: {doc.page_content[:200]}...")
            
            # Generate answer using the simpler QA chain
            try:
                # Use RetrievalQA which doesn't have the memory issue
                result = qa_chain.invoke({"query": question})
                
                # Extract answer and source documents
                if isinstance(result, dict):
                    response = result.get("result", "").strip()
                    source_docs = result.get("source_documents", docs)
                else:
                    response = str(result).strip()
                    source_docs = docs
                
                # Manually save to memory
                if memory and response:
                    memory.save_context({"input": question}, {"output": response})
                    
            except Exception as chain_error:
                print(f"⚠️ Chain error: {chain_error}")
                # Fallback: Use retriever + LLM directly
                context = "\n\n".join([doc.page_content for doc in docs[:3]])
                prompt = f"""You are a legal assistant specializing in Indian law.

Answer the user's question **ONLY using the provided context below**.
If the answer is not in the context, say: "I'm not sure based on the available information."

Be concise, accurate, and cite specific sections when mentioned.

----------------------
Context:
{context}

Question:
{question}

Answer:"""
                
                response = llm.invoke(prompt).strip()
                source_docs = docs
                
                # Save to memory in fallback case too
                if memory and response:
                    memory.save_context({"input": question}, {"output": response})
        
        # Process and clean response
        if not response or len(response) < 5:
            response = "I'm not sure based on the available information."
            confidence = 0.2
        elif "I'm not sure" in response or "not in the context" in response:
            confidence = 0.3
        else:
            confidence = 0.8
        
        # Extract source information
        sources = []
        for doc in source_docs[:3]:  # Top 3 sources
            section = doc.metadata.get("section", "Unknown")
            title = doc.metadata.get("title", "")
            source_info = {
                "section": section,
                "title": title,
                "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            }
            sources.append(source_info)
        
        # Save memory if session_id provided
        if session_id:
            save_memory(memory, session_id)
        
        result_dict = {
            "answer": response,
            "confidence": confidence,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        
        if debug:
            result_dict["debug"] = {
                "retrieved_docs": len(docs),
                "question_length": len(question),
                "response_length": len(response)
            }
            print(f"✅ Generated response with confidence: {confidence}")
        
        return result_dict
        
    except Exception as e:
        error_msg = f"I encountered an error processing your question: {str(e)}"
        print(f"❌ Error in generate_answer: {e}")
        
        return {
            "answer": error_msg,
            "confidence": 0.0,
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def get_simple_answer(question: str) -> str:
    """
    Simplified version that returns just the answer string
    For backward compatibility with existing code
    
    Args:
        question: User's legal question
    
    Returns:
        str: Generated answer
    """
    result = generate_answer(question)
    return result["answer"]

def reset_conversation():
    """Reset the conversation memory"""
    global _memory
    if _memory:
        _memory.clear()
        print("🧹 Conversation memory cleared")

def get_model_info():
    """Get information about loaded models"""
    return {
        "model_name": MODEL_NAME,
        "embedding_model": EMBED_MODEL,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "retrieval_k": RETRIEVAL_K,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

# Test function
if __name__ == "__main__":
    print("🧪 Testing LegalEase GPT...")
    
    # Test initialization
    try:
        components = get_or_initialize_components()
        print("✅ Components initialized successfully")
        
        # Test question
        test_question = "What is Section 420 of the IPC?"
        print(f"\n🔍 Testing with question: {test_question}")
        
        result = generate_answer(test_question, debug=True)
        print(f"\n📝 Answer: {result['answer']}")
        print(f"🎯 Confidence: {result['confidence']}")
        print(f"📚 Sources: {len(result['sources'])}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n✅ Test completed!")