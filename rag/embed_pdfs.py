import os
import re
import json
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import torch

# Configuration
PDF_DIR = "data"
VECTOR_STORE_DIR = "rag/vector_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BATCH_SIZE = 50

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using PyMuPDF (better) with PyPDF2 fallback
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        str: Extracted text
    """
    try:
        # Try PyMuPDF first (better text extraction)
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        
        doc.close()
        extracted_text = "\n\n".join(text_parts)
        
        if extracted_text.strip():
            print(f"✅ Extracted {len(extracted_text)} characters using PyMuPDF")
            return extracted_text
        else:
            raise Exception("No text extracted with PyMuPDF")
            
    except Exception as e:
        print(f"⚠️ PyMuPDF failed: {e}, trying PyPDF2...")
        
        # Fallback to PyPDF2
        try:
            reader = PdfReader(pdf_path)
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(text)
            
            extracted_text = "\n\n".join(text_parts)
            print(f"✅ Extracted {len(extracted_text)} characters using PyPDF2")
            return extracted_text
            
        except Exception as e2:
            print(f"❌ Both extraction methods failed: {e2}")
            return ""

def clean_text(text):
    """
    Clean and normalize extracted text
    
    Args:
        text: Raw extracted text
    
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Fix common PDF extraction issues
    text = text.replace('\x00', '')  # Remove null characters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
    
    # Normalize section headers
    text = re.sub(r'SECTION\s+(\d+)', r'Section \1', text, flags=re.IGNORECASE)
    text = re.sub(r'Article\s+(\d+)', r'Article \1', text, flags=re.IGNORECASE)
    
    return text.strip()

def enhanced_chunk_by_section(raw_text, filename=""):
    """
    Enhanced section chunking with better pattern matching and metadata
    
    Args:
        raw_text: Raw text from PDF
        filename: Source filename for metadata
    
    Returns:
        list: List of Document objects
    """
    chunks = []
    text = clean_text(raw_text)
    
    if not text:
        return chunks
    
    # Multiple patterns for different legal document types
    patterns = [
        # IPC Sections
        r'(?=Section\s+(\d+[A-Z]?)\.?\s*[-–—]?\s*(.+?)(?=\n|$))',
        # Articles (Constitution)
        r'(?=Article\s+(\d+[A-Z]?)\.?\s*[-–—]?\s*(.+?)(?=\n|$))',
        # CrPC Sections  
        r'(?=(\d+)\.\s*(.+?)(?=\n|$))',
        # General numbered sections
        r'(?=(\d+[A-Z]*)\.\s+(.+?)(?=\n|$))'
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )
    
    sections_found = False
    
    # Try each pattern
    for pattern_idx, pattern in enumerate(patterns):
        try:
            sections = re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            
            if len(sections) > 3:  # Found meaningful sections
                sections_found = True
                print(f"📋 Using pattern {pattern_idx + 1}, found {len(sections)//3} sections")
                
                for i in range(0, len(sections), 3):
                    if i + 2 < len(sections):
                        section_content = sections[i] if i < len(sections) else ""
                        section_num = sections[i + 1] if i + 1 < len(sections) else ""
                        section_title = sections[i + 2] if i + 2 < len(sections) else ""
                        
                        if not section_num:
                            continue
                        
                        # Clean and format section
                        section_title = section_title.strip()[:100]  # Limit title length
                        
                        # Determine document type
                        doc_type = "IPC" if "Section" in pattern else "Constitution" if "Article" in pattern else "Legal"
                        
                        if "Section" in pattern:
                            full_section = f"Section {section_num} of IPC - {section_title}\n{section_content}".strip()
                        elif "Article" in pattern:
                            full_section = f"Article {section_num} of Constitution - {section_title}\n{section_content}".strip()
                        else:
                            full_section = f"{section_num}. {section_title}\n{section_content}".strip()
                        
                        # Handle large sections by splitting
                        if len(full_section) > CHUNK_SIZE:
                            sub_chunks = text_splitter.split_text(full_section)
                            for j, sub_chunk in enumerate(sub_chunks):
                                chunks.append(Document(
                                    page_content=sub_chunk,
                                    metadata={
                                        "section": section_num,
                                        "title": section_title,
                                        "chunk_id": f"{section_num}_{j}",
                                        "source": doc_type,
                                        "filename": filename,
                                        "chunk_index": j,
                                        "total_chunks": len(sub_chunks)
                                    }
                                ))
                        else:
                            chunks.append(Document(
                                page_content=full_section,
                                metadata={
                                    "section": section_num,
                                    "title": section_title,
                                    "source": doc_type,
                                    "filename": filename,
                                    "chunk_index": 0,
                                    "total_chunks": 1
                                }
                            ))
                break  # Stop after first successful pattern
                
        except Exception as e:
            print(f"⚠️ Pattern {pattern_idx + 1} failed: {e}")
            continue
    
    # Fallback: If no sections found, use general text splitting
    if not sections_found:
        print("📄 No sections detected, using general text splitting")
        text_chunks = text_splitter.split_text(text)
        
        for i, chunk in enumerate(text_chunks):
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    "section": f"chunk_{i}",
                    "title": f"Text chunk {i + 1}",
                    "source": "General",
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                }
            ))
    
    return chunks

def create_embeddings_in_batches(all_chunks, embedder):
    """
    Create FAISS database in batches to handle memory limitations
    
    Args:
        all_chunks: List of Document objects
        embedder: Embedding model
    
    Returns:
        FAISS: Vector database
    """
    if not all_chunks:
        raise ValueError("No chunks to embed")
    
    print(f"🧠 Creating embeddings for {len(all_chunks)} chunks...")
    
    if len(all_chunks) <= BATCH_SIZE:
        # Small dataset, process all at once
        return FAISS.from_documents(all_chunks, embedder)
    
    # Large dataset, process in batches
    print(f"📦 Processing in batches of {BATCH_SIZE}...")
    
    # Create initial database with first batch
    first_batch = all_chunks[:BATCH_SIZE]
    db = FAISS.from_documents(first_batch, embedder)
    print(f"✅ Processed batch 1/{(len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    # Add remaining batches
    for i in range(BATCH_SIZE, len(all_chunks), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(all_chunks))
        batch = all_chunks[i:batch_end]
        
        # Create temporary database for this batch
        temp_db = FAISS.from_documents(batch, embedder)
        
        # Merge with main database
        db.merge_from(temp_db)
        
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"✅ Processed batch {batch_num}/{total_batches}")
    
    return db

def validate_pdf_files(pdf_dir):
    """
    Validate PDF directory and files
    
    Args:
        pdf_dir: Directory containing PDF files
    
    Returns:
        list: Valid PDF file paths
    """
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory '{pdf_dir}' not found!")
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'")
    
    valid_files = []
    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        if os.path.getsize(pdf_path) > 0:  # Check if file is not empty
            valid_files.append(filename)
        else:
            print(f"⚠️ Skipping empty file: {filename}")
    
    if not valid_files:
        raise ValueError("No valid PDF files found")
    
    return valid_files

def main():
    """
    Main function to process PDFs and create vector database
    """
    try:
        print("🚀 Starting PDF embedding process...")
        print(f"📁 PDF Directory: {PDF_DIR}")
        print(f"💾 Vector Store: {VECTOR_STORE_DIR}")
        print(f"🧠 Embedding Model: {EMBED_MODEL}")
        
        # Validate PDF files
        pdf_files = validate_pdf_files(PDF_DIR)
        print(f"📚 Found {len(pdf_files)} valid PDF files: {pdf_files}")
        
        # Process each PDF
        all_chunks = []
        processing_stats = {"total_files": 0, "successful_files": 0, "total_chunks": 0}
        
        for i, filename in enumerate(pdf_files, 1):
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"\n📄 Processing {i}/{len(pdf_files)}: {filename}")
            
            try:
                # Extract text
                raw_text = extract_text_from_pdf(pdf_path)
                if not raw_text.strip():
                    print(f"⚠️ No text extracted from {filename}")
                    continue
                
                # Create chunks
                section_chunks = enhanced_chunk_by_section(raw_text, filename)
                if section_chunks:
                    print(f"✅ Created {len(section_chunks)} chunks from {filename}")
                    all_chunks.extend(section_chunks)
                    processing_stats["successful_files"] += 1
                    processing_stats["total_chunks"] += len(section_chunks)
                else:
                    print(f"⚠️ No chunks created from {filename}")
                
                processing_stats["total_files"] += 1
                
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")
                continue
        
        # Validate results
        if not all_chunks:
            raise ValueError("No content was successfully processed!")
        
        print(f"\n📊 Processing Summary:")
        print(f"   • Files processed: {processing_stats['successful_files']}/{processing_stats['total_files']}")
        print(f"   • Total chunks: {processing_stats['total_chunks']}")
        
        # Initialize embeddings
        print(f"\n🧠 Initializing embedding model: {EMBED_MODEL}")
        embedder = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        )
        
        # Create vector database
        db = create_embeddings_in_batches(all_chunks, embedder)
        
        # Save vector database
        print(f"\n💾 Saving vector database...")
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        db.save_local(VECTOR_STORE_DIR)
        
        # Save metadata
        metadata = {
            "total_chunks": len(all_chunks),
            "embedding_model": EMBED_MODEL,
            "created_at": datetime.now().isoformat(),
            "source_files": pdf_files,
            "processing_stats": processing_stats,
            "config": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "batch_size": BATCH_SIZE
            }
        }
        
        metadata_path = os.path.join(VECTOR_STORE_DIR, "metadata.json")
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Vector database created successfully!")
        print(f"   • Location: {VECTOR_STORE_DIR}")
        print(f"   • Total chunks: {len(all_chunks)}")
        print(f"   • Metadata saved: {metadata_path}")
        
        # Test the database
        print(f"\n🧪 Testing vector database...")
        test_query = "Section 420"
        retriever = db.as_retriever(search_kwargs={"k": 2})
        test_docs = retriever.get_relevant_documents(test_query)
        
        if test_docs:
            print(f"✅ Database test successful! Found {len(test_docs)} relevant documents for '{test_query}'")
            print(f"   Sample result: {test_docs[0].page_content[:100]}...")
        else:
            print("⚠️ Database test failed - no documents retrieved")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return False

def get_database_info():
    """
    Get information about existing vector database
    
    Returns:
        dict: Database information or None if not found
    """
    try:
        metadata_path = os.path.join(VECTOR_STORE_DIR, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding='utf-8') as f:
                return json.load(f)
        else:
            return None
    except Exception as e:
        print(f"❌ Error reading database info: {e}")
        return None

def check_database_exists():
    """
    Check if vector database exists and is valid
    
    Returns:
        bool: True if database exists and is valid
    """
    try:
        if not os.path.exists(VECTOR_STORE_DIR):
            return False
        
        # Check for required FAISS files
        required_files = ["index.faiss", "index.pkl"]
        for file in required_files:
            if not os.path.exists(os.path.join(VECTOR_STORE_DIR, file)):
                return False
        
        return True
    except Exception:
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🏛️  LegalEase GPT - PDF Embedding System")
    print("=" * 60)
    
    # Check if database already exists
    if check_database_exists():
        info = get_database_info()
        if info:
            print(f"📊 Existing database found:")
            print(f"   • Created: {info.get('created_at', 'Unknown')}")
            print(f"   • Chunks: {info.get('total_chunks', 'Unknown')}")
            print(f"   • Files: {len(info.get('source_files', []))}")
        
        response = input("\n🔄 Database exists. Recreate? (y/N): ").strip().lower()
        if response != 'y':
            print("✋ Keeping existing database. Exiting.")
            exit(0)
    
    # Run the main process
    success = main()
    
    if success:
        print(f"\n🎉 Process completed successfully!")
        print(f"💡 You can now run: streamlit run streamlit_app.py")
    else:
        print(f"\n💥 Process failed. Please check the errors above.")
    
    print("=" * 60)