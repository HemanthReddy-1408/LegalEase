# LegalEase GPT: Contextual Legal Assistant with RAG and QLoRA Fine-Tuning

LegalEase GPT is a domain-specific assistant tailored for Indian law. It combines Retrieval-Augmented Generation (RAG) with a fine-tuned Falcon model to answer legal questions with high accuracy, using contextual document retrieval and persistent chat memory.

## Features

- **PDF Embedding**: Extracts and splits legal PDFs (IPC, Constitution, etc.) into structured chunks.
- **Vector Search**: Uses FAISS and sentence-transformers to store and retrieve document embeddings.
- **QLoRA Fine-Tuning**: Integrates an optionally fine-tuned `falcon-rw-1b` model using LoRA adapters.
- **Conversational Memory**: Maintains history-aware conversation via LangChain's buffer memory.
- **Streamlit Interface**: Lightweight web UI for interactive legal queries.
- **Session Persistence**: Stores chat memory and restores state using `sessions/*.json` files.
## Fine-Tuning (QLoRA)

You can fine-tune Falcon-RW-1B using QLoRA via the notebook:

`finetune/finetune_qlora.ipynb`

This notebook demonstrates how to:
- Load the base model
- Prepare legal QA datasets
- Apply QLoRA configuration using PEFT
- Train and save adapters to `finetune/qlora-legalease/`

The RAG pipeline automatically uses this adapter if present.

## Setup

### Requirements

- Python >= 3.10
- CUDA-compatible GPU (tested on CUDA 12.1 with 4GB VRAM)
- Conda or virtualenv

### Installation

```bash
conda create -n legalease310 python=3.10 -y
conda activate legalease310
pip install -r requirements.txt
```

Ensure `requirements.txt` includes:

- langchain
- langchain-community
- transformers
- sentence-transformers
- peft
- bitsandbytes
- faiss-cpu / faiss-gpu
- streamlit
- pymupdf
- pypdf2

## Usage

### 1. Embed PDFs

- Add your legal PDFs into the `data/` folder.
- Run embedding:

```bash
python embed_pdfs.py
```

- Output: FAISS index saved in `rag/vector_store/`.

### 2. Launch Chat App

```bash
streamlit run streamlit_app.py
```

- If `finetune/qlora-legalease/` exists, it loads the fine-tuned model.

### 3. Test CLI Output

```bash
python agent/rag_chain.py
```

- Runs a sample question and prints retrieved context + generated answer.

## Folder Structure

```
legalease-gpt/
├── agent/
│   ├── rag_chain.py
│   └── memory_config.py
├── data/                # Legal PDFs
├── rag/
│   └── vector_store/    # FAISS DB
├── finetune/
│   └── qlora-legalease/ # Fine-tuned LoRA (optional)
├── sessions/            # Memory storage
├── embed_pdfs.py
├── streamlit_app.py
├── README.md
```

## Notes

- Results are grounded only on the retrieved context.
- Supports chat memory persistence for ongoing sessions.
- Designed for legal Q&A, can be extended to other domains.

