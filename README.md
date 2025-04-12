# 🧠 RAGTutorial

A simple **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **ChromaDB**, and **Ollama** — built to answer questions based on your own documents.

---

## 📦 Features

- ✅ PDF loading and chunking
- ✅ Embedding generation using `nomic-embed-text` (via Ollama)
- ✅ Fast, local vector search using ChromaDB
- ✅ LLM responses powered by local models like `mistral`
- ✅ Fallbacks, per-document answers, and configurable prompts

---

## 🔧 Getting Started

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (optional) Check installed packages
pip list

# Deactivate when you're done
deactivate
```


## 🧠 Embeddings & Local LLM with Ollama

```bash
# Pull one of language models (used for generation)
ollama pull llama2
ollama pull mistral
ollama pull llama3.2

# Pull embedding model
ollama pull nomic-embed-text

# Start the Ollama server
ollama serve
```

## 🚀 Run the Pipeline

### 1. Load and embed documents
```bash
python load_data.py
```

### 2. Query your local RAG pipeline
```bash
python query.py "Your question here"
```

## ⚙️ Configuration
### Edit config.yaml to customize:
```yaml
embedding_model: "nomic-embed-text"
llm_model: "mistral"
multiple_responses: true
similarity_threshold: 0.85
```

## 🧩 How it Works

### 1. 📄 Loads and chunks your PDFs
### 2. 🔢 Converts chunks to vectors using nomic-embed-text
### 3. 🧠 Stores them in ChromaDB
### 4. 🧐 Embeds your question and finds similar chunks
### 5. 🤖 Feeds context into mistral or llama and generates an answer