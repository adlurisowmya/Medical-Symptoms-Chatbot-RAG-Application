# 🏥 Medical Assistant Chatbot

A friendly, professional medical chatbot powered by **Groq API** with a **Retrieval-Augmented Generation (RAG)** pipeline. This chatbot provides symptom analysis, possible condition predictions, and health guidance based on medical documents.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- **🤖 AI-Powered Responses**: Powered by Groq's LLaMA model for accurate, contextual answers
- **📚 Knowledge Base**: Supports PDF, CSV, and URL document ingestion
- **🧠 Persistent Memory**: User-specific conversation history stored in JSON format
- **🔍 Semantic Search**: FAISS vector database for fast, relevant document retrieval
- **📝 Source Citation**: Responses include references to source documents
- **⚠️ Safety First**: Built-in medical disclaimers and severity detection
- **👥 Multi-User Support**: Individual user sessions with personalized memory

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key ([Get one here](https://console.groq.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Sowmya
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

4. **Run the chatbot**
   ```bash
   python medical_chatbot.py
   ```

### Command Line Options

```bash
python medical_chatbot.py [OPTIONS]

Options:
  -r, --rebuild     Rebuild the vector database
  -s, --sources     Show source citations in responses
  -t, --test        Run a simple test query
```

## 📁 Project Structure

```
Sowmya/
├── medical_chatbot.py          # Main entry point
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── data/
│   ├── pdfs/                  # Medical PDF documents
│   ├── csvs/                  # CSV data files (e.g., disease_symptoms.csv)
│   └── urls.txt               # Website URLs to ingest
├── vector_db/                  # FAISS vector database (persistent)
├── memory/
│   └── users/                 # User conversation memory (JSON)
├── src/
│   ├── llm.py                 # Groq LLM setup
│   ├── vector_store.py        # FAISS vector database operations
│   ├── document_loader.py     # PDF, CSV, URL document loaders
│   ├── memory_manager.py      # User conversation memory management
│   ├── rag_chain.py           # RAG pipeline implementation
│   └── prompts.py             # System prompts and templates
└── utils/
    └── helpers.py             # Utility functions
```

## 🛠️ How It Works

### 1. Document Ingestion
The chatbot loads medical documents from:
- **PDF files** in `data/pdfs/`
- **CSV files** in `data/csvs/`
- **URLs** listed in `data/urls.txt`

### 2. Vector Database
Documents are processed using:
- **Sentence Transformers** for embeddings (`all-MiniLM-L6-v2`)
- **FAISS** for efficient similarity search

### 3. RAG Pipeline
1. User query is embedded and compared against the knowledge base
2. Relevant documents are retrieved (top-k results)
3. LLM generates response using retrieved context
4. Response includes citations to source documents

### 4. User Memory
- Conversations are stored in `memory/users/{username}.json`
- Memory persists across sessions
- Supports multiple users

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/history` | Show recent conversation history |
| `/clear` | Clear conversation memory |
| `/sources` | Toggle source citations |
| `/exit` | Exit the chatbot |

## ⚠️ Important Disclaimer

This chatbot is **NOT a doctor** and should not be used for medical diagnosis or emergencies. The information provided is for **educational purposes only**. Always consult with a qualified healthcare professional for medical advice.

If you are experiencing a medical emergency, call your local emergency services immediately.

## 🔧 Configuration

All settings are managed in [`config.py`](config.py) and `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key | Required |
| `GROQ_MODEL` | LLM model name | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `VECTOR_DB_PATH` | Vector database location | `vector_db` |
| `RETRIEVER_K` | Number of documents to retrieve | `5` |
| `MODEL_TEMPERATURE` | LLM temperature (0-1) | `0.3` |

## 📦 Dependencies

```
langchain
langchain-community
langchain-groq
sentence-transformers
faiss-cpu
pypdf
pandas
python-dotenv
beautifulsoup4
requests
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Stay healthy! 🩺**
