# Medical Chatbot Implementation Plan

## Overview
Build a comprehensive medical chatbot with Groq API, vector database (FAISS), user memory, and support for PDF, CSV, and website URL knowledge sources.

## Requirements
- **LLM**: Groq API with llama3-70b-8192 model
- **Vector Database**: FAISS with sentence-transformers embeddings
- **User Memory**: JSON files per user (persistent, supports multiple users)
- **Document Support**: PDF, CSV, Website URLs (WebBaseLoader)
- **Tone**: Friendly, professional, no hallucinations
- **Safety**: Always include medical disclaimers

## Project Structure
```
Sowmya/
├── medical_chatbot.py          # Main entry point
├── config.py                   # Configuration settings
├── requirements.txt            # Updated dependencies
├── .env                        # Environment variables
├── data/
│   ├── pdfs/                  # Medical PDF documents
│   ├── csvs/                  # CSV data files
│   └── urls.txt               # Website URLs to ingest
├── vector_db/                  # FAISS vector database
├── memory/
│   └── users/                 # JSON files per user for memory
├── src/
│   ├── __init__.py
│   ├── llm.py                 # Groq LLM setup
│   ├── vector_store.py        # FAISS vector database
│   ├── document_loader.py     # PDF, CSV, URL loaders
│   ├── memory_manager.py      # User conversation memory
│   ├── rag_chain.py           # RAG pipeline with memory
│   └── prompts.py             # System prompts
└── utils/
    └── __init__.py            # Helper utilities
```

## Implementation Steps

### 1. Configuration (config.py)
- Environment variable loading
- Model settings (temperature=0.3 for reduced hallucinations)
- Paths configuration
- Medical disclaimer text

### 2. Document Loaders (src/document_loader.py)
- **PDF Loader**: Using PyPDFLoader for medical PDFs
- **CSV Loader**: Using CSVLoader for structured medical data
- **URL Loader**: Using WebBaseLoader for website ingestion
- Unified interface for all loaders

### 3. Vector Store (src/vector_store.py)
- Build FAISS index from documents
- Load existing FAISS index
- Search functionality with configurable k value
- Persistent storage

### 4. User Memory (src/memory_manager.py)
- Create user-specific JSON files in memory/users/
- Store conversation history
- Load previous context
- Support multiple users

### 5. LLM Setup (src/llm.py)
- Groq API integration
- llama3-70b-8192 model configuration
- Temperature setting for accuracy

### 6. System Prompts (src/prompts.py)
- **System Prompt**: Friendly, professional medical assistant
- **Hallucination Prevention**: Always cite sources
- **Disclaimer**: "I am not a doctor"
- **Escalation**: Serious symptoms → see a doctor

### 7. RAG Chain (src/rag_chain.py)
- RetrievalQA with conversation memory
- Context-aware responses
- Source citation
- Confidence scoring

### 8. Main Application (medical_chatbot.py)
- User login/identification
- CLI interface
- Conversation loop
- Memory persistence

## Hallucination Prevention Strategies
1. **Source Citation**: Always reference knowledge base documents
2. **Retrieval-Augmented Generation**: Ground responses in retrieved context
3. **Uncertainty Statements**: "Based on the medical documents..."
4. **Disclaimer**: Remind users to consult healthcare professionals
5. **Temperature Control**: Low temperature (0.3) for factual responses

## System Prompt Template
```
You are a friendly, empathetic medical assistant chatbot.

Your responsibilities:
- Analyze user symptoms carefully
- Predict possible conditions using trusted medical documents
- Suggest precautions and basic care
- Clearly state that you are NOT a doctor

Rules:
- Never provide definitive medical diagnoses
- Always cite your sources from the knowledge base
- If symptoms seem serious, advise seeing a doctor immediately
- Express uncertainty when information is unclear

Tone:
Calm, friendly, supportive, and easy to understand.

IMPORTANT: You must base your responses on the provided medical documents.
If you cannot find relevant information, say so clearly and suggest consulting a healthcare professional.
```

## Dependencies
```
langchain
langchain-community
langchain-groq
sentence-transformers
faiss-cpu
pypdf
pandas
python-dotenv
langchain-huggingface
beautifulsoup4
requests
```

## User Memory Format (JSON)
```json
{
  "user_id": "username",
  "conversations": [
    {
      "timestamp": "2024-01-01T10:00:00Z",
      "user_message": "I have a headache",
      "bot_response": "Based on your symptoms..."
    }
  ],
  "preferences": {
    "name": "John",
    "known_conditions": []
  }
}
```

## Next Steps
Once the plan is approved, switch to Code mode for implementation.
