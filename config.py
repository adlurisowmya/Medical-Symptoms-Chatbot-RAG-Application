"""
Configuration settings for the Medical Chatbot.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file. Please set it before running.")

# Model Configuration
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.3"))

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Vector Database Configuration
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_db")

# Memory Configuration
MEMORY_PATH = os.getenv("MEMORY_PATH", "memory/users")
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))

# Document Paths
DATA_PDF_PATH = os.getenv("DATA_PDF_PATH", "data/pdfs")
DATA_CSV_PATH = os.getenv("DATA_CSV_PATH", "data/csvs")
URLS_FILE_PATH = os.getenv("URLS_FILE_PATH", "data/urls.txt")

# Retrieval Configuration
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))

# Medical Disclaimer
MEDICAL_DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER**
I am an AI assistant, NOT a doctor. The information provided is for educational purposes only and should not be considered medical advice. Always consult with a qualified healthcare professional for diagnosis and treatment. If you are experiencing a medical emergency, please call your local emergency services immediately.
"""

# Severity Keywords (for detecting serious symptoms)
SEVERITY_KEYWORDS = [
    "chest pain", "difficulty breathing", "severe headache", "stroke",
    "heart attack", "unconscious", "high fever", "severe bleeding",
    "allergic reaction", "anaphylaxis", "sudden numbness", "slurred speech"
]
