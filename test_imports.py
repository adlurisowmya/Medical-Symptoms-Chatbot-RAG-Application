import os
import sys

try:
    print("Testing imports...")
    from dotenv import load_dotenv
    print("dotenv loaded")
    import pandas as pd
    print("pandas loaded")

    from langchain_community.document_loaders import PyPDFLoader, CSVLoader
    print("langchain_community.document_loaders loaded")

    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("langchain_community.embeddings loaded")

    from langchain_community.vectorstores import FAISS
    print("langchain_community.vectorstores loaded")

    from langchain_groq import ChatGroq
    print("langchain_groq loaded")

    from langchain.chains import RetrievalQA
    print("langchain.chains loaded")

    from langchain.prompts import PromptTemplate
    print("langchain.prompts loaded")

    print("All imports successful!")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
