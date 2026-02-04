"""
Document Loaders for the Medical Chatbot.
Supports loading from PDF, CSV, and Website URLs.
"""

import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain_core.documents import Document
from config import DATA_PDF_PATH, DATA_CSV_PATH, URLS_FILE_PATH


def load_pdf_documents(pdf_path: str = DATA_PDF_PATH) -> List[Document]:
    """
    Load all PDF documents from the specified directory.
    
    Args:
        pdf_path: Path to directory containing PDF files
        
    Returns:
        List of Document objects from PDFs
    """
    documents = []
    
    if not os.path.exists(pdf_path):
        print(f"⚠️  PDF directory not found: {pdf_path}")
        return documents
    
    for filename in os.listdir(pdf_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(pdf_path, filename)
            try:
                print(f"📄 Loading PDF: {filename}")
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                print(f"   ✅ Loaded {len(docs)} pages from {filename}")
                documents.extend(docs)
            except Exception as e:
                print(f"   ❌ Error loading {filename}: {str(e)}")
    
    return documents


def load_csv_documents(csv_path: str = DATA_CSV_PATH) -> List[Document]:
    """
    Load all CSV documents from the specified directory.
    
    Args:
        csv_path: Path to directory containing CSV files
        
    Returns:
        List of Document objects from CSVs
    """
    documents = []
    
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV directory not found: {csv_path}")
        return documents
    
    for filename in os.listdir(csv_path):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(csv_path, filename)
            try:
                print(f"📊 Loading CSV: {filename}")
                loader = CSVLoader(file_path)
                docs = loader.load()
                print(f"   ✅ Loaded {len(docs)} rows from {filename}")
                documents.extend(docs)
            except Exception as e:
                print(f"   ❌ Error loading {filename}: {str(e)}")
    
    return documents


def load_url_documents(urls_file_path: str = URLS_FILE_PATH) -> List[Document]:
    """
    Load documents from URLs listed in a text file.
    
    Args:
        urls_file_path: Path to file containing URLs (one per line)
        
    Returns:
        List of Document objects from URLs
    """
    documents = []
    
    if not os.path.exists(urls_file_path):
        print(f"⚠️  URLs file not found: {urls_file_path}")
        return documents
    
    try:
        with open(urls_file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        if not urls:
            print("⚠️  No URLs found in URLs file")
            return documents
        
        print(f"🌐 Loading {len(urls)} URL(s)...")
        
        # Load URLs in batches to avoid timeouts
        batch_size = 5
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            try:
                loader = WebBaseLoader(web_paths=batch)
                docs = loader.load()
                print(f"   ✅ Loaded {len(docs)} document(s) from URLs")
                documents.extend(docs)
            except Exception as e:
                print(f"   ⚠️  Error loading batch starting at URL {batch[0]}: {str(e)}")
                # Try loading individually
                for url in batch:
                    try:
                        loader = WebBaseLoader(web_paths=[url])
                        docs = loader.load()
                        print(f"   ✅ Loaded from: {url}")
                        documents.extend(docs)
                    except Exception as e2:
                        print(f"   ❌ Error loading {url}: {str(e2)}")
    
    except Exception as e:
        print(f"❌ Error reading URLs file: {str(e)}")
    
    return documents


def load_all_documents(
    include_pdfs: bool = True,
    include_csvs: bool = True,
    include_urls: bool = True,
    pdf_path: str = DATA_PDF_PATH,
    csv_path: str = DATA_CSV_PATH,
    urls_file_path: str = URLS_FILE_PATH
) -> List[Document]:
    """
    Load all documents from configured sources.
    
    Args:
        include_pdfs: Whether to load PDF documents
        include_csvs: Whether to load CSV documents
        include_urls: Whether to load URL documents
        pdf_path: Path to PDF directory
        csv_path: Path to CSV directory
        urls_file_path: Path to URLs file
        
    Returns:
        Combined list of all Document objects
    """
    all_documents = []
    
    print("\n" + "="*50)
    print("📚 LOADING MEDICAL DOCUMENTS")
    print("="*50)
    
    if include_pdfs:
        pdf_docs = load_pdf_documents(pdf_path)
        print(f"   PDFs: {len(pdf_docs)} documents loaded")
        all_documents.extend(pdf_docs)
    
    if include_csvs:
        csv_docs = load_csv_documents(csv_path)
        print(f"   CSVs: {len(csv_docs)} documents loaded")
        all_documents.extend(csv_docs)
    
    if include_urls:
        url_docs = load_url_documents(urls_file_path)
        print(f"   URLs: {len(url_docs)} documents loaded")
        all_documents.extend(url_docs)
    
    print("="*50)
    print(f"✅ Total: {len(all_documents)} documents loaded")
    print("="*50 + "\n")
    
    return all_documents


def load_single_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_single_csv(file_path: str) -> List[Document]:
    """
    Load a single CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of Document objects
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    
    loader = CSVLoader(file_path)
    return loader.load()


def load_single_url(url: str) -> List[Document]:
    """
    Load content from a single URL.
    
    Args:
        url: The URL to load
        
    Returns:
        List of Document objects
    """
    loader = WebBaseLoader(web_paths=[url])
    return loader.load()
