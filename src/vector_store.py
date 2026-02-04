"""
Vector Store Management for the Medical Chatbot.
Uses FAISS for efficient similarity search.
"""

import os
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import VECTOR_DB_PATH, EMBEDDING_MODEL


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    Get the embeddings model.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
        encode_kwargs={'normalize_embeddings': True}
    )


def build_vector_db(
    documents: List[Document],
    embeddings=None,
    vector_db_path: str = VECTOR_DB_PATH,
    allow_dangerous_deserialization: bool = True
) -> FAISS:
    """
    Build a FAISS vector database from documents.
    
    Args:
        documents: List of Document objects
        embeddings: Embeddings model (auto-created if None)
        vector_db_path: Path to save the vector database
        allow_dangerous_deserialization: Required for loading FAISS
        
    Returns:
        FAISS vector database instance
    """
    if not documents:
        print("⚠️  No documents provided to build vector database")
        return None
    
    print("\n" + "="*50)
    print("🔨 BUILDING VECTOR DATABASE")
    print("="*50)
    
    if embeddings is None:
        embeddings = get_embeddings()
    
    print(f"📊 Creating embeddings for {len(documents)} documents...")
    
    # Create FAISS index
    vector_db = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    # Ensure directory exists
    os.makedirs(vector_db_path, exist_ok=True)
    
    # Save locally
    vector_db.save_local(vector_db_path)
    print(f"✅ Vector database saved to: {vector_db_path}")
    print("="*50 + "\n")
    
    return vector_db


def load_vector_db(
    vector_db_path: str = VECTOR_DB_PATH,
    embeddings=None,
    allow_dangerous_deserialization: bool = True
) -> Optional[FAISS]:
    """
    Load an existing FAISS vector database.
    
    Args:
        vector_db_path: Path to the vector database
        embeddings: Embeddings model (auto-created if None)
        allow_dangerous_deserialization: Required for loading FAISS
        
    Returns:
        FAISS vector database instance, or None if not found
    """
    if not os.path.exists(vector_db_path):
        print(f"⚠️  Vector database not found at: {vector_db_path}")
        print("   Run with --rebuild flag to create a new one.")
        return None
    
    if embeddings is None:
        embeddings = get_embeddings()
    
    try:
        vector_db = FAISS.load_local(
            folder_path=vector_db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        print(f"✅ Vector database loaded from: {vector_db_path}")
        return vector_db
    except Exception as e:
        print(f"❌ Error loading vector database: {str(e)}")
        return None


def search_vector_db(
    vector_db: FAISS,
    query: str,
    k: int = 5
) -> List[Document]:
    """
    Search the vector database for similar documents.
    
    Args:
        vector_db: FAISS vector database instance
        query: Search query
        k: Number of results to return
        
    Returns:
        List of relevant Document objects
    """
    if vector_db is None:
        print("❌ Vector database not initialized")
        return []
    
    try:
        docs = vector_db.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"❌ Error searching vector database: {str(e)}")
        return []


def get_retriever(
    vector_db: FAISS,
    k: int = 5,
    search_type: str = "similarity"
):
    """
    Get a retriever from the vector database.
    
    Args:
        vector_db: FAISS vector database instance
        k: Number of documents to retrieve
        search_type: Type of search (similarity, mmr, similarity_score_threshold)
        
    Returns:
        Retriever instance
    """
    if vector_db is None:
        print("❌ Vector database not initialized")
        return None
    
    return vector_db.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )


def add_documents_to_vector_db(
    vector_db: FAISS,
    documents: List[Document],
    embeddings=None
) -> FAISS:
    """
    Add new documents to an existing vector database.
    
    Args:
        vector_db: Existing FAISS vector database
        documents: New Document objects to add
        embeddings: Embeddings model
        
    Returns:
        Updated FAISS vector database
    """
    if vector_db is None:
        return build_vector_db(documents, embeddings)
    
    if embeddings is None:
        embeddings = get_embeddings()
    
    # Add documents to existing index
    vector_db.add_documents(documents)
    
    # Save updated index
    vector_db.save_local(VECTOR_DB_PATH)
    
    print(f"✅ Added {len(documents)} documents to vector database")
    return vector_db


def delete_documents_from_vector_db(
    vector_db: FAISS,
    ids: List[str]
) -> FAISS:
    """
    Delete documents from the vector database by their IDs.
    
    Args:
        vector_db: FAISS vector database
        ids: List of document IDs to delete
        
    Returns:
        Updated FAISS vector database
    """
    if vector_db is None:
        print("❌ Vector database not initialized")
        return None
    
    vector_db.delete(ids)
    vector_db.save_local(VECTOR_DB_PATH)
    
    print(f"✅ Deleted {len(ids)} documents from vector database")
    return vector_db


def get_vector_db_stats(vector_db: FAISS) -> dict:
    """
    Get statistics about the vector database.
    
    Args:
        vector_db: FAISS vector database instance
        
    Returns:
        Dictionary with statistics
    """
    if vector_db is None:
        return {"error": "Vector database not initialized"}
    
    try:
        # Get the underlying FAISS index
        index = vector_db.index
        
        return {
            "total_documents": index.ntotal,
            "embedding_dim": index.d if hasattr(index, 'd') else "Unknown",
            "is_trained": index.is_trained if hasattr(index, 'is_trained') else True
        }
    except Exception as e:
        return {"error": str(e)}
