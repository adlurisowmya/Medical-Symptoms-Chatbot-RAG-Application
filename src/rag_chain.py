"""
RAG Chain with Conversation Memory for the Medical Chatbot.
Combines retrieval, generation, and memory management.
"""

from typing import Dict, List, Optional, Any
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from src.memory_manager import MemoryManager
from src.vector_store import get_retriever, search_vector_db
from src.prompts import get_system_prompt
from config import RETRIEVER_K, MEDICAL_DISCLAIMER, SEVERITY_KEYWORDS


class MedicalRAGChain:
    """
    RAG Chain with conversation memory for medical queries.
    """
    
    def __init__(
        self,
        llm: ChatGroq,
        vector_db: FAISS,
        memory_manager: MemoryManager,
        retriever_k: int = RETRIEVER_K
    ):
        """
        Initialize the Medical RAG Chain.
        
        Args:
            llm: Groq LLM instance
            vector_db: FAISS vector database
            memory_manager: MemoryManager for conversation history
            retriever_k: Number of documents to retrieve
        """
        self.llm = llm
        self.vector_db = vector_db
        self.memory_manager = memory_manager
        self.retriever_k = retriever_k
        
        # Create the chain
        self._create_chain()
    
    def _create_chain(self):
        """Create the RAG chain."""
        
        # System prompt
        system_prompt = get_system_prompt()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "## Conversation History:\n{conversation_history}\n\nContext from medical knowledge base:\n{context}\n\nUser's question/symptoms:\n{question}")
        ])
        
        # Create chain
        self.chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "conversation_history": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_chat_history(self, user_id: str) -> str:
        """
        Format conversation history for the chain.
        
        Args:
            user_id: User identifier
            
        Returns:
            Formatted conversation history string
        """
        conversations = self.memory_manager.get_conversation_history(user_id)
        
        if not conversations:
            return ""
        
        formatted_history = []
        for conv in conversations[-10:]:  # Last 10 conversations
            formatted_history.append(f"User: {conv['user_message']}")
            formatted_history.append(f"Assistant: {conv['bot_response'][:200]}...")
        
        return "\n".join(formatted_history)
    
    def _check_severity(self, query: str) -> bool:
        """
        Check if symptoms indicate a severe condition.
        
        Args:
            query: User's query
            
        Returns:
            True if severe symptoms detected
        """
        query_lower = query.lower()
        for keyword in SEVERITY_KEYWORDS:
            if keyword in query_lower:
                return True
        return False
    
    def _get_severity_warning(self) -> str:
        """
        Get a severity warning message.
        
        Returns:
            Warning message string
        """
        return f"""
🚨 **IMPORTANT - SEEK MEDICAL ATTENTION**

Based on your symptoms, you should seek medical attention immediately. 
Please contact your local emergency services or go to the nearest emergency room.

{MEDICAL_DISCLAIMER}
"""
    
    def invoke(
        self,
        user_id: str,
        query: str,
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            user_id: User identifier
            query: User's question/symptoms
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        # Check for severe symptoms
        if self._check_severity(query):
            return {
                "answer": self._get_severity_warning(),
                "sources": [],
                "severity_detected": True
            }
        
        # Get relevant documents from vector DB
        docs = search_vector_db(self.vector_db, query, k=self.retriever_k)
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Get conversation history
        chat_history = self._format_chat_history(user_id)
        
        # Get sources
        sources = [{"content": doc.page_content[:200], "metadata": doc.metadata} for doc in docs]
        
        # Create input for chain
        chain_input = {
            "context": context,
            "question": query,
            "conversation_history": chat_history
        }
        
        # Invoke chain
        try:
            answer = self.chain.invoke(chain_input)
        except Exception as e:
            # Fallback if chain fails
            answer = f"I apologize, but I encountered an error processing your query: {str(e)}\n\n{MEDICAL_DISCLAIMER}"
        
        return {
            "answer": answer,
            "sources": sources,
            "severity_detected": False
        }
    
    def run(
        self,
        user_id: str,
        query: str,
        return_sources: bool = False
    ) -> str:
        """
        Run the chain with a user query.
        
        Args:
            user_id: User identifier
            query: User's question/symptoms
            return_sources: Whether to include sources in output
            
        Returns:
            Response string
        """
        result = self.invoke(user_id, query, return_sources)
        
        answer = result["answer"]
        
        if return_sources and result["sources"]:
            sources_text = "\n\n📚 **Sources:**\n"
            for i, source in enumerate(result["sources"], 1):
                sources_text += f"{i}. {source['content'][:100]}...\n"
            answer += sources_text
        
        return answer


def create_rag_chain(
    llm: ChatGroq,
    vector_db: FAISS,
    memory_manager: MemoryManager,
    retriever_k: int = RETRIEVER_K
) -> MedicalRAGChain:
    """
    Create a Medical RAG Chain.
    
    Args:
        llm: Groq LLM instance
        vector_db: FAISS vector database
        memory_manager: MemoryManager for conversation history
        retriever_k: Number of documents to retrieve
        
    Returns:
        MedicalRAGChain instance
    """
    return MedicalRAGChain(
        llm=llm,
        vector_db=vector_db,
        memory_manager=memory_manager,
        retriever_k=retriever_k
    )


class SimpleRAGChain:
    """
    Simpler RAG chain without advanced memory features.
    Useful for basic queries.
    """
    
    def __init__(
        self,
        llm: ChatGroq,
        vector_db: FAISS,
        retriever_k: int = RETRIEVER_K
    ):
        """
        Initialize the Simple RAG Chain.
        
        Args:
            llm: Groq LLM instance
            vector_db: FAISS vector database
            retriever_k: Number of documents to retrieve
        """
        self.llm = llm
        self.vector_db = vector_db
        self.retriever_k = retriever_k
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", get_system_prompt()),
            ("human", "Context from medical knowledge base:\n{context}\n\nUser's question/symptoms:\n{question}")
        ])
        
        # Create chain
        self.chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | llm
            | StrOutputParser()
        )
    
    def run(self, query: str, context: str = None) -> str:
        """
        Run the chain with a query.
        
        Args:
            query: User's question/symptoms
            context: Optional context override
            
        Returns:
            Response string
        """
        # Get context from vector DB if not provided
        if context is None and self.vector_db is not None:
            docs = self.vector_db.similarity_search(query, k=self.retriever_k)
            context = "\n\n".join([doc.page_content for doc in docs])
        
        if context is None:
            context = "No medical knowledge base available."
        
        return self.chain.invoke({"context": context, "question": query})
