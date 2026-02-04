#!/usr/bin/env python3
"""
Medical Assistant Chatbot
=========================
A friendly, professional medical chatbot powered by Groq API with RAG pipeline.
Features:
- Vector database for medical knowledge (FAISS)
- User memory with JSON files
- PDF, CSV, and URL document support
- Conversation history awareness
- Hallucination prevention with source citation
"""

import os
import sys
import argparse
from pathlib import Path

# Set USER_AGENT for langchain
os.environ["USER_AGENT"] = "MedicalChatbot/1.0"

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GROQ_API_KEY,
    VECTOR_DB_PATH,
    MEDICAL_DISCLAIMER
)
from src.llm import load_llm
from src.vector_store import (
    build_vector_db,
    load_vector_db,
    get_embeddings
)
from src.document_loader import load_all_documents
from src.memory_manager import MemoryManager
from src.rag_chain import create_rag_chain
from utils.helpers import (
    print_welcome,
    print_goodbye,
    print_help,
    print_colored,
    get_user_input,
    check_dependencies
)


class MedicalChatbot:
    """
    Main Medical Chatbot class.
    Handles initialization, conversation loop, and user interaction.
    """
    
    def __init__(
        self,
        rebuild_db: bool = False,
        show_sources: bool = False
    ):
        """
        Initialize the Medical Chatbot.
        
        Args:
            rebuild_db: Whether to rebuild the vector database
            show_sources: Whether to show source citations
        """
        self.show_sources = show_sources
        self.user_id = None
        self.rag_chain = None
        self.memory_manager = None
        
        # Initialize components
        self._initialize_components(rebuild_db)
    
    def _initialize_components(self, rebuild_db: bool):
        """Initialize all chatbot components."""
        print_colored("\n🔧 Initializing Medical Chatbot...", "cyan")
        
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Initialize LLM
        print_colored("📦 Loading Groq LLM...", "cyan")
        self.llm = load_llm()
        print_colored("   ✅ LLM loaded successfully", "green")
        
        # Initialize Memory Manager
        print_colored("📦 Initializing Memory Manager...", "cyan")
        self.memory_manager = MemoryManager()
        print_colored("   ✅ Memory Manager initialized", "green")
        
        # Initialize Vector Database
        print_colored("📦 Loading Vector Database...", "cyan")
        self.vector_db = self._load_or_build_vector_db(rebuild_db)
        
        if self.vector_db is None:
            print_colored("❌ Failed to initialize vector database", "red")
            sys.exit(1)
        
        # Initialize RAG Chain
        print_colored("📦 Creating RAG Chain...", "cyan")
        self.rag_chain = create_rag_chain(
            llm=self.llm,
            vector_db=self.vector_db,
            memory_manager=self.memory_manager
        )
        print_colored("   ✅ RAG Chain ready", "green")
        
        print_colored("✨ Medical Chatbot initialized successfully!\n", "green")
    
    def _load_or_build_vector_db(self, rebuild: bool) -> any:
        """Load or build the vector database."""
        # Check if FAISS index files exist
        index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
        pkl_file = os.path.join(VECTOR_DB_PATH, "index.pkl")
        
        if rebuild or not (os.path.exists(index_file) and os.path.exists(pkl_file)):
            if rebuild:
                print_colored("🔄 Rebuilding vector database...", "yellow")
            else:
                print_colored("📚 No existing database found. Building new one...", "yellow")
            
            # Load documents
            documents = load_all_documents()
            
            if not documents:
                print_colored("⚠️  No documents found. Creating empty database...", "yellow")
                embeddings = get_embeddings()
                from langchain_core.documents import Document
                documents = [Document(page_content="No medical documents loaded.", metadata={"source": "system"})]
            
            # Build vector database
            return build_vector_db(documents)
        else:
            return load_vector_db()
    
    def _authenticate_user(self) -> str:
        """Authenticate or create a user."""
        print_colored("\n👤 User Authentication", "cyan", bold=True)
        print("Enter your username (or type 'new' to create a new user):")
        
        while True:
            username = get_user_input("Username: ").strip()
            
            if not username:
                print_colored("❌ Username cannot be empty", "red")
                continue
            
            if username.lower() == 'new':
                print("Enter a new username:")
                username = get_user_input("New username: ").strip()
            
            # Check if user exists
            existing_users = self.memory_manager.list_users()
            
            if username in existing_users:
                print_colored(f"✅ Welcome back, {username}!", "green")
            else:
                print_colored(f"✅ Created new user: {username}", "green")
            
            return username
    
    def _process_command(self, command: str) -> bool:
        """
        Process a chatbot command.
        
        Args:
            command: Command string
            
        Returns:
            True if command was processed, False to continue
        """
        cmd = command.lower().strip()
        
        if cmd == "/help":
            print_help()
            return True
        
        elif cmd == "/history":
            history = self.memory_manager.get_conversation_history(self.user_id)
            print_colored(f"\n📜 Conversation History ({len(history)} messages):", "cyan")
            for i, conv in enumerate(history[-5:], 1):
                print(f"{i}. User: {conv['user_message'][:50]}...")
            return True
        
        elif cmd == "/clear":
            self.memory_manager.clear_user_memory(self.user_id)
            print_colored("✅ Conversation history cleared", "green")
            return True
        
        elif cmd == "/sources":
            self.show_sources = not self.show_sources
            status = "enabled" if self.show_sources else "disabled"
            print_colored(f"✅ Source citations {status}", "green")
            return True
        
        elif cmd == "/exit":
            return False
        
        else:
            return False  # Not a command, treat as regular input
    
    def _handle_user_input(self, user_input: str) -> str:
        """
        Process user input and generate response.
        
        Args:
            user_input: User's message
            
        Returns:
            Bot's response
        """
        try:
            # Get response from RAG chain
            result = self.rag_chain.invoke(
                user_id=self.user_id,
                query=user_input,
                return_sources=self.show_sources
            )
            
            answer = result["answer"]
            
            # Save to memory
            self.memory_manager.save_conversation(
                user_id=self.user_id,
                user_message=user_input,
                bot_response=answer
            )
            
            return answer
            
        except Exception as e:
            print_colored(f"❌ Error processing query: {str(e)}", "red")
            return f"I apologize, but I encountered an error. Please try again.\n\n{MEDICAL_DISCLAIMER}"
    
    def run(self):
        """Run the main conversation loop."""
        # Authenticate user
        self.user_id = self._authenticate_user()
        
        # Print welcome message
        print_welcome()
        print(f"Logged in as: {self.user_id}")
        print(f"Memory file: memory/users/{self.user_id}.json")
        print("\nType '/help' for available commands or describe your symptoms.")
        
        # Main conversation loop
        while True:
            try:
                user_input = get_user_input("\n🧑 You: ")
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.startswith('/'):
                    if user_input == '/exit':
                        break
                    elif not self._process_command(user_input):
                        print_colored("❌ Unknown command. Type '/help' for options.", "yellow")
                    continue
                
                # Process user input
                response = self._handle_user_input(user_input)
                
                # Print bot response
                print_colored("\n🤖 Medical Assistant:", "cyan", bold=True)
                print(response)
                
            except KeyboardInterrupt:
                print("\n")
                break
        
        # Goodbye
        print_goodbye()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Medical Assistant Chatbot with Groq API and RAG pipeline"
    )
    
    parser.add_argument(
        "--rebuild",
        "-r",
        action="store_true",
        help="Rebuild the vector database"
    )
    
    parser.add_argument(
        "--sources",
        "-s",
        action="store_true",
        help="Show source citations in responses"
    )
    
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run a simple test query"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check API key
    if not GROQ_API_KEY:
        print_colored("❌ GROQ_API_KEY not found in .env file", "red")
        print("Please set your Groq API key in the .env file:")
        print("  GROQ_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Create and run chatbot
    chatbot = MedicalChatbot(
        rebuild_db=args.rebuild,
        show_sources=args.sources
    )
    
    if args.test:
        # Run a simple test
        print_colored("\n🧪 Running test query...", "yellow")
        response = chatbot._handle_user_input("What are common symptoms of the flu?")
        print_colored("\n🤖 Response:", "cyan")
        print(response)
    else:
        # Run main conversation loop
        chatbot.run()


if __name__ == "__main__":
    main()
