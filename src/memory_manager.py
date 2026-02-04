"""
User Memory Management for the Medical Chatbot.
Stores conversation history in JSON files per user.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from config import MEMORY_PATH, MAX_CONVERSATION_HISTORY


class MemoryManager:
    """Manages user conversation memory using JSON files."""
    
    def __init__(self, memory_path: str = MEMORY_PATH):
        """
        Initialize the Memory Manager.
        
        Args:
            memory_path: Path to store user memory files
        """
        self.memory_path = memory_path
        self._ensure_memory_directory()
    
    def _ensure_memory_directory(self) -> None:
        """Ensure the memory directory exists."""
        if not os.path.exists(self.memory_path):
            os.makedirs(self.memory_path, exist_ok=True)
    
    def _get_user_file_path(self, user_id: str) -> str:
        """Get the file path for a user's memory."""
        # Sanitize user_id for filename (remove special characters)
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
        return os.path.join(self.memory_path, f"{safe_user_id}.json")
    
    def get_user_memory(self, user_id: str) -> Dict:
        """
        Load a user's memory from file.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing user's conversation history and preferences
        """
        file_path = self._get_user_file_path(user_id)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # Return default memory if file is corrupted
                return self._create_default_memory(user_id)
        else:
            return self._create_default_memory(user_id)
    
    def _create_default_memory(self, user_id: str) -> Dict:
        """Create default memory structure for a new user."""
        return {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "conversations": [],
            "preferences": {
                "name": user_id,
                "known_conditions": [],
                "allergies": []
            }
        }
    
    def save_conversation(
        self, 
        user_id: str, 
        user_message: str, 
        bot_response: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save a conversation turn to user's memory.
        
        Args:
            user_id: Unique identifier for the user
            user_message: The user's message
            bot_response: The bot's response
            metadata: Optional additional metadata
        """
        memory = self.get_user_memory(user_id)
        
        conversation_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response,
            "metadata": metadata or {}
        }
        
        # Add new conversation
        memory["conversations"].append(conversation_entry)
        
        # Trim to max history (keep last N conversations)
        if len(memory["conversations"]) > MAX_CONVERSATION_HISTORY:
            memory["conversations"] = memory["conversations"][-MAX_CONVERSATION_HISTORY:]
        
        # Save to file
        file_path = self._get_user_file_path(user_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    
    def get_conversation_history(self, user_id: str) -> List[Dict]:
        """
        Get the conversation history for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            List of conversation entries
        """
        memory = self.get_user_memory(user_id)
        return memory.get("conversations", [])
    
    def get_formatted_history(self, user_id: str, max_turns: int = 5) -> str:
        """
        Get formatted conversation history for LLM context.
        
        Args:
            user_id: Unique identifier for the user
            max_turns: Maximum number of recent turns to include
            
        Returns:
            Formatted conversation history string
        """
        conversations = self.get_conversation_history(user_id)
        
        if not conversations:
            return ""
        
        # Get only the most recent turns
        recent_conversations = conversations[-max_turns:]
        
        formatted_history = []
        for conv in recent_conversations:
            formatted_history.append(f"User: {conv['user_message']}")
            formatted_history.append(f"Assistant: {conv['bot_response'][:200]}..." if len(conv['bot_response']) > 200 else f"Assistant: {conv['bot_response']}")
        
        return "\n".join(formatted_history)
    
    def update_user_preferences(self, user_id: str, preferences: Dict) -> None:
        """
        Update user preferences.
        
        Args:
            user_id: Unique identifier for the user
            preferences: Dictionary of preferences to update
        """
        memory = self.get_user_memory(user_id)
        memory["preferences"].update(preferences)
        
        file_path = self._get_user_file_path(user_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """
        Get user preferences.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary of user preferences
        """
        memory = self.get_user_memory(user_id)
        return memory.get("preferences", {})
    
    def clear_user_memory(self, user_id: str) -> None:
        """
        Clear all memory for a user.
        
        Args:
            user_id: Unique identifier for the user
        """
        memory = self._create_default_memory(user_id)
        file_path = self._get_user_file_path(user_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    
    def list_users(self) -> List[str]:
        """
        List all users with existing memory files.
        
        Returns:
            List of user IDs
        """
        if not os.path.exists(self.memory_path):
            return []
        
        users = []
        for filename in os.listdir(self.memory_path):
            if filename.endswith('.json'):
                users.append(filename.replace('.json', ''))
        
        return users
