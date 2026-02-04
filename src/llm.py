"""
LLM Setup for the Medical Chatbot.
Uses Groq API for fast inference.
"""

from langchain_groq import ChatGroq
from config import GROQ_API_KEY, GROQ_MODEL, MODEL_TEMPERATURE


def load_llm(
    model: str = GROQ_MODEL,
    temperature: float = MODEL_TEMPERATURE,
    api_key: str = GROQ_API_KEY
) -> ChatGroq:
    """
    Load the Groq LLM.
    
    Args:
        model: Model name (e.g., llama3-70b-8192)
        temperature: Temperature for generation (0.0-1.0)
        api_key: Groq API key
        
    Returns:
        ChatGroq instance
    """
    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=api_key,
        max_tokens=1024,  # Limit response length
        timeout=60,  # 60 second timeout
        max_retries=3  # Retry failed requests
    )


def load_llm_with_settings(
    model: str = GROQ_MODEL,
    temperature: float = MODEL_TEMPERATURE,
    max_tokens: int = 1024,
    api_key: str = GROQ_API_KEY
) -> ChatGroq:
    """
    Load the Groq LLM with custom settings.
    
    Args:
        model: Model name
        temperature: Temperature for generation
        max_tokens: Maximum tokens in response
        api_key: Groq API key
        
    Returns:
        ChatGroq instance
    """
    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=api_key,
        max_tokens=max_tokens,
        timeout=60,
        max_retries=3
    )


# Available Groq models
GROQ_MODELS = {
    "llama3-70b-8192": {
        "description": "Llama 3 70B - Best for complex reasoning",
        "max_tokens": 8192,
        "context_window": 8192
    },
    "llama3-8b-8192": {
        "description": "Llama 3 8B - Fast and efficient",
        "max_tokens": 8192,
        "context_window": 8192
    },
    "mixtral-8x7b-32768": {
        "description": "Mixtral 8x7B - Large context window",
        "max_tokens": 32768,
        "context_window": 32768
    },
    "gemma-7b-it": {
        "description": "Gemma 7B - Instruction tuned",
        "max_tokens": 8192,
        "context_window": 8192
    }
}


def get_available_models() -> dict:
    """
    Get list of available Groq models.
    
    Returns:
        Dictionary of available models
    """
    return GROQ_MODELS


def check_api_connection(api_key: str = GROQ_API_KEY) -> bool:
    """
    Check if the Groq API is accessible.
    
    Args:
        api_key: Groq API key
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        llm = load_llm(api_key=api_key)
        # Simple test call
        llm.invoke("Hello, are you working?")
        return True
    except Exception:
        return False
