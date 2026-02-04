"""
Helper utilities for the Medical Chatbot.
"""

import os
import sys
from datetime import datetime
from typing import Optional


def print_colored(text: str, color: str = "white", bold: bool = False) -> None:
    """
    Print colored text to console.
    
    Args:
        text: Text to print
        color: Color name (red, green, yellow, blue, white, cyan, magenta)
        bold: Whether to use bold text
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "white": "\033[97m",
        "cyan": "\033[96m",
        "magenta": "\033[95m"
    }
    
    reset = "\033[0m"
    bold_code = "\033[1m" if bold else ""
    
    color_code = colors.get(color.lower(), colors["white"])
    
    print(f"{bold_code}{color_code}{text}{reset}")


def print_header(text: str) -> None:
    """
    Print a formatted header.
    
    Args:
        text: Header text
    """
    print("\n" + "="*60)
    print_colored(f"  {text}", "cyan", bold=True)
    print("="*60 + "\n")


def format_timestamp(timestamp: str, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp string.
    
    Args:
        timestamp: ISO format timestamp
        format: Output format
        
    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime(format)
    except ValueError:
        return timestamp


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing special characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace special characters
    filename = "".join(c for c in filename if c.isalnum() or c in ".-_")
    return filename


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate an API key.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False
    
    if len(api_key) < 10:
        return False
    
    if " " in api_key:
        return False
    
    return True


def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.
    
    Returns:
        True if all dependencies are available
    """
    required_modules = [
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
        ("langchain_groq", "langchain-groq"),
        ("faiss", "faiss-cpu"),
        ("pypdf", "pypdf"),
        ("pandas", "pandas"),
        ("dotenv", "python-dotenv")
    ]
    
    missing = []
    
    for module, package in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print_colored(f"❌ Missing dependencies: {', '.join(missing)}", "red")
        print_colored("Install with: pip install " + " ".join(missing), "yellow")
        return False
    
    print_colored("✅ All dependencies are installed", "green")
    return True


def clear_screen() -> None:
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_welcome() -> None:
    """Print welcome message."""
    print_colored("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          🩺  Medical Assistant Chatbot  🩺               ║
    ║                                                          ║
    ║    Powered by Groq API + FAISS Vector Database           ║
    ║    With persistent user memory and RAG pipeline          ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """, "cyan", bold=True)


def print_goodbye() -> None:
    """Print goodbye message."""
    print_colored("""
    👋 Thank you for using the Medical Assistant Chatbot!
    
    Remember: This chatbot is not a doctor. Always consult 
    healthcare professionals for medical advice.
    
    Stay healthy! 💙
    """, "cyan")


def print_help() -> None:
    """Print help message."""
    print("""
    Available Commands:
    - /help     : Show this help message
    - /history  : Show conversation history
    - /clear    : Clear conversation history
    - /sources  : Toggle source citations
    - /exit     : Exit the chatbot
    
    Tips:
    - Describe your symptoms in detail
    - Ask follow-up questions for more information
    - The bot remembers your conversation history
    """)


def get_user_input(prompt: str = "You: ") -> str:
    """
    Get user input from console.
    
    Args:
        prompt: Input prompt
        
    Returns:
        User input string
    """
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        return "/exit"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def retry_operation(func, max_retries: int = 3, delay: int = 1):
    """
    Retry an operation with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Initial delay in seconds
        
    Returns:
        Result of the function
    """
    import time
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            wait_time = delay * (2 ** attempt)
            print_colored(f"⚠️  Attempt {attempt + 1} failed. Retrying in {wait_time}s...", "yellow")
            time.sleep(wait_time)
