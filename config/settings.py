import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings configuration"""
    
    # Gemini API Configuration
    GOOGLE_API_KEY =  os.getenv("GOOGLE_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
    
    CHUNK_SIZE: int = os.getenv("CHUNK_SIZE")
    CHUNK_OVERLAP: int = os.getenv("CHUNK_OVERLAP")
    TOP_K: int = os.getenv("TOP_K")
    TEMPERATURE: float = os.getenv("TEMPERATURE")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
        
    @classmethod
    def get_summary(cls):
        """Get configuration summary"""
        return {
            'EMBEDDING_MODEL': cls.EMBEDDING_MODEL,
            'GOOGLE_API_KEY': cls.GOOGLE_API_KEY,
            'LLM_MODEL': cls.LLM_MODEL,
            'CHUNK_SIZE': cls.CHUNK_SIZE,
            'CHUNK_OVERLAP': cls.CHUNK_OVERLAP,
            'TOP_K': cls.TOP_K,
            'TEMPERATURE': cls.TEMPERATURE,
            'COLLECTION_NAME': cls.COLLECTION_NAME,
            'VECTOR_STORE_PATH': cls.VECTOR_STORE_PATH
        }