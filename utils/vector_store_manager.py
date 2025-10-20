import os
import chromadb
from typing import List, Optional
from langchain.schema import Document
from langchain_chroma import Chroma
from utils.embedder import EmbeddingCreator
from config.settings import Settings


class VectorStoreManager:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self):
        print("Initializing VectorStoreManager...")
        self.settings = Settings.get_summary()
        self.collection_name = self.settings['COLLECTION_NAME']
        self.vector_store_path = self.settings['VECTOR_STORE_PATH']
        os.makedirs(self.vector_store_path, exist_ok=True)

        self.embedding_manager = EmbeddingCreator()
        self.vector_store = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embedding_manager.get_embeddings(),
            collection_name=self.collection_name
        )
        print("✓ VectorStoreManager initialized...")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store and persist"""
        print("Adding documents to vector store...")
        
        # Add documents to ChromaDB
        doc_ids = self.vector_store.add_documents(documents)
                
        print(f"✓ Added {len(doc_ids)} documents to vector store")
        return doc_ids
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search on the vector store"""
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with scores"""
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection"""
        client = chromadb.PersistentClient(path=self.vector_store_path)
        collection = client.get_collection(self.collection_name)
        
        return {
            "count": collection.count(),
            "collection_name": self.collection_name
        }
    
    def delete_collection(self):
        """Delete the entire collection"""
        client = chromadb.PersistentClient(path=self.vector_store_path)
        client.delete_collection(self.collection_name)
        print("Collection deleted")