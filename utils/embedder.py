from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

from config.settings import Settings


class EmbeddingCreator:
    def __init__(self):
        
        print("Initializing Embedding Creator...")
        
        self.settings = Settings.get_summary()
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=self.settings['EMBEDDING_MODEL'],
            google_api_key=self.settings['GOOGLE_API_KEY']
        )
        
        print("Embedding Creator initialized...")
        
    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Get the embeddings instance"""
        return self.embedding_model
        
        
    def generate_embeddings_for_documents(self, documents: List[Document]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        texts = [doc.page_content for doc in documents]
        return self.embedding_model.embed_documents(texts)
    
    def generate_embedding_for_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        embeddings = self.embedding_model.embed_query(query)
        print("create a embedding of length:", len(embeddings))
        return embeddings