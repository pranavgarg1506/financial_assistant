from utils.embedder import EmbeddingCreator
from utils.document_loader import DocumentLoader
from utils.text_processor import TextProcessor
from utils.vector_store_manager import VectorStoreManager
from utils.query_engine import QueryEngine
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"



def main():
    
    ## Load documents from directory
    loader = DocumentLoader(directory_path="data/inputs")
    documents = loader.load_all_documents()
    
    processor = TextProcessor()
    chunks = processor.split_documents(documents)
        
    # Step 2: Create embeddings and store in vector database
    print("\n🔮 Creating embeddings and storing in vector database...")
    vector_store = VectorStoreManager()
    
    # Add documents to vector store
    doc_ids = vector_store.add_documents(documents)
    
    # Step 3: Display collection statistics
    stats = vector_store.get_collection_stats()
    print(f"\n✅ Vector store created successfully!")
    print(f"   - Total documents in collection: {stats['count']}")
    print(f"   - Collection name: {stats['collection_name']}")
    
    query_engine = QueryEngine(vector_store)

    # Simple query
    answer, sources = query_engine.query("Tell me the key financial metrics of TCS and how it is different from last quarters?")
    print(f"Answer: {answer}")        



if __name__ == "__main__":
    main()