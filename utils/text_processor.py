"""
✅ Splits documents into chunks using LangChain's RecursiveCharacterTextSplitter
✅ Configurable chunk size (default: 1000 characters)
✅ Overlap between chunks (default: 200 characters) - so context isn't lost
✅ Preserves metadata from original documents
✅ Statistics & preview to see what chunks look like
✅ Reusable - import in any project

Text Processor Module
Handles document chunking using LangChain text splitters
"""

from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import Settings


class TextProcessor:
    """Processes and chunks documents for embedding"""
    
    def __init__(
        self,
        separators: Optional[List[str]] = None
    ):
        print("Initializing TextProcessor...")
        self.settings = Settings.get_summary()
        self.chunk_size=int(self.settings['CHUNK_SIZE'])
        self.chunk_overlap=int(self.settings['CHUNK_OVERLAP'])
        
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=separators,
            is_separator_regex=False
        )
        print("TextProcessor initialized...")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            print("No documents to split")
            return []
        
        print(f"\n{'='*50}")
        print(f"Chunking {len(documents)} documents")
        print(f"Chunk size: {self.chunk_size} characters")
        print(f"Chunk overlap: {self.chunk_overlap} characters")
        print(f"{'='*50}\n")
        
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
        print(f"Average chunks per document: {len(chunks) / len(documents):.2f}\n")
        
        # self._display_chunk_details(chunks)
        
        return chunks
    
    
    def _display_chunk_details(self, chunks: List[Document]):
        print(f"\n{'='*50}")
        print(f"CHUNK DETAILS")
        print(f"{'='*50}\n")
        
        for i, chunk in enumerate(chunks, 1):
            chunk_size = len(chunk.page_content)
            source_file = chunk.metadata.get('file_name', 'Unknown')
            
            print(f"Chunk #{i}")
            print(f"├─ Size: {chunk_size} characters")
            print(f"├─ Source: {source_file}")
            print(f"└─ {'─'*45}\n")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split a single text string into chunks
        
        Args:
            text: Text string to split
            
        Returns:
            List of text chunks
        """
        chunks = self.text_splitter.split_text(text)
        print(f"✓ Split text into {len(chunks)} chunks")
        return chunks
    
    def get_chunk_info(self, chunks: List[Document]) -> dict:
        """
        Get information about chunks
        
        Args:
            chunks: List of chunked documents
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        info = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }
        
        return info
    
    def preview_chunks(self, chunks: List[Document], num_chunks: int = 3):
        """
        Preview first few chunks
        
        Args:
            chunks: List of chunked documents
            num_chunks: Number of chunks to preview
        """
        print(f"\n{'='*50}")
        print(f"Previewing first {min(num_chunks, len(chunks))} chunks")
        print(f"{'='*50}\n")
        
        for i, chunk in enumerate(chunks[:num_chunks], 1):
            print(f"Chunk {i}:")
            print(f"  Length: {len(chunk.page_content)} characters")
            print(f"  Source: {chunk.metadata.get('file_name', 'Unknown')}")
            print(f"  Preview: {chunk.page_content[:150]}...")
            print()