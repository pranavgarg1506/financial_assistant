"""
Use LangChain document loaders for each file type
Load PDF, DOCX, TXT, CSV, etc. from directory
Extract text and metadata
Return list of LangChain Document objects
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredHTMLLoader
)
from langchain.schema import Document

class DocumentLoader:
    """Loads documents from directory with support for multiple file formats"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.txt': 'text',
        '.doc': 'doc',
        '.docx': 'docx',
        '.ppt': 'ppt',
        '.pptx': 'pptx',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.csv': 'csv',
        '.json': 'json',
        '.html': 'html',
        '.md': 'markdown'
    }
    
    def __init__(self, directory_path: str):
        self.directory_path = Path(directory_path)
        
        ## Error Handling
        if not self.directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        if not self.directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
    
    def _get_loader(self, file_path: Path):
        """
        Get appropriate loader for file type
        
        Args:
            file_path: Path to file
            
        Returns:
            Loader instance or None
        """
        ext = file_path.suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(ext)
        
        if not file_type:
            return None
        
        loader_map = {
            'pdf': PyPDFLoader,
            'text': TextLoader,
            'docx': Docx2txtLoader,
            'pptx': UnstructuredPowerPointLoader,
            'excel': UnstructuredExcelLoader,
            'csv': CSVLoader,
            'json': lambda path: JSONLoader(file_path=path, jq_schema='.', text_content=False),
            'html': UnstructuredHTMLLoader,
            'markdown': UnstructuredMarkdownLoader
        }
        
        loader_class = loader_map.get(file_type)
        if not loader_class:
            return None
        
        return loader_class(str(file_path))
    
    def load_single_file(self, file_path: Path) -> List[Document]:
        try:
            loader = self._get_loader(file_path)
            
            if not loader:
                print(f"Skipping unsupported file: {file_path.name}")
                return []
            
            documents = loader.load()
                        
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source_file': str(file_path),
                    'file_name': file_path.name,
                    'file_type': self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())
                })
            
            print(f"✓ Loaded: {file_path.name} ({len(documents)} documents)")
            return documents
            
        except Exception as e:
            print(f"✗ Error loading {file_path.name}: {str(e)}")
            return []
    
    def load_all_documents(self) -> List[Document]:
        all_documents = []
        
        print(f"\n{'='*50}")
        print(f"Loading documents from: {self.directory_path}")
        print(f"{'='*50}\n")
        
        # Get all files
        files = [f for f in self.directory_path.rglob("*") if f.is_file()]
        print(f"Found {len(files)} files in directory\n")
        
        # Load each file
        for file_path in files:
            documents = self.load_single_file(file_path)
            all_documents.extend(documents)
        
        print(f"\n{'='*50}")
        print(f"Total documents loaded: {len(all_documents)}")
        print(f"{'='*50}\n")
        
        return all_documents
    
    def get_file_statistics(self) -> Dict:
        """
        Get statistics about files in directory
        
        Returns:
            Dictionary with file statistics
        """
        files = [f for f in self.directory_path.rglob("*") if f.is_file()]
        
        stats = {
            'total_files': len(files),
            'supported_files': 0,
            'unsupported_files': 0,
            'files_by_type': {}
        }
        
        for file_path in files:
            ext = file_path.suffix.lower()
            file_type = self.SUPPORTED_EXTENSIONS.get(ext)
            
            if file_type:
                stats['supported_files'] += 1
                stats['files_by_type'][file_type] = stats['files_by_type'].get(file_type, 0) + 1
            else:
                stats['unsupported_files'] += 1
        
        return stats
    
    def list_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats
        
        Returns:
            List of supported extensions
        """
        return list(self.SUPPORTED_EXTENSIONS.keys())