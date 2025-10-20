from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from typing import Tuple, List, Dict, Any
from config.settings import Settings


class QueryEngine:
    def __init__(self, vector_store):
        print("Initializing Advanced Query Engine...")
        settings = Settings.get_summary()

        self.llm = ChatGoogleGenerativeAI(
            model=settings["LLM_MODEL"],
            google_api_key=settings["GOOGLE_API_KEY"],
            temperature=0.1
        )
        self.vector_store = vector_store

    def query(self, question: str, k: int = 5) -> Tuple[str, List[Document]]:
        """Manual RAG implementation with more control."""
        print(f"🔍 Processing query: {question}")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(question, k=k)
        
        if not retrieved_docs:
            return "No relevant documents found to answer your question.", []
        
        # Step 2: Build context
        context = self._build_context(retrieved_docs)
        
        # Step 3: Generate prompt
        prompt = self._create_prompt(question, context)
        
        # Step 4: Get LLM response
        response = self.llm.invoke(prompt)
        answer = response.content
        
        return answer, retrieved_docs

    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from documents."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source_file', doc.metadata.get('file_name', f'Document {i}'))
            context_parts.append(f"[Source: {source}]\n{doc.page_content}")
        return "\n\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for the LLM."""
        return f"""Based on the following context, please answer the question. 
If the context doesn't contain enough information to answer the question, 
please say "I cannot answer this question based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    def query_with_custom_prompt(self, question: str, custom_prompt: str, k: int = 5) -> Tuple[str, List[Document]]:
        """Query with custom prompt template."""
        retrieved_docs = self.vector_store.similarity_search(question, k=k)
        context = self._build_context(retrieved_docs)
        
        # Format custom prompt with context and question
        formatted_prompt = custom_prompt.format(context=context, question=question)
        
        response = self.llm.invoke(formatted_prompt)
        return response.content, retrieved_docs