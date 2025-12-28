import sys
import os
from typing import Optional
from langchain_core.tools import tool
from utils.vector_store_retriever import VectorStoreRetriever

# Initialize retriever instance (singleton pattern)
_retriever: Optional[VectorStoreRetriever] = None


def get_retriever(host: str = "localhost", port: int = 8000) -> VectorStoreRetriever:
    """Get or create the VectorStoreRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = VectorStoreRetriever(host=host, port=port)
    return _retriever


@tool
def vector_store_search(query: str, n_results: int = 4) -> str:
    """
    Search the vector store for relevant documents based on a query.
    
    This tool retrieves the most relevant documents from the indexed vector store
    that match the given query. Use this when you need to find information from
    the document collection.
    
    Args:
        query: The search query/question to find relevant documents
        n_results: Number of results to return (default: 4)
        
    Returns:
        A formatted string containing the retrieved documents and their metadata
    """
    try:
        retriever = get_retriever()
        results = retriever.query(query_texts=[query], n_results=n_results)
        
        if not results or not results.get("documents") or not results["documents"][0]:
            return "No relevant documents found in the vector store."
        
        # Format the results for the LLM
        documents = results["documents"][0]
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        
        formatted_results = []
        for i, doc in enumerate(documents):
            result_text = f"\n--- Result {i+1} ---\n"
            result_text += f"Content: {doc}\n"
            
            if metadatas and i < len(metadatas):
                metadata = metadatas[i]
                if metadata:
                    result_text += f"Metadata: {metadata}\n"
            
            if distances and i < len(distances):
                result_text += f"Relevance Score: {1 - distances[i]:.4f}\n"  # Convert distance to similarity
            
            formatted_results.append(result_text)
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching vector store: {str(e)}"


# Alternative: Create a tool that can be configured with retriever parameters
def create_retriever_tool(host: str = "localhost", port: int = 8000, n_results: int = 4):
    """
    Create a retriever tool with custom configuration.
    
    Args:
        host: ChromaDB server host
        port: ChromaDB server port
        n_results: Default number of results
        
    Returns:
        A configured LangChain tool
    """
    retriever = VectorStoreRetriever(host=host, port=port)
    
    @tool
    def search_vector_store(query: str) -> str:
        """
        Search the vector store for relevant documents based on a query.
        
        Args:
            query: The search query/question to find relevant documents
            
        Returns:
            A formatted string containing the retrieved documents and their metadata
        """
        try:
            results = retriever.query(query_texts=[query], n_results=n_results)
            
            if not results or not results.get("documents") or not results["documents"][0]:
                return "No relevant documents found in the vector store."
            
            documents = results["documents"][0]
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []
            
            formatted_results = []
            for i, doc in enumerate(documents):
                result_text = f"\n--- Result {i+1} ---\n"
                result_text += f"Content: {doc}\n"
                
                if metadatas and i < len(metadatas):
                    metadata = metadatas[i]
                    if metadata:
                        result_text += f"Metadata: {metadata}\n"
                
                if distances and i < len(distances):
                    result_text += f"Relevance Score: {1 - distances[i]:.4f}\n"
                
                formatted_results.append(result_text)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching vector store: {str(e)}"
    
    return search_vector_store


if __name__ == "__main__":
    # Test the tool
    tool = vector_store_search
    result = tool.invoke({"query": "What is this document about?", "n_results": 2})
    print(result)

