import sys
import os
from typing import Optional
from langchain_core.tools import tool
from utils.vector_store_retriever import VectorStoreRetriever

_retriever: Optional[VectorStoreRetriever] = None


def get_retriever(host: str = "localhost", port: int = 8000) -> VectorStoreRetriever:
    global _retriever
    if _retriever is None:
        _retriever = VectorStoreRetriever(host=host, port=port)
    return _retriever


@tool
def vector_store_search(query: str, n_results: int = 4) -> str:
    """Search the vector store for relevant documents based on a query."""
    try:
        retriever = get_retriever()
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


if __name__ == "__main__":
    result = vector_store_search.invoke({"query": "What is this document about?", "n_results": 2})
    print(result)
