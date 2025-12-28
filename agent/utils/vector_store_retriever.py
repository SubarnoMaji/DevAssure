import chromadb
import logging
import sys
import os

# Add project root to path to import from indexer
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from indexer.utils.vector_store_config import COLLECTION_NAME, EMBEDDING_MODEL
from indexer.utils.embeddor import QwenEmbeddor

class VectorStoreRetriever:
    def __init__( 
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL
    ):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.embeddor = QwenEmbeddor(model_name=embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embeddor
        )
    
    def query(self, query_texts, n_results=4, **kwargs):
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            **kwargs
        )

    def __call__(self, query_texts, n_results=4, **kwargs):
        """
        Make instances directly callable for querying.
        Example: retriever(["text"], n_results=3)
        """
        return self.query(query_texts, n_results, **kwargs)


if __name__ == "__main__":
    retriever = VectorStoreRetriever()
    results = retriever(["what invoice is this?"], n_results=4)
    print(results)

