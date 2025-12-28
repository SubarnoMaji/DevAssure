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

client = chromadb.HttpClient(host='localhost', port=8000)
ef = QwenEmbeddor(model_name=EMBEDDING_MODEL)

collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


results = collection.query(
    query_texts=["This is a query document about hawaii"],
    n_results=2 
)
print(results)
