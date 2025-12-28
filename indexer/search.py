import chromadb
from indexer_config import COLLECTION_NAME, EMBEDDING_MODEL
from embeddor import QwenEmbeddor

client = chromadb.HttpClient(host='localhost', port=8000)
ef = QwenEmbeddor(model_name=EMBEDDING_MODEL)

collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


results = collection.query(
    query_texts=["This is a query document about hawaii"],
    n_results=2 
)
print(results)
