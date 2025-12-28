import chromadb
from indexer_config import COLLECTION_NAME, EMBEDDING_MODEL, CHROMA_DB_PATH
from embeddor import QwenEmbeddor

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
ef = QwenEmbeddor(model_name=EMBEDDING_MODEL)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

collection.add(
    ids=["id1", "id2"],
    documents=[ 
        "This is a document about pineapple",
        "This is a document about oranges"
    ]
)
