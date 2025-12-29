import chromadb
from .vector_store_config import COLLECTION_NAME, EMBEDDING_MODEL, CHROMA_DB_PATH
from .embeddor import QwenEmbeddor


class ChromaVectorStore:
    def __init__(
        self,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        db_path=CHROMA_DB_PATH
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.embeddor = QwenEmbeddor(model_name=self.embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embeddor
        )

    def add_documents(self, ids, documents, metadatas=None):
        if metadatas is not None:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        else:
            self.collection.add(
                ids=ids,
                documents=documents
            )

    def delete_documents(self, ids):
        self.collection.delete(ids=ids)

    def get_documents_by_source(self, source_file):
        results = self.collection.get(
            where={"source": source_file}
        )
        return results.get("ids", [])

    def get_number_of_documents(self):
        return self.collection.count()

    def get_collection(self):
        return self.collection
