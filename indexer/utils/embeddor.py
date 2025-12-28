from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer


class QwenEmbeddor(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        """
        input: List[str]
        output: List[List[float]]
        """
        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()
