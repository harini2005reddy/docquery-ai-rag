from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:

    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            print("Loading embedding model...")
            cls._model = SentenceTransformer(MODEL_NAME)

        return cls._model


    def generate_embedding(self, text: str) -> list[float]:

        model = self.get_model()

        embedding = model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()


    def generate_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        model = self.get_model()

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()