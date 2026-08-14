import chromadb


class VectorService:
    """
    Handles storing, searching, and clearing PDF chunks
    in ChromaDB.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="chroma_db"
        )

        self.collection_name = "pdf_chunks"

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "PDF document chunks with embeddings",
                "hnsw:space": "cosine",
            }
        )

    def clear_collection(self):
        """
        Delete all existing chunks from the current collection.

        This should be called before processing a new document
        so old document chunks do not affect new answers.
        """

        existing_ids = self.collection.get(
            include=[]
        )["ids"]

        if existing_ids:
            self.collection.delete(
                ids=existing_ids
            )

    def store_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        """
        Store PDF chunks and their embeddings in ChromaDB.

        Each chunk contains:
        - text
        - page
        - chunk
        - token_count
        """

        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must be the same."
            )

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(chunks, start=1):

            # Unique ID for each chunk.
            # Including index prevents duplicate IDs.
            chunk_id = (
                f"page_{chunk['page']}"
                f"_chunk_{chunk['chunk']}"
                f"_{index}"
            )

            ids.append(chunk_id)

            documents.append(
                chunk["text"]
            )

            metadatas.append(
                {
                    "page": int(chunk["page"]),
                    "chunk": int(chunk["chunk"]),
                    "token_count": int(
                        chunk["token_count"]
                    ),
                }
            )

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return {
            "stored_chunks": len(chunks)
        }

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search for the most relevant chunks using
        vector similarity.
        """

        total_chunks = self.collection.count()

        # No document uploaded.
        if total_chunks == 0:
            return []

        # ChromaDB cannot retrieve more results
        # than currently exist.
        actual_top_k = min(
            top_k,
            total_chunks
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved_chunks = []

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        for index in range(len(documents)):

            metadata = metadatas[index]

            retrieved_chunks.append(
                {
                    "text": documents[index],
                    "page": metadata.get(
                        "page",
                        "Unknown"
                    ),
                    "chunk": metadata.get(
                        "chunk",
                        "Unknown"
                    ),
                    "token_count": metadata.get(
                        "token_count",
                        0
                    ),
                    "distance": distances[index],
                }
            )

        return retrieved_chunks

    def get_count(self) -> int:
        """
        Return the total number of chunks currently
        stored in ChromaDB.
        """

        return self.collection.count()