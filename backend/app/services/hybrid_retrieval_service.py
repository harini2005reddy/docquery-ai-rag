import chromadb
from rank_bm25 import BM25Okapi

from app.services.embedding_service import EmbeddingService


class HybridRetrievalService:

    COLLECTION_NAME = "pdf_chunks"

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="chroma_db"
        )

        self.collection = self.client.get_collection(
            name=self.COLLECTION_NAME
        )

        print(
            f"Using ChromaDB collection: "
            f"{self.COLLECTION_NAME}"
        )

        self.embedding_service = EmbeddingService()

        # Get all documents once for BM25 indexing
        collection_data = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = collection_data["documents"]
        self.metadatas = collection_data["metadatas"]
        self.ids = collection_data["ids"]

        # Tokenize all chunks for BM25
        self.tokenized_documents = [
            document.lower().split()
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    def vector_search(
        self,
        query: str,
        candidate_k: int = 20,
    ) -> list[dict]:

        query_embedding = (
            self.embedding_service.generate_embeddings(
                [query]
            )[0]
        )

        candidate_k = min(
            candidate_k,
            self.collection.count()
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        vector_results = []

        for rank, (
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):

            # Create a unique identifier
            key = (
                f'{metadata.get("page")}_'
                f'{metadata.get("chunk")}'
            )

            vector_results.append(
                {
                    "key": key,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                    "vector_rank": rank,
                }
            )

        return vector_results

    def bm25_search(
        self,
        query: str,
        candidate_k: int = 20,
    ) -> list[dict]:

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        ranked_indices = ranked_indices[
            :candidate_k
        ]

        bm25_results = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):

            metadata = self.metadatas[index]

            key = (
                f'{metadata.get("page")}_'
                f'{metadata.get("chunk")}'
            )

            bm25_results.append(
                {
                    "key": key,
                    "text": self.documents[index],
                    "metadata": metadata,
                    "bm25_score": float(
                        scores[index]
                    ),
                    "bm25_rank": rank,
                }
            )

        return bm25_results

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion.

        Steps:
        1. Retrieve a larger candidate pool from vector search.
        2. Retrieve a larger candidate pool from BM25.
        3. Combine rankings using RRF.
        4. Return the final top_k results.
        """

        # Get larger candidate pools
        vector_results = self.vector_search(
            query=query,
            candidate_k=candidate_k,
        )

        bm25_results = self.bm25_search(
            query=query,
            candidate_k=candidate_k,
        )

        combined_results = {}

        # Add vector search results
        for result in vector_results:

            key = result["key"]

            combined_results[key] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "vector_rank": result["vector_rank"],
                "bm25_rank": None,
                "rrf_score": 0.0,
            }

        # Add BM25 search results
        for result in bm25_results:

            key = result["key"]

            if key not in combined_results:

                combined_results[key] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "vector_rank": None,
                    "bm25_rank": result["bm25_rank"],
                    "rrf_score": 0.0,
                }

            else:
                combined_results[key][
                    "bm25_rank"
                ] = result["bm25_rank"]

        # Apply Reciprocal Rank Fusion
        for result in combined_results.values():

            vector_rank = result["vector_rank"]
            bm25_rank = result["bm25_rank"]

            rrf_score = 0.0

            if vector_rank is not None:
                rrf_score += (
                    1 / (rrf_k + vector_rank)
                )

            if bm25_rank is not None:
                rrf_score += (
                    1 / (rrf_k + bm25_rank)
                )

            result["rrf_score"] = rrf_score

        # Sort by final RRF score
        final_results = sorted(
            combined_results.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )

        return final_results[:top_k]