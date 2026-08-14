import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.hybrid_retrieval_service import (
    HybridRetrievalService,
)
from app.services.generation_service import (
    GenerationService,
)


router = APIRouter()


class QuestionRequest(BaseModel):
    question: str


@router.post("/chat/stream")
def stream_answer(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    retrieval_service = HybridRetrievalService()

    retrieved_chunks = retrieval_service.search(
        query=question,
        top_k=3,
    )

    generation_service = GenerationService()


    def generate():

        # Send retrieved chunks first
        chunks_data = []

        for result in retrieved_chunks:

            metadata = result.get("metadata", {})

            chunks_data.append(
                {
                    "page": metadata.get("page"),
                    "chunk": metadata.get("chunk"),
                    "text": result.get("text", ""),
                    "rrf_score": result.get(
                        "rrf_score",
                        0,
                    ),
                }
            )

        yield (
            "data: "
            + json.dumps(
                {
                    "type": "context",
                    "chunks": chunks_data,
                }
            )
            + "\n\n"
        )


        # Stream LLM tokens
        for token in generation_service.generate_answer_stream(
            question=question,
            retrieved_chunks=retrieved_chunks,
        ):

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "token",
                        "content": token,
                    }
                )
                + "\n\n"
            )


        # Tell frontend generation is complete
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "done",
                }
            )
            + "\n\n"
        )


    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )