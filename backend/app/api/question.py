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
from app.services.evaluation_service import (
    EvaluationService,
)


router = APIRouter()


# --------------------------------
# Request model
# --------------------------------
class QuestionRequest(BaseModel):
    question: str


# ================================================
# NORMAL QUESTION API
# POST /api/ask
# ================================================
@router.post("/ask")
def ask_question(data: QuestionRequest):

    question = data.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        # --------------------------------
        # 1. Retrieve relevant chunks
        # --------------------------------
        retrieval_service = HybridRetrievalService()

        retrieved_chunks = retrieval_service.search(
            query=question,
            top_k=5,
        )

        print(
            f"Retrieved chunks: "
            f"{len(retrieved_chunks)}"
        )

        # --------------------------------
        # No context found
        # --------------------------------
        if not retrieved_chunks:
            return {
                "answer": (
                    "Information not found in the "
                    "provided document."
                ),
                "chunks": [],
                "evaluation": {
                    "faithfulness_score": 0.0,
                    "verdict": "No context available",
                },
            }

        # --------------------------------
        # 2. Generate grounded answer
        # --------------------------------
        generation_service = GenerationService()

        answer = generation_service.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        print(f"Generated answer: {answer}")

        # --------------------------------
        # 3. Evaluate faithfulness
        # --------------------------------
        evaluation_service = EvaluationService()

        evaluation = (
            evaluation_service.evaluate_faithfulness(
                question=question,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
            )
        )

        print(
            f"Evaluation result: {evaluation}"
        )

        # --------------------------------
        # 4. Format chunks
        # --------------------------------
        formatted_chunks = format_chunks(
            retrieved_chunks
        )

        # --------------------------------
        # 5. Return response
        # --------------------------------
        return {
            "answer": answer,
            "chunks": formatted_chunks,
            "evaluation": evaluation,
        }

    except Exception as error:

        print(
            "\n========== ASK ERROR =========="
        )
        print(error)
        print(
            "================================\n"
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ================================================
# STREAMING QUESTION API
# POST /api/chat/stream
# ================================================
@router.post("/chat/stream")
def stream_question(data: QuestionRequest):

    question = data.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    def generate():

        try:

            # --------------------------------
            # 1. Retrieve chunks
            # --------------------------------
            retrieval_service = HybridRetrievalService()

            retrieved_chunks = retrieval_service.search(
                query=question,
                top_k=5,
            )

            print(
                f"Retrieved chunks: "
                f"{len(retrieved_chunks)}"
            )

            formatted_chunks = format_chunks(
                retrieved_chunks
            )

            # --------------------------------
            # Send retrieved context FIRST
            # --------------------------------
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "context",
                        "chunks": formatted_chunks,
                    }
                )
                + "\n\n"
            )

            # --------------------------------
            # No context found
            # --------------------------------
            if not retrieved_chunks:

                answer = (
                    "Information not found in the "
                    "provided document."
                )

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "token",
                            "content": answer,
                        }
                    )
                    + "\n\n"
                )

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "evaluation",
                            "evaluation": {
                                "faithfulness_score": 0.0,
                                "verdict": "No context available",
                            },
                        }
                    )
                    + "\n\n"
                )

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "done",
                        }
                    )
                    + "\n\n"
                )

                return

            # --------------------------------
            # 2. Generate answer with streaming
            # --------------------------------
            generation_service = GenerationService()

            complete_answer = ""

            for token in generation_service.generate_answer_stream(
                question=question,
                retrieved_chunks=retrieved_chunks,
            ):

                complete_answer += token

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

            print(
                f"Complete answer: {complete_answer}"
            )

            # --------------------------------
            # 3. Evaluate faithfulness
            # --------------------------------
            evaluation_service = EvaluationService()

            evaluation = (
                evaluation_service.evaluate_faithfulness(
                    question=question,
                    answer=complete_answer,
                    retrieved_chunks=retrieved_chunks,
                )
            )

            print(
                f"Evaluation result: {evaluation}"
            )

            # --------------------------------
            # Send evaluation
            # --------------------------------
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "evaluation",
                        "evaluation": evaluation,
                    }
                )
                + "\n\n"
            )

            # --------------------------------
            # Streaming complete
            # --------------------------------
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                    }
                )
                + "\n\n"
            )

        except Exception as error:

            print(
                "\n========== STREAM ERROR =========="
            )
            print(error)
            print(
                "==================================\n"
            )

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "message": str(error),
                    }
                )
                + "\n\n"
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


# ================================================
# HELPER: FORMAT RETRIEVED CHUNKS
# ================================================
def format_chunks(retrieved_chunks):

    formatted_chunks = []

    for result in retrieved_chunks:

        metadata = result.get(
            "metadata",
            {},
        )

        formatted_chunks.append(
            {
                "page": result.get(
                    "page",
                    metadata.get("page", 0),
                ),
                "chunk": result.get(
                    "chunk",
                    metadata.get("chunk", 0),
                ),
                "text": result.get(
                    "text",
                    "",
                ),
                "rrf_score": result.get(
                    "rrf_score",
                    0.0,
                ),
            }
        )

    return formatted_chunks