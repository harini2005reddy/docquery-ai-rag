import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.upload import router as upload_router
from app.api.chat import router as chat_router


load_dotenv()


app = FastAPI(
    title="DocuQuery AI API",
    description=(
        "RAG-based document question answering system"
    ),
    version="1.0.0",
)


allowed_origins = [
    "http://localhost:5173",
]


frontend_url = os.getenv(
    "FRONTEND_URL"
)

if frontend_url:
    allowed_origins.append(
        frontend_url
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    upload_router,
    prefix="/api",
    tags=["Upload"],
)


app.include_router(
    chat_router,
    prefix="/api",
    tags=["Chat"],
)


@app.get("/")
def root():
    return {
        "message": (
            "DocuQuery AI Backend is running"
        )
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }