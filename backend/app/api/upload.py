from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.pdf_service import (
    extract_pdf_pages,
    chunk_pdf_pages,
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService


router = APIRouter()


# -----------------------------
# Upload folder
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF.

    Pipeline:
    1. Save PDF
    2. Extract text from pages
    3. Create chunks
    4. Generate embeddings
    5. Clear previous document from ChromaDB
    6. Store new chunks in ChromaDB
    7. Return document information
    """

    # -----------------------------
    # Validate file
    # -----------------------------
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    try:
        # -----------------------------
        # 1. Save PDF
        # -----------------------------
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"\nPDF uploaded: {safe_filename}")

        # -----------------------------
        # 2. Extract pages
        # -----------------------------
        pages = extract_pdf_pages(str(file_path))

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the PDF.",
            )

        print(f"Pages extracted: {len(pages)}")

        # -----------------------------
        # 3. Create chunks
        # -----------------------------
        chunks = chunk_pdf_pages(pages)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks could be created from the PDF.",
            )

        print(f"Chunks created: {len(chunks)}")

        # -----------------------------
        # 4. Generate embeddings
        # -----------------------------
        embedding_service = EmbeddingService()

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embedding_service.generate_embeddings(texts)

        print(
            f"Embeddings generated: {len(embeddings)}"
        )

        # -----------------------------
        # 5. Connect to ChromaDB
        # -----------------------------
        vector_service = VectorService()

        # Remove all chunks from the previous document
        vector_service.clear_collection()

        print("Previous document cleared from ChromaDB")

        # -----------------------------
        # 6. Store new chunks
        # -----------------------------
        result = vector_service.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        total_chunks = vector_service.get_count()

        print(
            f"Chunks stored in ChromaDB: {total_chunks}"
        )

        # -----------------------------
        # 7. Return response
        # -----------------------------
        return {
            "success": True,
            "message": "PDF processed successfully.",
            "filename": safe_filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "stored_chunks": result["stored_chunks"],
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Upload error: {error}")

        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(error)}",
        )

    finally:
        await file.close()