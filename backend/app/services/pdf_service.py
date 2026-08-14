import fitz

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer


# Tokenizer used to measure chunk size.
# This is the same model family we can use later
# for our embedding model.
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def extract_pdf_pages(file_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Each page keeps its page number so that
    we can later generate source citations.
    """

    pages = []

    with fitz.open(file_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    return pages


def chunk_pdf_pages(pages: list[dict]) -> list[dict]:
    """
    Split each PDF page into semantic chunks.

    Chunk configuration:
        - 450 tokens per chunk
        - 60 token overlap
        - ~13.3% overlap

    Each chunk keeps:
        - page number
        - chunk number
        - text
        - token count
    """

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=450,
        chunk_overlap=60,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunks = []

    for page in pages:
        page_number = page["page"]
        page_text = page["text"]

        page_chunks = splitter.split_text(page_text)

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):
            token_count = len(
                tokenizer.encode(
                    chunk_text,
                    add_special_tokens=False,
                )
            )

            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_number,
                    "chunk": chunk_number,
                    "token_count": token_count,
                }
            )

    return chunks