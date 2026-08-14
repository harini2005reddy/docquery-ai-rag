import sys
from pathlib import Path
import re

import streamlit as st


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# BACKEND IMPORTS
# ============================================================

from backend.app.services.pdf_service import (
    extract_pdf_pages,
    chunk_pdf_pages,
)

from backend.app.services.embedding_service import (
    EmbeddingService,
)

from backend.app.services.vector_service import (
    VectorService,
)

from backend.app.services.hybrid_retrieval_service import (
    HybridRetrievalService,
)

from backend.app.services.generation_service import (
    GenerationService,
)


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuQuery AI",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle-text {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }

    .answer-box {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .source-box {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #eeeeee;
        margin-bottom: 0.7rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "pages_count" not in st.session_state:
    st.session_state.pages_count = 0

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_uploaded_file(uploaded_file):
    """
    Save uploaded PDF temporarily.
    """

    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / uploaded_file.name

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def process_pdf(uploaded_file):
    """
    Complete PDF processing pipeline:

    1. Save PDF
    2. Extract pages
    3. Create chunks
    4. Generate embeddings
    5. Clear old ChromaDB data
    6. Store new chunks
    """

    file_path = save_uploaded_file(uploaded_file)

    # --------------------------------------------------------
    # Extract PDF text
    # --------------------------------------------------------

    pages = extract_pdf_pages(
        str(file_path)
    )

    if not pages:
        raise ValueError(
            "Could not extract text from this PDF."
        )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = chunk_pdf_pages(pages)

    if not chunks:
        raise ValueError(
            "No chunks could be created from this PDF."
        )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embedding_service = EmbeddingService()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = (
        embedding_service.generate_embeddings(
            texts
        )
    )

    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    vector_service = VectorService()

    vector_service.clear_collection()

    result = vector_service.store_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    return {
        "filename": uploaded_file.name,
        "pages": len(pages),
        "chunks": len(chunks),
        "stored_chunks": result["stored_chunks"],
    }


def generate_answer(question):
    """
    Retrieve relevant chunks and generate answer.
    """

    # --------------------------------------------------------
    # Hybrid Retrieval
    # --------------------------------------------------------

    retrieval_service = HybridRetrievalService()

    retrieved_chunks = retrieval_service.search(
        query=question,
        top_k=5,
        candidate_k=20,
    )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    generation_service = GenerationService()

    answer = ""

    for token in generation_service.generate_answer_stream(
        question=question,
        retrieved_chunks=retrieved_chunks,
    ):
        answer += token

    return answer, retrieved_chunks


def get_page_and_chunk(result):
    """
    Safely extract page and chunk information.
    """

    metadata = result.get(
        "metadata",
        {}
    )

    page = result.get(
        "page",
        metadata.get("page", "Unknown")
    )

    chunk = result.get(
        "chunk",
        metadata.get("chunk", "Unknown")
    )

    return page, chunk


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title-text">📄 DocuQuery AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle-text">
    Upload a PDF and ask questions. Answers are generated
    only from your document using RAG.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:

        st.write(
            f"**Selected:** {uploaded_file.name}"
        )

        if st.button(
            "Process Document",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Processing PDF..."
                ):

                    result = process_pdf(
                        uploaded_file
                    )

                # --------------------------------------------
                # Update session state
                # --------------------------------------------

                st.session_state.document_uploaded = True

                st.session_state.document_name = (
                    result["filename"]
                )

                st.session_state.pages_count = (
                    result["pages"]
                )

                st.session_state.chunks_count = (
                    result["chunks"]
                )

                st.session_state.messages = []

                st.success(
                    "Document processed successfully!"
                )

                st.rerun()

            except Exception as error:

                st.session_state.document_uploaded = False

                st.error(
                    f"Error processing PDF: {str(error)}"
                )


    # --------------------------------------------------------
    # DOCUMENT INFORMATION
    # --------------------------------------------------------

    if st.session_state.document_uploaded:

        st.divider()

        st.subheader("Document Info")

        st.write(
            f"**File:** "
            f"{st.session_state.document_name}"
        )

        st.write(
            f"**Pages:** "
            f"{st.session_state.pages_count}"
        )

        st.write(
            f"**Chunks:** "
            f"{st.session_state.chunks_count}"
        )

        st.divider()

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
        ):

            st.session_state.messages = []

            st.rerun()


# ============================================================
# MAIN CONTENT
# ============================================================

if not st.session_state.document_uploaded:

    st.info(
        "👈 Upload a PDF from the sidebar and click "
        "**Process Document** to start asking questions."
    )


else:

    # --------------------------------------------------------
    # SHOW CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        role = message["role"]

        with st.chat_message(role):

            st.markdown(
                message["content"]
            )

            # Show sources for assistant messages
            if (
                role == "assistant"
                and "sources" in message
                and message["sources"]
            ):

                with st.expander(
                    "📚 Retrieved Sources"
                ):

                    for index, source in enumerate(
                        message["sources"],
                        start=1,
                    ):

                        page, chunk = (
                            get_page_and_chunk(
                                source
                            )
                        )

                        text = source.get(
                            "text",
                            ""
                        )

                        st.markdown(
                            f"**Source {index} — "
                            f"Page {page}, "
                            f"Chunk {chunk}**"
                        )

                        st.write(text)

                        if index < len(
                            message["sources"]
                        ):
                            st.divider()


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about your document..."
    )


    if question:

        # ----------------------------------------------------
        # Display user question
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)


        # ----------------------------------------------------
        # Generate assistant answer
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            placeholder = st.empty()

            try:

                with st.spinner(
                    "Searching the document..."
                ):

                    answer, sources = generate_answer(
                        question
                    )

                # Display answer
                placeholder.markdown(answer)

                # Display retrieved context
                if sources:

                    with st.expander(
                        "📚 Retrieved Sources"
                    ):

                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):

                            page, chunk = (
                                get_page_and_chunk(
                                    source
                                )
                            )

                            text = source.get(
                                "text",
                                ""
                            )

                            st.markdown(
                                f"### Source {index}"
                            )

                            st.markdown(
                                f"**Page:** {page}  |  "
                                f"**Chunk:** {chunk}"
                            )

                            st.write(text)

                            if index < len(sources):
                                st.divider()


                # --------------------------------------------
                # Save assistant message
                # --------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )


            except Exception as error:

                error_message = (
                    f"❌ Error generating answer: "
                    f"{str(error)}"
                )

                placeholder.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DocuQuery AI • RAG-based PDF Question Answering "
    "using Hybrid Retrieval and Groq"
)