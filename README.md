# 📄 DocuQuery AI

### Intelligent Document Q&A System using Retrieval-Augmented Generation (RAG)

🔗 **Live Demo:** (https://youtu.be/aOZAJ-_cHWc)

DocuQuery AI is an end-to-end Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents 
and ask questions based strictly on the content of the uploaded document.

The system processes multi-page PDFs, splits the document into chunks, generates semantic embeddings, stores them in
ChromaDB, and retrieves relevant information using Hybrid Retrieval. Answers are generated using Groq and include explicit
page and chunk citations.

---

## 🚀 Live Application

🔗 **Try DocuQuery AI:** (https://docquery-ai-rag-di448hm3ky4yqyjqzey2bd.streamlit.app/)

---

## ✨ Features

- 📄 Upload and process PDF documents
- 📚 Support for multi-page PDFs
- ✂️ Recursive text chunking with overlap
- 🧠 Semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- 🗄️ Persistent vector storage using ChromaDB
- 🔍 Dense semantic similarity search
- 🔎 BM25 keyword-based retrieval
- 🔀 Hybrid Retrieval combining dense search and BM25
- 🤖 Grounded answer generation using Groq LLM
- 🛡️ Hallucination prevention through strict document grounding
- 📌 Explicit page and chunk citations
- 📂 Collapsible retrieved source context
- 🌐 Interactive Streamlit web interface
- ☁️ Public cloud deployment using Streamlit Community Cloud

---

## 🏗️ System Architecture

```text
                    ┌─────────────────┐
                    │   Upload PDF    │
                    └────────┬────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   PDF Processing     │
                  │       PyMuPDF        │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Text Chunking        │
                  │ Recursive Splitter   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Generate Embeddings  │
                  │ all-MiniLM-L6-v2     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      ChromaDB        │
                  │    Vector Store      │
                  └──────────┬───────────┘
                             │
                             ▼
User Query ───────► Hybrid Retrieval
                    ┌──────────────────────┐
                    │ Dense Vector Search  │
                    │        +             │
                    │    BM25 Search       │
                    │        +             │
                    │ Reciprocal Rank Fusion│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Groq LLM        │
                    │ Strict Grounding     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Answer + Citations   │
                    │ [Page X, Chunk Y]    │
                    └──────────────────────┘
```

---

## 🔄 RAG Pipeline

### 1. PDF Processing

PDF files are uploaded through the Streamlit interface and processed using **PyMuPDF**.

The system extracts text page by page while preserving page metadata. This allows the application to provide page-level 
citations in the final response.

---

### 2. Text Chunking

The extracted document text is split into smaller chunks using LangChain's:

```python
RecursiveCharacterTextSplitter
```

Chunking helps preserve semantic meaning while ensuring that the retrieved context fits within the LLM context window.

Each chunk stores metadata such as:

- PDF filename
- Page number
- Chunk number

This metadata is later used to generate citations such as:

```text
[Page 4, Chunk 2]
```

---

### 3. Embedding Generation

Semantic embeddings are generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model converts document chunks and user queries into dense vector representations.

Embeddings are normalized before retrieval to support semantic similarity comparison.

---

### 4. Vector Storage

Document embeddings and metadata are stored in **ChromaDB**.

ChromaDB provides persistent vector storage and semantic similarity retrieval for the processed PDF chunks.

---

### 5. Hybrid Retrieval

The system combines two retrieval approaches:

#### Dense Vector Search

Semantic search retrieves chunks based on the meaning of the user's question.

#### BM25 Keyword Search

BM25 retrieves chunks based on important keywords and lexical matches.

#### Reciprocal Rank Fusion

The results from vector search and BM25 search are combined to improve retrieval quality.

This approach helps retrieve relevant information for both:

- Semantic questions
- Exact keyword-based questions

---

### 6. Grounded Answer Generation

The retrieved document chunks are passed to a Groq-powered LLM.

The prompt is designed to enforce strict grounding.

The model must answer only from the retrieved document context.

If the answer cannot be found in the provided document, the system responds:

```text
Information not found in the provided document.
```

This helps prevent unsupported answers and reduces hallucinations.

---

### 7. Citation System

Every answer is associated with retrieved document sources.

Sources contain:

- Page number
- Chunk number
- Retrieved document content

Example:

```text
The system is designed to predict product prices by combining textual
descriptions with engineered numeric features. [Page 1, Chunk 1]
```

Users can also expand the **Retrieved Sources** section to inspect the document context used for retrieval.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Text Chunking | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Keyword Retrieval | Rank-BM25 |
| Retrieval Strategy | Hybrid Search + Reciprocal Rank Fusion |
| LLM | Groq |
| Deep Learning | PyTorch |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```text
docquery-ai-rag/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat.py
│   │   │   ├── question.py
│   │   │   └── upload.py
│   │   │
│   │   ├── models/
│   │   │   └── schemas.py
│   │   │
│   │   ├── services/
│   │   │   ├── embedding_service.py
│   │   │   ├── generation_service.py
│   │   │   ├── hybrid_retrieval_service.py
│   │   │   ├── pdf_service.py
│   │   │   └── vector_service.py
│   │   │
│   │   └── main.py
│   │
│   └── requirements.txt
│
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/harini2005reddy/docquery-ai-rag.git
cd docquery-ai-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file:

```text
GROQ_API_KEY=your_groq_api_key_here
```

> Never commit your `.env` file or API key to GitHub.

### 6. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will start locally and can be accessed through the URL displayed in the terminal.

---

## 🧪 Example Questions

After uploading a document, try questions such as:

```text
What is the main topic of this document?
```

```text
What are the input features used by the system?
```

```text
How are product descriptions processed?
```

```text
What evaluation metrics are used?
```

```text
What are the limitations of the system?
```

For an unsupported question, the system should respond:

```text
Information not found in the provided document.
```

---

## 🖥️ Application Workflow

1. Upload a PDF document.
2. Click **Process Document**.
3. The PDF text is extracted page by page.
4. Text is split into semantic chunks.
5. Embeddings are generated for each chunk.
6. Chunks are stored in ChromaDB.
7. Enter a question about the uploaded document.
8. Hybrid Retrieval finds relevant chunks using dense search and BM25.
9. The LLM generates a grounded response.
10. Retrieved sources are displayed with page and chunk citations.

---

## 🛡️ Hallucination Prevention

A core objective of this project is to prevent the LLM from generating information outside the uploaded document.

The generation pipeline restricts the model to the retrieved document context.

If sufficient information is not available, the application returns:

> **Information not found in the provided document.**

This ensures that responses remain grounded in the source document.

---

## 🎥 Demo

A short walkthrough demonstrates:

- PDF upload
- Document processing
- Question answering
- Hybrid retrieval
- Retrieved context inspection
- Page and chunk citations
- Unsupported question handling

**Demo Video/GIF:** Add your demo link or GIF here.

---

## 🎯 Assessment Requirements Covered

- [x] Multi-page PDF ingestion
- [x] Text extraction
- [x] Recursive text chunking
- [x] Dense embeddings
- [x] ChromaDB vector storage
- [x] Semantic similarity retrieval
- [x] Hybrid retrieval using BM25 and dense search
- [x] Groq LLM integration
- [x] Strict document grounding
- [x] Hallucination prevention
- [x] Page and chunk citations
- [x] Interactive Streamlit interface
- [x] Collapsible retrieved context
- [x] Public GitHub repository
- [x] Public live deployment

---

## 🔮 Future Improvements

- Token-by-token streaming responses
- Support for multiple PDFs
- Chat history persistence
- RAG evaluation using Ragas
- Automated faithfulness scoring
- Improved citation formatting
- Support for additional document formats
- Advanced reranking models

---

## 👩‍💻 Author

**Harini Ravula**

AI Engineer Intern Technical Assessment  
Intelligent Document Q&A System with Retrieval-Augmented Generation (RAG)
