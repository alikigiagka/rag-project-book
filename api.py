"""FastAPI web server interface for the Book RAG system.

This module exposes HTTP endpoints for querying available course material, executing
hybrid document retrieval and Gemini response generation, and serving the static frontend interface.
"""

import json
import os
import threading
import warnings
from typing import Dict, List, Optional

# Suppress transitive dependency deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import BOOK
from src.llm import generate_response
from src.retriever import Retriever

app = FastAPI(title="RAG Book QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever: Optional[Retriever] = None
retriever_lock = threading.Lock()


def get_retriever() -> Retriever:
    """Returns singleton instance of Retriever using thread-safe double-checked locking.

    Returns:
        Initialized Retriever object instance.

    Raises:
        HTTPException: If initializing the retriever fails due to missing index files.
    """
    global _retriever
    if _retriever is None:
        # Double-checked locking prevents race conditions during concurrent worker thread startup
        with retriever_lock:
            if _retriever is None:
                try:
                    _retriever = Retriever()
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Σφάλμα φόρτωσης του RAG Retriever: {e}",
                    )
    return _retriever


class ChatRequest(BaseModel):
    """Pydantic model representing incoming chat API request payloads."""

    course_key: str
    query: str


class SourceItem(BaseModel):
    """Pydantic model representing a single referenced document source with page numbers."""

    name: str
    pages: List[int] = []


class ChatResponse(BaseModel):
    """Pydantic model representing outgoing chat API response payloads."""

    answer: str
    sources: List[SourceItem]


class CourseItem(BaseModel):
    """Pydantic model representing course metadata in catalog endpoint."""

    key: str
    name: str


evaluation_lock = threading.Lock()


def log_evaluation_record(query: str, context_texts: List[str], answer: str) -> None:
    """Appends QA interaction data to the offline evaluation dataset file in a thread-safe manner.

    Args:
        query: User input question string.
        context_texts: List of retrieved context passages formatted for the LLM.
        answer: Generated answer text or error string.
    """
    evaluation_file = "rag_evaluation_data.json"
    retrieved_context_str = "\n---\n".join(context_texts)

    # Thread lock prevents concurrent request handlers from corrupting JSON disk writes
    with evaluation_lock:
        evaluation_dataset = []

        try:
            if os.path.exists(evaluation_file):
                with open(evaluation_file, "r", encoding="utf-8") as f:
                    evaluation_dataset = json.load(f)
        except Exception:
            pass

        evaluation_record = {
            "query": query,
            "retrieved_context": retrieved_context_str,
            "generated_answer": answer,
        }
        evaluation_dataset.append(evaluation_record)

        try:
            with open(evaluation_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_dataset, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[Καταγραφή] Σφάλμα αποθήκευσης στο '{evaluation_file}': {e}")


@app.get("/api/courses", response_model=List[CourseItem])
def list_courses() -> List[Dict[str, str]]:
    """Retrieves list of available books and courses.

    Returns:
        List of course dictionary items containing course key and title.
    """
    return [{"key": "bigdata_book", "name": BOOK["name"]}]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Handles chat queries via hybrid document retrieval and Gemini response generation.

    Args:
        req: ChatRequest object containing course identifier and query text.

    Returns:
        ChatResponse object containing generated answer string and structured source references.

    Raises:
        HTTPException: If query is empty (400), course key is invalid (404), or Gemini API fails (502).
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Η ερώτηση είναι κενή.")

    if req.course_key != "bigdata_book":
        raise HTTPException(status_code=404, detail="Το επιλεγμένο βιβλίο/μάθημα δεν βρέθηκε.")

    retriever = get_retriever()
    context_texts, sources_with_pages = retriever.retrieve(query)

    if not context_texts:
        answer = "Δεν βρέθηκαν σχετικά αποσπάσματα για αυτή την ερώτηση στο βιβλίο."
        log_evaluation_record(query, [], answer)
        return ChatResponse(answer=answer, sources=[])

    answer, error = generate_response(query, context_texts)

    if error:
        raise HTTPException(
            status_code=502,
            detail=f"Σφάλμα κατά την επικοινωνία με το Gemini API: {error}",
        )

    sources = [
        SourceItem(name=name, pages=sorted(list(pages)))
        for name, pages in sorted(sources_with_pages.items())
    ]

    log_evaluation_record(query, context_texts, answer)

    return ChatResponse(answer=answer, sources=sources)


@app.get("/")
def serve_index() -> FileResponse:
    """Serves the main single-page web interface index HTML document.

    Returns:
        FileResponse serving static/index.html.
    """
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")

