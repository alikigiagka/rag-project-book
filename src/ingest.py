
"""Document ingestion and index creation pipeline.

This module processes PDF documents by extracting raw text, detecting chapter and section
structures, generating sliding window chunks, processing visual figures via Gemini Vision API,
computing vector embeddings for ChromaDB storage, and building a BM25 sparse index.
"""

import glob
import json
import os
import pickle
import re
import string
from typing import Dict, List, Optional, Union
import chromadb
from dotenv import load_dotenv
import fitz
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

from src.config import (
    BOOK,
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    GEMINI_VISION_MODEL_NAME,
    MIN_CHUNK_WORDS,
)


def tokenize_greek(text: str) -> List[str]:
    """Cleans and tokenizes Greek text into term tokens for BM25 indexing.

    Args:
        text: Raw text string to be tokenized.

    Returns:
        List of cleaned single-word terms with punctuation stripped.
    """
    text = text.lower().replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if len(t) > 1]


def detect_chapter(line: str) -> Optional[str]:
    """Detects chapter and section headers from PDF lines using structural regex rules.

    Args:
        line: Single text line extracted from document page.

    Returns:
        Extracted chapter/section title string if matched, otherwise None.
    """
    line = line.strip()
    if not line:
        return None

    # Exclude isolated page numbers to prevent false positive heading detections
    if re.fullmatch(r"\d+", line):
        return None

    # Regex matching explicit Greek or English chapter headers (e.g., "Κεφάλαιο 1" or "Chapter 1")
    if re.match(r"^(κεφάλαιο|chapter)\s+\d+", line, re.IGNORECASE):
        return line[:80]

    # Regex matching hierarchical numeric section headers (e.g., "1.2 Section Title") restricted to 2-12 words
    m = re.match(r"^(\d+)(\.\d+){0,2}\s+(.+)$", line)
    if m:
        title_words = line.split()
        if 2 <= len(title_words) <= 12:
            return line[:80]

    # Heuristic for uppercase lines: 4+ letters in full uppercase within length constraint denote titles
    letters = [c for c in line if c.isalpha()]
    if (
        letters
        and len(letters) >= 4
        and all(c.isupper() for c in letters)
        and len(line) <= 80
    ):
        return line

    return None


def sliding_window_chunks(
    words: List[str],
    page_map: List[int],
    chapter_map: List[Optional[str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Union[str, int]]]:
    """Splits a stream of words into overlapping chunks with tracked metadata.

    Args:
        words: Sequential list of words extracted from document.
        page_map: Parallel list mapping each word index to its page number.
        chapter_map: Parallel list mapping each word index to its active chapter header.
        chunk_size: Target word count per chunk. Defaults to CHUNK_SIZE.
        overlap: Number of overlapping words between consecutive chunks. Defaults to CHUNK_OVERLAP.

    Returns:
        List of chunk dictionaries containing text, start page, end page, and chapter metadata.
    """
    chunks = []
    start = 0
    total = len(words)

    while start < total:
        end = min(start + chunk_size, total)
        chunk_words = words[start:end]
        text = " ".join(chunk_words)

        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(
                {
                    "text": text,
                    "page_start": page_map[start],
                    "page_end": page_map[end - 1],
                    "chapter": chapter_map[start] or "-",
                }
            )

        if end == total:
            break
        start += chunk_size - overlap

    return chunks


def describe_image(
    image_bytes: bytes, image_ext: str, page_num: int, surrounding_text: str
) -> str:
    """Generates textual descriptions of document figures using Gemini Vision LLM.

    Args:
        image_bytes: Raw binary content of the image.
        image_ext: File extension string indicating format (e.g., "png", "jpg").
        page_num: Page number where the image resides.
        surrounding_text: Text surrounding the image on the PDF page for context grounding.

    Returns:
        Generated description text string or error indicator message.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[Σφάλμα: Δεν βρέθηκε GEMINI_API_KEY στο .env]"

    client = genai.Client(api_key=api_key)

    context_hint = ""
    if surrounding_text:
        # Context is trimmed to the first 100 words of surrounding page text to ground Vision LLM output
        words = surrounding_text.split()[:100]
        context_hint = " ".join(words)

    prompt = "Περίγραψε αυτή την εικόνα περιεκτικά στα Ελληνικά. "
    if context_hint:
        prompt += f"Το κείμενο της σελίδας γύρω από την εικόνα είναι το εξής (για δικό σου context): {context_hint}. "
    prompt += "Αν είναι διάγραμμα: γράψε μια λίστα με τα components και τη ροή. Αν είναι πίνακας: κάνε transcribe τα δεδομένα. Αν είναι κώδικας: αντέγραψέ τον ακριβώς."

    mime_type = f"image/{image_ext}"
    if image_ext.lower() == "jpg":
        mime_type = "image/jpeg"

    try:
        response = client.models.generate_content(
            model=GEMINI_VISION_MODEL_NAME,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
        return response.text.strip()
    except Exception as e:
        print(f"  [Σφάλμα] Vision LLM (Gemini): {e}")
        return f"[Σφάλμα Vision LLM: {e}]"


def run_ingestion() -> None:
    """Executes full document ingestion pipeline.

    Parses input PDF documents, generates structural chunks, loads optional visual
    descriptions, creates dense vector embeddings stored in ChromaDB, and exports
    a serialized BM25 index file.
    """
    print("=" * 60)
    print(f"  Ingestion βιβλίου: {BOOK['name']}")
    print("=" * 60)

    print("\nΦόρτωση Sentence-Transformer...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        chroma_client.delete_collection(name=BOOK["collection"])
        print(f"[OK] Η παλιά collection '{BOOK['collection']}' διαγράφηκε.")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name=BOOK["collection"])

    pdf_files = glob.glob(os.path.join(BOOK["path"], "*.pdf"))
    if not pdf_files:
        print(f"[Σφάλμα] Δεν βρέθηκαν PDF αρχεία στο '{BOOK['path']}'.")
        print("  Τοποθετήστε το βιβλίο PDF στον φάκελο και επαναλάβετε.")
        return

    print(
        f"\nΒρέθηκαν {len(pdf_files)} PDF αρχεία: {[os.path.basename(f) for f in pdf_files]}"
    )

    all_chunks = []

    for filepath in pdf_files:
        filename = os.path.basename(filepath)
        print(f"\n--- Επεξεργασία: {filename} ---")
        doc = fitz.open(filepath)
        total_pages = len(doc)
        print(f"   Σελίδες: {total_pages}")

        all_words = []
        page_map = []
        chapter_map = []
        current_chapter = None

        page_texts = {}

        print("   Εξαγωγή κειμένου και ανίχνευση κεφαλαίων...")
        for page_num in tqdm(range(total_pages), desc="   Σελίδες"):
            page = doc.load_page(page_num)
            raw = page.get_text("text").strip()
            page_texts[page_num] = raw

            for line in raw.splitlines():
                ch = detect_chapter(line)
                if ch:
                    current_chapter = ch

                words = line.split()
                for w in words:
                    all_words.append(w)
                    page_map.append(page_num + 1)
                    chapter_map.append(current_chapter)

        print("   Δημιουργία text chunks (sliding window)...")
        text_chunks = sliding_window_chunks(all_words, page_map, chapter_map)
        print(f"   {len(text_chunks)} text chunks δημιουργήθηκαν.")

        for ch in text_chunks:
            all_chunks.append(
                {
                    "text": ch["text"],
                    "source": filename,
                    "type": "text",
                    "page_start": ch["page_start"],
                    "page_end": ch["page_end"],
                    "chapter": ch["chapter"],
                }
            )

        # Pre-generated vision descriptions from offline processing are ingested if present
        json_path = "image_descriptions_greek.json"

        if os.path.exists(json_path):
            print("   Ενσωμάτωση εικόνων από το JSON αρχείο...")
            with open(json_path, "r", encoding="utf-8") as f:
                image_data = json.load(f)

            for img in image_data:
                page_num = img.get("page", 1)
                description = img.get("description", "")

                image_chunk_text = (
                    f"[ΕΙΚΟΝΑ - Σελίδα {page_num}] " f"Περιγραφή: {description}"
                )

                all_chunks.append(
                    {
                        "text": image_chunk_text,
                        "source": filename,
                        "type": "image",
                        "page_start": page_num,
                        "page_end": page_num,
                        "chapter": "-",
                    }
                )
            print(f"   {len(image_data)} image chunks δημιουργήθηκαν από το JSON.")
        else:
            print(f"  [Προειδοποίηση] Δεν βρέθηκε το αρχείο {json_path}")

        doc.close()

    if not all_chunks:
        print("\n[Σφάλμα] Κανένα chunk δεν δημιουργήθηκε. Ελέγξτε το PDF.")
        return

    print(f"\nΣύνολο chunks: {len(all_chunks)}")
    print("Δημιουργία embeddings και αποθήκευση στη ChromaDB...")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    chroma_metadata = [
        {
            "source": c["source"],
            "type": c["type"],
            "page_start": int(c["page_start"]),
            "page_end": int(c["page_end"]),
            "chapter": str(c["chapter"]),
        }
        for c in all_chunks
    ]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=chroma_metadata,
        ids=ids,
    )
    print(f"[OK] {len(all_chunks)} chunks αποθηκεύτηκαν στη ChromaDB.")

    print("Χτίσιμο BM25 index...")
    tokenized = [tokenize_greek(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    bm25_path = BOOK["bm25_index"]
    if os.path.exists(bm25_path):
        os.remove(bm25_path)

    with open(bm25_path, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": texts,
                "metadata": chroma_metadata,
                "ids": ids,
            },
            f,
        )

    print(f"[OK] BM25 index αποθηκεύτηκε στο '{bm25_path}'.")
    print("\n=== Ingestion ολοκληρώθηκε επιτυχώς! ===")