
"""Configuration settings for the Retrieval-Augmented Generation (RAG) system.

This module defines directory paths, model identifiers, chunking parameters,
retrieval thresholds, and image processing constraints used across ingestion
and query pipelines.
"""

import os
from typing import Dict

BOOK: Dict[str, str] = {
    "name": "Τεχνολογίες Επεξεργασίας και Ανάλυσης Μεγάλων Δεδομένων",
    "path": "data/book",
    "collection": "bigdata_book",
    "bm25_index": "bm25_bigdata.pkl",
}

DATA_BASE_DIR: str = "data"
CHROMA_PATH: str = "chroma_db"

# Multilingual model fine-tuned for dense asymmetric search queries
EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large-instruct"
GEMINI_LLM_MODEL_NAME: str = "gemini-2.5-flash"
GEMINI_VISION_MODEL_NAME: str = "gemini-2.5-flash"

PROCESS_IMAGES: bool = True
VISION_TIMEOUT: int = 180
MIN_IMAGE_WIDTH: int = 150
MIN_IMAGE_HEIGHT: int = 150
# Minimum surface area threshold in square pixels to ignore small icons and logos
MIN_IMAGE_AREA: int = 50_000

# Chunk size of 400 words keeps text snippets within the embedding model token context limit
CHUNK_SIZE: int = 400
# Overlap prevents split-sentence context loss across adjacent chunks
CHUNK_OVERLAP: int = 100
# Minimum word threshold filters out header, footer, and page-number fragments
MIN_CHUNK_WORDS: int = 30

# Reciprocal Rank Fusion smoothing parameter (k=60) per Cormack et al.
RRF_K: int = 60
TOP_K_DENSE: int = 8
TOP_K_SPARSE: int = 8
FINAL_TOP_K: int = 5

os.makedirs(DATA_BASE_DIR, exist_ok=True)
os.makedirs(BOOK["path"], exist_ok=True)
