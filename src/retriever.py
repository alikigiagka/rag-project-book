
"""Hybrid retriever combining vector embeddings and BM25 search.

This module implements a hybrid retrieval framework that queries ChromaDB for dense
semantic similarity and BM25 for sparse keyword matching, combining the resulting
rankings using Reciprocal Rank Fusion (RRF).
"""

import os
import pickle
import string
from typing import Dict, List, Set, Tuple
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    BOOK,
    CHROMA_PATH,
    EMBEDDING_MODEL_NAME,
    FINAL_TOP_K,
    RRF_K,
    TOP_K_DENSE,
    TOP_K_SPARSE,
)


def tokenize_greek(text: str) -> List[str]:
    """Tokenizes and cleans Greek text for lexical indexing.

    Args:
        text: Raw input string to be tokenized.

    Returns:
        List of cleaned token strings with punctuation removed and short terms filtered.
    """
    text = text.lower().replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if len(t) > 1]


def rrf_score(
    dense_ranks: Dict[str, int], sparse_ranks: Dict[str, int], k: int = RRF_K
) -> List[Tuple[str, float]]:
    """Calculates Reciprocal Rank Fusion (RRF) scores across retrieval modalities.

    The score formula score(d) = sum(1 / (rank(d) + k)) rewards items that rank highly
    in both dense vector search and sparse BM25 search.

    Args:
        dense_ranks: Mapping from document chunk ID to its 1-based dense rank position.
        sparse_ranks: Mapping from document chunk ID to its 1-based sparse rank position.
        k: Smoothing constant preventing high-rank dominance. Defaults to RRF_K.

    Returns:
        List of (chunk_id, rrf_score) tuples sorted in descending score order.
    """
    scores: Dict[str, float] = {}
    for chunk_id, rank in dense_ranks.items():
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank + k)
    for chunk_id, rank in sparse_ranks.items():
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    """Hybrid document retriever leveraging ChromaDB dense embeddings and BM25 lexical index."""

    def __init__(self) -> None:
        """Initializes retriever model, ChromaDB persistent client, and BM25 index.

        Raises:
            RuntimeError: If the specified ChromaDB collection does not exist.
            FileNotFoundError: If the pickled BM25 index file is missing.
        """
        print(f"Loading retriever for '{BOOK['name']}'...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        try:
            self.collection = self.chroma_client.get_collection(name=BOOK["collection"])
        except Exception:
            raise RuntimeError(
                f"Collection '{BOOK['collection']}' not found. "
                "Please run ingestion first: python main.py --ingest"
            )

        bm25_path = BOOK["bm25_index"]
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(
                f"BM25 index '{bm25_path}' not found. "
                "Please run ingestion first: python main.py --ingest"
            )

        with open(bm25_path, "rb") as f:
            data = pickle.load(f)

        self.bm25: BM25Okapi = data["bm25"]
        self.all_chunks: List[str] = data["chunks"]
        self.all_metadata: List[Dict] = data["metadata"]
        self.all_ids: List[str] = data["ids"]
        # Fast lookup mapping from chunk ID to positional array index
        self.id_to_idx: Dict[str, int] = {cid: i for i, cid in enumerate(self.all_ids)}

        print(f"[OK] Loaded {len(self.all_chunks)} chunks.")

    def retrieve(self, query: str) -> Tuple[List[str], Dict[str, Set[int]]]:
        """Performs hybrid dense and sparse search and merges results via RRF.

        Args:
            query: User input query string.

        Returns:
            Tuple containing:
                - List of formatted text passages including source header metadata.
                - Dictionary mapping source filenames to sets of referenced page numbers.
        """
        query_emb = self.model.encode([query]).tolist()
        dense_result = self.collection.query(
            query_embeddings=query_emb,
            n_results=TOP_K_DENSE,
        )

        dense_ranks: Dict[str, int] = {}
        if dense_result["ids"]:
            for rank, cid in enumerate(dense_result["ids"][0]):
                dense_ranks[cid] = rank + 1

        tokenized_q = tokenize_greek(query)
        sparse_scores = self.bm25.get_scores(tokenized_q)
        top_sparse = np.argsort(sparse_scores)[::-1][:TOP_K_SPARSE]

        sparse_ranks: Dict[str, int] = {}
        for rank, idx in enumerate(top_sparse):
            if sparse_scores[idx] > 0:
                sparse_ranks[self.all_ids[idx]] = rank + 1

        rrf_ranking = rrf_score(dense_ranks, sparse_ranks)
        # Score threshold of 0.01 filters out candidate chunks with zero overlap in both modalities
        rrf_ranking = [(cid, s) for cid, s in rrf_ranking if s > 0.01]
        top_ids = [cid for cid, _ in rrf_ranking[:FINAL_TOP_K]]

        if not top_ids:
            return [], {}

        context_texts: List[str] = []
        sources_with_pages: Dict[str, Set[int]] = {}

        for cid in top_ids:
            idx = self.id_to_idx[cid]
            meta = self.all_metadata[idx]
            text = self.all_chunks[idx]
            source = meta["source"]
            chunk_type = meta.get("type", "text")

            page_start = meta.get("page_start")
            page_end = meta.get("page_end")
            chapter = meta.get("chapter", "-")

            if page_start and page_end:
                if page_start == page_end:
                    page_info = f"Σελίδα {page_start}"
                else:
                    page_info = f"Σελίδες {page_start}-{page_end}"
            else:
                page_info = "Σελίδα άγνωστη"

            if chunk_type == "image":
                header = (
                    f"[ΕΙΚΟΝΑ | {page_info} | Κεφάλαιο: {chapter}]\n"
                    f"Πηγή: {source}\n"
                    f"Περιεχόμενο: {text}"
                )
            else:
                header = (
                    f"[ΚΕΙΜΕΝΟ | {page_info} | Κεφάλαιο: {chapter}]\n"
                    f"Πηγή: {source}\n"
                    f"Κείμενο: {text}"
                )

            context_texts.append(header)

            if source not in sources_with_pages:
                sources_with_pages[source] = set()
            if page_start:
                sources_with_pages[source].update(
                    range(page_start, (page_end or page_start) + 1)
                )

        return context_texts, sources_with_pages