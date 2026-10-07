
"""Command-Line Interface (CLI) entry point for the RAG system.

This module handles command-line argument parsing to trigger either the document
ingestion pipeline or an interactive question-answering session with dataset logging for evaluation.
"""

import argparse
import json
import warnings
from typing import Dict, List, Optional, Union

# Suppress transitive dependency deprecation warnings from PyTorch and HuggingFace
warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import BOOK
from src.ingest import run_ingestion
from src.llm import generate_response
from src.retriever import Retriever


def chat() -> None:
    """Executes interactive CLI chat loop for user queries and records evaluation data."""
    print(f"\n{'='*55}")
    print(f"  RAG Βοηθός - {BOOK['name']}")
    print(f"{'='*55}")
    print("Πληκτρολογήστε 'exit' ή 'quit' για έξοδο.\n")

    try:
        retriever = Retriever()
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[Σφάλμα]: {e}")
        return

    evaluation_file = "rag_evaluation_data.json"
    evaluation_dataset: List[Dict[str, Union[str, Optional[str]]]] = []
    try:
        with open(evaluation_file, "r", encoding="utf-8") as f:
            evaluation_dataset = json.load(f)
        print(
            f"[Καταγραφή] Βρέθηκαν {len(evaluation_dataset)} προηγούμενες εγγραφές. Νέες ερωτήσεις θα προστεθούν στο τέλος."
        )
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    while True:
        try:
            query = input("\nΕρώτηση: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nΈξοδος.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("Αντίο!")
            break

        print("\nΠοιο είναι το είδος της ερώτησης; (Προαιρετικά - Πατήστε Enter για παράλειψη)")
        print("  1. Factual Recall")
        print("  2. Conceptual Understanding")
        print("  3. Multi-hop Reasoning")
        print("  4. Unanswerable")
        print("  5. Image-based")

        question_types = {
            "1": "Factual Recall",
            "2": "Conceptual Understanding",
            "3": "Multi-hop Reasoning",
            "4": "Unanswerable",
            "5": "Image-based",
        }

        question_type: Optional[str] = None
        try:
            choice = input("Επιλογή (1-5 ή Enter): ").strip()
            if choice in question_types:
                question_type = question_types[choice]
        except (EOFError, KeyboardInterrupt):
            pass

        context_texts, sources_with_pages = retriever.retrieve(query)

        if not context_texts:
            print("Δεν βρέθηκαν σχετικά αποσπάσματα για αυτή την ερώτηση.\n")
            continue

        response_text, error = generate_response(query, context_texts)

        if error:
            print(f"\n[Σφάλμα Gemini: {error}]\n")
            final_answer = f"Error: {error}"
        else:
            print(f"\n--- Απάντηση ---\n{response_text}\n")
            final_answer = response_text

        # Record QA interaction tuple for offline evaluation dataset construction
        retrieved_context_str = "\n---\n".join(context_texts)
        evaluation_record: Dict[str, Union[str, Optional[str]]] = {
            "query": query,
            "retrieved_context": retrieved_context_str,
            "generated_answer": final_answer,
        }
        if question_type:
            evaluation_record["question_type"] = question_type

        evaluation_dataset.append(evaluation_record)

    if evaluation_dataset:
        with open("rag_evaluation_data.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_dataset, f, ensure_ascii=False, indent=4)
        print(
            "\n[Καταγραφή] Τα δεδομένα αξιολόγησης αποθηκεύτηκαν στο 'rag_evaluation_data.json'"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Book RAG - CLI")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Εκτέλεση ingestion (επεξεργασία του βιβλίου PDF)",
    )
    args = parser.parse_args()

    if args.ingest:
        run_ingestion()
    else:
        chat()

