# RAG System - Technologies for Big Data Analytics

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system tailored for the university textbook **"Technologies for Big Data Analytics"** (Greek: *Τεχνολογίες Επεξεργασίας και Ανάλυσης Μεγάλων Δεδομένων*, Authors: A. Karakasidis, G. Koloniari, A. Gounaris, A. N. Papadopoulos, Kallipos Open Academic Editions, 2025).

This project constitutes one of the two codebase implementations developed for the Undergraduate Thesis titled **"Design and Evaluation of a Retrieval-Augmented Generation Question-Answering System for Educational Content"** (*Σχεδιασμός και Αξιολόγηση Συστήματος Ερωτοαπαντήσεων Retrieval-Augmented Generation για Εκπαιδευτικό Υλικό*) by Aliki Giagka, under the supervision of Assoc. Prof. Georgia Koloniari at the Department of Applied Informatics, University of Macedonia (July 2026).

---

## Architecture & System Features

The system implements an advanced hybrid RAG architecture specifically tuned for processing dense academic textbooks:

- **Sliding Window Chunking**: Text content is split using a word-level sliding window (`CHUNK_SIZE = 400` words, `CHUNK_OVERLAP = 100` words, `MIN_CHUNK_WORDS = 30` words). This ensures context continuity across complex paragraphs.
- **Chapter & Section Detection**: Automatic regex-based structural parsing identifies chapter titles and section headings, attaching `chapter`, `page_start`, and `page_end` metadata to every chunk.
- **Multimodal Visual Content Ingestion**: Figures, code snippets, and tables are extracted from PDF pages and described in Greek using Gemini 2.5 Flash as a Vision LLM. Descriptions are stored as independent chunks (`type: "image"`) in the vector database with their own metadata, allowing autonomous retrieval.
- **Dense Vector Retrieval**: Uses `intfloat/multilingual-e5-large-instruct` sentence embeddings stored in ChromaDB (collection: `bigdata_book`), retrieving top 8 dense candidates (`TOP_K_DENSE = 8`).
- **Sparse Keyword Retrieval**: Uses BM25Okapi with custom Greek tokenization (`tokenize_greek`: Unicode NFD normalization, lowercasing, accent and punctuation removal), retrieving top 8 sparse candidates (`TOP_K_SPARSE = 8`).
- **Hybrid Fusion via RRF**: Combines dense and sparse search rankings using Reciprocal Rank Fusion (`RRF_K = 60`), applies a score threshold (> 0.01), and selects the top 5 chunks (`FINAL_TOP_K = 5`) for prompt construction.
- **Grounded Response Generation**: Uses Google's `gemini-2.5-flash` (`temperature = 0`) with system instructions enforcing answers strictly grounded in the retrieved text, explicit page/chapter citation, and strict refusal guidelines when information is missing.
- **Dual User Interfaces**: Interactive command-line interface (CLI) and a FastAPI web server serving a single-page web UI.
- **Evaluation Pipeline**: Built-in interaction logging (`rag_evaluation_data.json`) evaluated via the LLM-as-a-Judge methodology using Gemma 3 4B across 4 key RAGAS metrics: Faithfulness, Answer Relevance, Context Precision, and Context Recall.

---

## Repository Structure

```
├── api.py                    # FastAPI web server and HTTP API endpoints
├── main.py                   # CLI entry point for interactive chat and ingestion
├── requirements.txt          # Python dependencies
├── .env.example              # Template for environment configuration
├── src/
│   ├── config.py             # Global configurations, paths, and hyperparameters
│   ├── ingest.py             # Document processing, chunking, vision API, and indexing
│   ├── llm.py                # Gemini LLM generation and system prompt logic
│   └── retriever.py          # Hybrid RRF retriever implementation (ChromaDB + BM25)
├── static/                   # Frontend Web UI static assets (HTML/CSS/JS)
├── evaluation/               # RAG evaluation scripts and Jupyter notebooks
├── google colab/             # Colab notebooks for GPU-accelerated ingestion & embeddings
└── data/                     # Source PDF material directory (data/book/)
```

---

## Prerequisites & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/USERNAME/rag-project-book.git
   cd rag-project-book
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Copy `.env.example` to `.env` and set your Google Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

---

## Usage

### 1. Document Ingestion
Document ingestion (including GPU embedding calculation and Vision LLM figure description) is optimized for Google Colab (`google colab/ingest_colab.ipynb`).

Alternatively, to run ingestion locally, place the textbook PDF into `data/book/` and run:
```bash
python main.py --ingest
```

### 2. Interactive CLI Chat
Run the terminal chat interface:
```bash
python main.py
```

### 3. FastAPI Web Application
Start the FastAPI server:
```bash
uvicorn api:app --reload
```
Open a browser and visit `http://127.0.0.1:8000`.

---

## Citation & Academic Context

If you use this codebase, please cite the underlying thesis:

```bibtex
@thesis{giagka2026rag,
  author       = {Aliki Giagka},
  title        = {Design and Evaluation of a Retrieval-Augmented Generation Question-Answering System for Educational Material},
  school       = {University of Macedonia, Department of Applied Informatics},
  year         = {2026},
  type         = {Undergraduate Thesis},
  supervisor   = {Georgia Koloniari},
  address      = {Thessaloniki, Greece}
}
```

---

## License

Distributed under the MIT License.
