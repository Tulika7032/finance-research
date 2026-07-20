"""
RAG Agent

Handles:
- PDF ingestion
- Text extraction
- Text cleaning
- Smart paragraph-aware chunking
- Saving chunks with metadata to JSON
- Keyword-based retrieval for report QA

No vector database required.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from backend.config import settings


class RAGAgent:
    """Lightweight RAG helper using JSON storage."""

    FINANCE_TERMS = {
        "revenue", "sales", "income", "profit", "loss",
        "cash", "cashflow", "cashflows",
        "operating", "investing", "financing",
        "asset", "assets", "liability", "liabilities",
        "debt", "borrowings", "loan", "loans",
        "inventory", "goodwill", "impairment",
        "dividend", "equity", "capital",
        "margin", "ebit", "ebitda", "eps",
        "tax", "expenses",
        "risk", "risks",
        "segment", "segments",
        "share", "shares",
        "customer", "customers",
    }

    STOP_WORDS = {
        "what", "which", "when", "where", "who",
        "how", "why", "is", "are", "was", "were",
        "the", "a", "an", "of", "to", "and",
        "for", "in", "on", "at", "from", "with",
        "does", "did", "do", "be", "by",
        "company", "report"
    }

    def __init__(self) -> None:
        self.reports_dir = Path(settings.reports_folder)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # PDF Extraction
    # --------------------------------------------------

    def extract_text(self, pdf_path: Path) -> list[dict]:
        """
        Extract text page by page.

        Returns:
            [
                {
                    "page": 1,
                    "text": "..."
                }
            ]
        """

        reader = PdfReader(pdf_path)
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = self.clean_text(text)

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text,
                })

        return pages

    # --------------------------------------------------
    # Text Cleaning
    # --------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Clean noisy PDF text."""

        text = text.replace("\xa0", " ")
        text = text.replace("\t", " ")
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # --------------------------------------------------
    # Chunking
    # --------------------------------------------------

    def chunk_text(
        self,
        pages: list[dict],
        chunk_size: int = 900,
    ) -> list[dict]:
        """
        Split text into paragraph-aware chunks.

        Returns:
            [
                {
                    "page": 1,
                    "chunk_id": 0,
                    "text": "..."
                }
            ]
        """

        chunks = []
        chunk_id = 0

        for page in pages:
            paragraphs = [
                p.strip()
                for p in page["text"].split("\n\n")
                if p.strip()
            ]

            current = ""

            for paragraph in paragraphs:
                if len(current) + len(paragraph) <= chunk_size:
                    current += paragraph + "\n\n"
                else:
                    if current.strip():
                        chunks.append({
                            "chunk_id": chunk_id,
                            "page": page["page"],
                            "text": current.strip(),
                        })
                        chunk_id += 1
                        
                    current = paragraph + "\n\n"

            if current.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": current.strip(),
                })

                chunk_id += 1

        return chunks

    # --------------------------------------------------
    # PDF Ingestion
    # --------------------------------------------------

    def ingest_pdf(self, pdf_path: Path) -> int:
        """
        Extract text, create chunks and save to JSON.

        Returns:
            Number of chunks stored.
        """

        pages = self.extract_text(pdf_path)
        chunks = self.chunk_text(pages)

        output_file = self.reports_dir / f"{pdf_path.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Stored {len(chunks)} chunks")

        return len(chunks)

    # --------------------------------------------------
    # Helper Functions
    # --------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Convert text into lowercase word tokens."""

        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def keyword_score(
        self,
        question: str,
        chunk: str,
    ) -> int:
        """
        Compute a simple keyword relevance score.
        """

        q_words = [
            word
            for word in self.tokenize(question)
            if word not in self.STOP_WORDS
        ]
        c_words = self.tokenize(chunk)

        freq = Counter(c_words)
        score = 0

        for word in q_words:
            score += freq[word]

            if word in self.FINANCE_TERMS:
                score += freq[word] * 5
                
        # Boost chunks that contain important question keywords
        chunk_lower = chunk.lower()
        matched = 0
        for word in set(q_words):
            if len(word) > 3 and word in chunk_lower:
                matched += 1

        score += matched * 2
        return score
    
    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int = 8,
    ) -> str:
        """
        Retrieve the most relevant chunks from the latest report.
        """

        report_files = sorted(
            self.reports_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        if not report_files:
            return ""

        latest_report = report_files[0]

        with open(latest_report, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            return ""

        scored_chunks = []

        for chunk in chunks:

            score = self.keyword_score(
                question,
                chunk["text"],
            )

            # Small bonus if the page contains numbers
            if re.search(r"\d", chunk["text"]):
                score += 2

            scored_chunks.append((score, chunk))

        scored_chunks.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        selected = []

        seen = set()

        for score, chunk in scored_chunks:

            if score <= 0:
                continue

            text = chunk["text"].strip()

            if text in seen:
                continue

            seen.add(text)

            selected.append(chunk)

            if len(selected) >= top_k:
                break

        # Fallback if nothing matched
        if not selected:
            selected = chunks[:top_k]

        context = []

        for chunk in selected:

            context.append(
                f"""
========================
Page: {chunk['page']}
Chunk: {chunk['chunk_id']}
========================

{chunk['text']}
""".strip()
            )

        return "\n\n".join(context)


rag_agent = RAGAgent()