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

import numpy as np  

import json
import re
from collections import Counter
from pathlib import Path

from openai import embeddings
from pypdf import PdfReader

from backend.config import settings
from sentence_transformers import SentenceTransformer


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
        "tax", "expenses","revenue",
        "risk", "risks","Net Sales", "Net Income",
        "segment", "segments","Gross Profit",
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

        # Semantic embedding model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


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
    # Semantic Embeddings
    # --------------------------------------------------

    def generate_embedding(
        self, 
        text: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple chunks effectively.
        """

        embedding = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        return embedding.tolist()


    def cosine_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        """

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if denominator == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / denominator)

    # --------------------------------------------------
    # Section Detection
    # --------------------------------------------------

    def detect_section(self, paragraph: str) -> str:
        """
        Detect likely section headings.

        Returns:
        Heading if detected, otherwise "General".
        """

        first_line = paragraph.strip().split("\n")[0].strip()

        # Normalize whitespace
        first_line = re.sub(r"\s+", " ", first_line)

        # Ignore very long first lines (likely body text)
        if len(first_line) > 80:
            return "General"

        # ALL CAPS heading
        if first_line.isupper():
            return first_line.title()

        # Heading ending with :
        if first_line.endswith(":"):
            return first_line[:-1]

        # Title Case heading
        words = first_line.split()

        if (
            1 <= len(words) <= 10
            and all(
                word[0].isupper()
                or word.lower() in {"and", "of", "to", "for", "the", "&", "in"}
                for word in words
            )
        ):
            return first_line

        return "General"

    # --------------------------------------------------
    # Chunking
    # --------------------------------------------------

    def chunk_text(
        self,
        pages: list[dict],
        document_name: str,
        chunk_size: int = 900,
    ) -> list[dict]:
        """
        Split text into paragraph-aware chunks.

        Returns:
            [
                {
                    "page": 1,
                    "chunk_id": 0,
                    "section": self.detect_section(paragraph),
                    "document": document_name,
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
            current_section = "General"

            for paragraph in paragraphs:
                detected = self.detect_section(paragraph)
                if detected != "General":
                    current_section = detected  
                if len(current) + len(paragraph) <= chunk_size:
                    current += paragraph + "\n\n"
                else:
                    if current.strip():
                        chunks.append({
                            "chunk_id": chunk_id,
                            "page": page["page"],
                            "document": document_name,
                            "section": current_section,
                            "text": current.strip(),
                        })
                        chunk_id += 1

                    current = paragraph + "\n\n"

            if current.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "document": document_name,
                    "section": current_section,
                    "text": current.strip(),
                })

                chunk_id += 1

        return chunks

    # --------------------------------------------------
    # PDF Ingestion
    # --------------------------------------------------

    def ingest_pdf(self, pdf_path: Path) -> int:
        """
        Extract text, create chunks, generate semantic embeddings
        and save everything into JSON.

        Returns:
            Number of chunks stored.
        """

        pages = self.extract_text(pdf_path)
        chunks = self.chunk_text(pages,  pdf_path.stem,)

        if not chunks:
            print("No text extracted from PDF.")
            return 0
        
        # --------------------------------------
        # Generate embeddings for every chunk
        # --------------------------------------

        chunk_texts = [
            chunk["text"] 
            for chunk in chunks
        ]

        embeddings = self.generate_embedding(chunk_texts)

        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")

        # Attach embeddings to each chunk
        for chunk, embedding in zip(chunks, embeddings):

            chunk["embedding"] = embedding

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
        Retrieve the most relevant chunks from the latest report
        using Hybrid Semantic Retrieval.
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

        # ------------------------------------------
        # Generate embedding for user query
        # ------------------------------------------

        question_embedding = self.generate_embedding([question])[0]

        scored_chunks = []

        for chunk in chunks:

            # ------------------------------------------
            # Semantic similarity
            # ------------------------------------------

            semantic_score = self.cosine_similarity(
                question_embedding,
                chunk["embedding"],
            )

            # ------------------------------------------
            # Keyword relevance
            # ------------------------------------------

            keyword_score = self.keyword_score(
                question,
                chunk["text"],
            )

            
            # ------------------------------------------
            # Normalize keyword score
            # ------------------------------------------

            normalized_keyword = min(
                keyword_score / 20,
                1.0,
            )

            # ------------------------------------------
            # Finance bonus
            # ------------------------------------------

            finance_bonus = 0

            chunk_lower = chunk["text"].lower()

            question_lower = question.lower()
            for term in self.FINANCE_TERMS:
                if term in question_lower and term in chunk_lower:
                    finance_bonus += 0.05

            finance_bonus = min(finance_bonus, 0.50)

            #------------------------------------------
            # Numeric bonus
            # ------------------------------------------

            numeric_bonus = 0

            if re.search(r"\d", chunk["text"]):
                numeric_bonus = 0.10

            # ------------------------------------------
            # Final Hybrid Score
            # ------------------------------------------

            final_score = (
                0.65 * semantic_score +
                0.25 * normalized_keyword +
                finance_bonus +
                numeric_bonus
            )

            scored_chunks.append((final_score, chunk))

        # ------------------------------------------
        # Sort by final score
        # ------------------------------------------

        scored_chunks.sort(
            key=lambda x: x[0],
            reverse=True,
        )
        print("\nTOP RETRIEVED CHUNKS\n")

        for score, chunk in scored_chunks[:8]:
            print("=" * 60)
            print(f"Score: {score:.3f}")
            print(f"Page: {chunk['page']}")
            print(f"Section: {chunk.get('section')}")
            print(chunk["text"][:250])

        selected = []
        seen = set()

        for score, chunk in scored_chunks:

            if score <= 0:
                continue

            text = chunk["text"].strip()

            if text in seen:
                continue

            seen.add(text)

            chunk["score"] = round(score, 3)

            selected.append(chunk)

            if len(selected) >= top_k:
                break

        # ------------------------------------------
        # Fallback
        # ------------------------------------------

        if not selected:

            for chunk in chunks[:top_k]:
                chunk["score"] = 0.0

            selected = chunks[:top_k]

        # ------------------------------------------
        # Build Context
        # ------------------------------------------

        context = []

        for chunk in selected:

            context.append(
            f"""
========================================
Page: {chunk['page']}
Section: {chunk.get('section', 'General')}
Chunk: {chunk['chunk_id']}
Score: {chunk['score']:.3f}
========================================

{chunk['text']}
""".strip()
        )

        return "\n\n".join(context)

rag_agent = RAGAgent()