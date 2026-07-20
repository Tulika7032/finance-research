"""
Analysis Agent

Responsible for:
- Company analysis
- Report question answering
- Loading prompts
- Communicating with GROQ
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from backend.config import settings


PROMPTS_DIR = Path(__file__).parent / "prompts"

SYSTEM_PROMPT = (
    PROMPTS_DIR / "system_prompt.txt"
).read_text(encoding="utf-8")

ANALYSIS_PROMPT = (
    PROMPTS_DIR / "analysis_prompt.txt"
).read_text(encoding="utf-8")


class AnalysisAgent:
    """LLM-powered analysis agent."""

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    # --------------------------------------------------
    # Helper
    # --------------------------------------------------

    def _chat(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=settings.groq_model,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response.choices[0].message.content or ""

    # --------------------------------------------------
    # Company Analysis
    # --------------------------------------------------

    def generate_company_analysis(
        self,
        context: dict,
    ) -> dict:

        overview = context.get("overview", {})
        stock = context.get("stock", {})
        news = context.get("news", [])

        prompt = f"""
{ANALYSIS_PROMPT}

Company Information

{json.dumps(overview, indent=2)}

Stock Information

{json.dumps(stock, indent=2)}

Recent News

{json.dumps(news, indent=2)}

Generate a professional equity research report.

Use Markdown.

Include:

# Executive Summary

# Financial Health

# SWOT Analysis

# Growth Opportunities

# Risks

# Investment Recommendation
"""

        analysis = self._chat(prompt)

        return {
            "company": overview.get("Name"),
            "analysis": analysis,
        }

    # --------------------------------------------------
    # Report Chat
    # --------------------------------------------------

    def answer_report_question(
        self,
        question: str,
        context: str,
    ) -> dict:
        """
        Answer questions using only the uploaded annual report.
        """

        prompt = f"""
You are a professional financial analyst answering questions about a company's annual report.

Use ONLY the information provided in the context.

Instructions:
- Do not make up facts or numbers.
- If the answer is not present in the context, say:
  "The uploaded report does not contain enough information to answer this question."
- Quote financial figures exactly as they appear.
- If page numbers are available, mention them.
- Keep the answer concise but complete.
- Use bullet points if they improve readability.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = self._chat(prompt)

        return {
            "question": question,
            "answer": answer,
        }


analysis_agent = AnalysisAgent()