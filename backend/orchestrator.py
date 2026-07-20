"""
Simple orchestrator.

Routes requests to the correct agent.
"""

from backend.research import research_agent
from backend.rag import rag_agent
from backend.llm import analysis_agent


class Orchestrator:

    def analyze_company(self, symbol: str):

        context = research_agent.collect(symbol)

        return analysis_agent.generate_company_analysis(context)

    def upload_report(self, file_path: str):

        chunks = rag_agent.ingest_pdf(file_path)

        return {
            "status": "success",
            "chunks": chunks,
        }

    def chat_report(self, question: str):

        context = rag_agent.retrieve(question)

        return analysis_agent.answer_report_question(
            question=question,
            context=context,
        )


orchestrator = Orchestrator()