# research.py

from __future__ import annotations

import requests

from backend.config import settings


class ResearchAgent:

    ALPHA_URL = "https://www.alphavantage.co/query"
    NEWS_URL = "https://newsapi.org/v2/everything"

    def _alpha(self, function: str, symbol: str):
        response = requests.get(
            self.ALPHA_URL,
            params={
                "function": function,
                "symbol": symbol,
                "apikey": settings.alpha_vantage_api_key,
            },
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    def company_overview(self, symbol: str):
        return self._alpha("OVERVIEW", symbol)

    def stock_quote(self, symbol: str):
        return self._alpha("GLOBAL_QUOTE", symbol).get(
            "Global Quote",
            {},
        )

    def news(self, company: str):

        response = requests.get(
            self.NEWS_URL,
            params={
                "q": company,
                "pageSize": 5,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": settings.news_api_key,
            },
            timeout=30,
        )

        response.raise_for_status()

        return response.json().get("articles", [])

    def collect(self, symbol: str):

        overview = self.company_overview(symbol)

        company = overview.get("Name") or symbol

        return {
            "overview": overview,
            "stock": self.stock_quote(symbol),
            "news": self.news(company),
        }


research_agent = ResearchAgent()