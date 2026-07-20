"""Environment-based configuration for Finance Research Copilot."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    # API Keys
    groq_api_key: str | None
    alpha_vantage_api_key: str | None
    news_api_key: str | None

    # GROQ
    groq_model: str

    # Uploads
    upload_folder: str

    # Reports (JSON chunks)
    reports_folder: str

    # FastAPI
    api_host: str
    api_port: int


def get_settings() -> Settings:
    return Settings(
        # API Keys
        groq_api_key=os.getenv("GROQ_API_KEY"),
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        news_api_key=os.getenv("NEWS_API_KEY"),

        # GROQ
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),

        # Uploads
        upload_folder=os.getenv(
            "UPLOAD_FOLDER",
            "data/uploads",
        ),

        # Report storage
        reports_folder=os.getenv(
            "REPORTS_FOLDER",
            "data/reports",
        ),

        # FastAPI
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
    )


settings = get_settings()