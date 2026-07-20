# Finance Research Copilot

An AI-powered financial research assistant that performs company analysis and answers questions about annual reports using a lightweight Retrieval-Augmented Generation (RAG) pipeline.

Built with **FastAPI**, **Streamlit**, **Groq**, and **Docker**.

---

## Features

- Company Analysis using financial data and recent news
- AI-generated Equity Research Reports
- Stock Overview and Company Information
- Financial News Retrieval
- Annual Report PDF Upload
- Lightweight RAG for Report Question Answering
- Dockerized Frontend and Backend
- Markdown-formatted AI responses

---

## Tech Stack

- Python
- FastAPI
- Streamlit
- Groq API
- Alpha Vantage API
- NewsAPI
- PyPDF
- Docker

---

## Architecture

```
                +----------------------+
                |   Streamlit Frontend |
                +----------+-----------+
                           |
                           | HTTP
                           ▼
                +----------------------+
                |    FastAPI Backend   |
                +----------+-----------+
                           |
                           ▼
                  +------------------+
                  |   Orchestrator   |
                  +---+----------+---+
                      |          |
          +-----------+          +-----------+
          |                                  |
          ▼                                  ▼
 +----------------+                  +----------------+
 | ResearchAgent  |                  |    RAGAgent    |
 +----------------+                  +----------------+
          |                                  |
          ▼                                  ▼
 Alpha Vantage API                 Uploaded Annual Report
 NewsAPI                           JSON Chunks
          \                                  /
           \                                /
            +-------------+----------------+
                          |
                          ▼
                 +------------------+
                 |  AnalysisAgent   |
                 |   (Groq LLM)     |
                 +------------------+
                          |
                          ▼
                    AI-generated Response
```

---

## Retrieval-Augmented Generation (RAG)

This project implements a lightweight RAG pipeline.

Workflow:

1. Upload an annual report (PDF)
2. Extract and clean the text
3. Split the report into paragraph-aware chunks
4. Store chunks as JSON with page metadata
5. Retrieve relevant chunks using keyword-based scoring
6. Send only the retrieved context to the LLM for question answering

---

## Project Structure

```
Finance-Research-Copilot/
│
├── backend/
│   ├── main.py
│   ├── orchestrator.py
│   ├── research.py
│   ├── rag.py
│   ├── llm.py
│   ├── schemas.py
│   ├── config.py
│   └── prompts/
│
├── frontend/
│   └── app.py
│
├── data/
│   ├── uploads/
│   └── reports/
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Environment Variables

Create a `.env` file in the project root and configure the following variables:

```env
GROQ_API_KEY=your_groq_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
NEWS_API_KEY=your_news_api_key

GROQ_MODEL=llama-3.3-70b-versatile
REPORTS_FOLDER=data/reports
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/<your-username>/finance-research-copilot.git

cd finance-research-copilot
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Backend

```bash
uvicorn backend.main:app --reload
```

The FastAPI documentation is available at:

```
http://localhost:8000/docs
```

---

### Frontend

```bash
streamlit run frontend/app.py
```

The Streamlit application is available at:

```
http://localhost:8501
```

---

## Running with Docker

Build and start both the frontend and backend:

```bash
docker compose up --build
```

---

## Data Sources

This project uses the following external APIs:

- Groq API (LLM inference)
- Alpha Vantage API (Company and Stock Data)
- NewsAPI (Financial News)

Users must provide their own API keys.

---
