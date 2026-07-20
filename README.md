# Finance Research Copilot

AI-powered financial research assistant built using:

- FastAPI
- Streamlit
- OpenAI
- ChromaDB
- Docker

## Features

- Company Analysis
- Financial News
- PDF Upload
- RAG over Annual Reports
- AI-powered Investment Insights

## Run

Backend

```bash
uvicorn backend.main:app --reload
```

Frontend

```bash
streamlit run frontend/app.py
```

Docker

```bash
docker compose up --build
```
