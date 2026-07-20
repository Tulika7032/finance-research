from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.orchestrator import orchestrator
from backend.schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    UploadResponse,
)

app = FastAPI(title="Finance Research Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalyzeRequest):
    return orchestrator.analyze_company(request.company)


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    path = Path(settings.upload_folder) / file.filename

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(await file.read())

    return orchestrator.upload_report(path)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return orchestrator.chat_report(request.question)