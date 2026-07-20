from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    company: str

class ChatRequest(BaseModel):
    question: str

class UploadResponse(BaseModel):
    status: str
    chunks: int

class AnalysisResponse(BaseModel):
    company: str | None
    analysis: str

class ChatResponse(BaseModel):
    question: str
    answer: str