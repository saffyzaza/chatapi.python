"""Pydantic schemas for the CSV analyze endpoints."""
from typing import Optional
from pydantic import BaseModel


class HistoryMessage(BaseModel):
    model_config = {"extra": "allow"}
    role: str
    text: str


class AnalyzeRequest(BaseModel):
    sessionId: str
    prompt: str
    history: Optional[list[HistoryMessage]] = None
    mode: str = "normal"  # "normal" | "tavily"
