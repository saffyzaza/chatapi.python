"""Pydantic schemas for PubMed research pipeline."""
from pydantic import BaseModel


class PubMedRequest(BaseModel):
    sessionId: str = ""
    prompt: str
    retmax: int = 10
