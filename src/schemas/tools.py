from typing import Optional
from pydantic import BaseModel


class CompareRequest(BaseModel):
    sessionId: str
    prompt: str
    history: Optional[list] = None


class ReportRequest(BaseModel):
    sessionId: str
    prompt: str
    history: Optional[list] = None


class WorkplanRequest(BaseModel):
    sessionId: str
    prompt: str
    doc_type: str = "workplan"  # workplan | plan | policy


class DatabaseRequest(BaseModel):
    sessionId: str
    prompt: str
    attached_files: list[dict] = []  # [{id: str, name: str}]
    history: Optional[list] = None
