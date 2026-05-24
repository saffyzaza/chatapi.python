"""Pydantic schemas for ThaiJo research pipeline."""
from typing import Optional
from pydantic import BaseModel


class ThaiJoRequest(BaseModel):
    sessionId: str
    prompt: str
    doc_type: str = "policy"   # policy | plan | workplan


class ThaiJoGenerateRequest(BaseModel):
    """Request to generate a structured report from already-fetched articles."""
    sessionId: str = ""
    query: str
    articles_text: str
    doc_type: str = "policy"   # policy | plan | workplan
    topic_plan: str = ""       # user-selected topics + notes (optional)


class ThaiJoTopicsRequest(BaseModel):
    """Request to generate topic heading suggestions from articles."""
    query: str
    articles_text: str
    doc_type: str = "policy"


class ThaiJoArticle(BaseModel):
    """One article returned by ThaiJo API."""
    pdf_url: str = ""
    summary: str = ""
    reference: str = ""


class ThaiJoReportJson(BaseModel):
    """Structured report JSON produced by the Report Generator agent."""
    title: str
    subtitle: str = ""
    journal_name: str = ""
    volume_info: str = ""
    authors: list[str] = []
    affiliations: list[str] = []
    corresponding: str = ""
    abstract_th: str
    abstract_en: str = ""
    keywords_th: list[str] = []
    keywords_en: list[str] = []
    introduction: list[str] = []
    methods: list[str] = []
    results: list[dict] = []          # [{heading, paragraphs: []}]
    table_head: list[str] = []
    table_rows: list[list[str]] = []
    discussion: list[str] = []
    recommendation: list[str] = []
    fig1_cap: str = ""
    fig2_cap: str = ""
    references: list[str] = []
    source_count: int = 0
