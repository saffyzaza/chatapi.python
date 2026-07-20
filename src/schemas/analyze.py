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
    mode: str = "normal"   # "normal" | "tavily" | "stats" | "obsidian" | "multi"
    tools: list[str] = []  # selected tool ids for multi-tool mode
    # ชนิดเอกสารที่ผู้ใช้เลือกไว้ล่วงหน้าตอนกดปุ่ม "สร้างรายงาน" (policy/plan/workplan)
    # — ใช้เฉพาะ mode=report-gather เพื่อ echo กลับใน event "final" ให้ frontend
    # ข้ามขั้นตอนเลือกประเภทเอกสารใน wizard ไปเริ่มสร้างหัวข้อได้ทันทีที่ข้อมูลพร้อม
    doc_type: Optional[str] = None
    # ชื่อแหล่งข้อมูลที่ต้องการ "ลองใหม่" เดี่ยว ๆ (obsidian/stats/thaijo/pubmed/tavily)
    # — ใช้เฉพาะ mode=report-gather-retry เมื่อผู้ใช้กดปุ่มลองใหม่บน badge ที่พัง
    retry_source: Optional[str] = None
    # ชื่อเรื่องสั้นๆ ที่ผู้ใช้พิมพ์จริง (แยกจาก prompt ซึ่งอาจถูกเสริมด้วยหัวข้อที่เลือก
    # ไว้ล่วงหน้าจนยาวมาก) — ใช้เป็น reportTitle ใน event "final" กันชื่อรายงานยาวเฟื้อย
    report_title: Optional[str] = None
