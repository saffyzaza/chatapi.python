"""Pydantic schemas for Accident Chat API."""
from typing import Any
from pydantic import BaseModel, Field


class AccidentChatRequest(BaseModel):
    question: str = Field(description="คำถามเชิงนโยบายหรือข้อมูลอุบัติเหตุ (ภาษาไทย)")
    province: str = Field(
        default="",
        description="ชื่อจังหวัด เช่น 'อุบลราชธานี' หรือ '' สำหรับทั้งเขต 10",
    )
    district: str = Field(
        default="",
        description="ชื่ออำเภอ หรือ '' สำหรับทุกอำเภอ",
    )
    year_start: int = Field(default=2021, description="ปีเริ่มต้น ค.ศ.")
    year_end: int = Field(default=2026, description="ปีสิ้นสุด ค.ศ.")


class AccidentChatQuickRequest(BaseModel):
    tool: str = Field(
        description=(
            "Tool: hotspot_roads | district_road_comparison | fatal_timeband | "
            "weather_accident_stats | behavior_stats | seasonal_comparison | "
            "weekend_vs_weekday | monthly_vehicle_pattern | late_night_vehicles | "
            "kpi_trend | serious_injury_ratio | top_cause_shift | "
            "district_death_vs_accident | district_summary | road_district_breakdown | "
            "province_executive_summary"
        )
    )
    province: str = Field(default="")
    district: str = Field(default="")
    road_name: str = Field(default="")
    year: int = Field(default=2024)
    year_start: int = Field(default=2021)
    year_end: int = Field(default=2026)
    top_n: int = Field(default=10)
    month1: int = Field(default=4)
    month2: int = Field(default=11)
    year1: int = Field(default=2023)
    year2: int = Field(default=2024)
    topic: str = Field(default="helmet")


class AccidentChatResponse(BaseModel):
    question: str
    answer: str = Field(description="คำตอบภาษาไทย Markdown")
    raw_data: str = Field(default="", description="ข้อมูลดิบจาก SQL tools")
    data_limitations: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    elapsed_seconds: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
