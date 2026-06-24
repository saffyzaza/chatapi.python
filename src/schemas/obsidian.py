"""Pydantic schemas for Obsidian Knowledge Vault endpoints."""
from pydantic import BaseModel, Field


class ObsidianSearchRequest(BaseModel):
    query: str = Field(..., description="คำค้นหา (ภาษาไทย)")
    province: str | None = Field(None, description="กรองตามจังหวัด")
    district: str | None = Field(None, description="กรองตามอำเภอ")
    tag: str | None = Field(None, description="กรองตาม tag")
    note_type: str | None = Field(None, description="MOC | district | report | research | policy")
    vault_id: str = Field("health_region_10", description="Vault ID")
    top_k: int = Field(5, ge=1, le=20, description="จำนวนผลลัพธ์")


class ObsidianNoteResult(BaseModel):
    note_id: str
    vault_id: str
    title: str | None
    province: str | None
    district: str | None
    note_type: str | None
    tags: list[str] = []
    source_file: str | None
    year: int | None
    snippet: str                        # matched section text (~600 chars)
    matched_section: str | None = None  # ## header where match was found
    score: float = 0.0                  # pg_trgm similarity score (0–1)
    is_graph_neighbor: bool = False     # True = surfaced via wikilink expansion


class ObsidianSearchResponse(BaseModel):
    count: int
    results: list[ObsidianNoteResult]
    query: str
    vault_id: str
    tier: str = "unknown"   # "chunk+graph" | "full_note_fallback"
    error: str | None = None


class ObsidianAskRequest(BaseModel):
    question: str = Field(..., description="คำถามภาษาไทย")
    province: str | None = Field(None, description="กรองตามจังหวัด (optional)")
    vault_id: str = Field("health_region_10")
    session_id: str | None = None


class ObsidianNoteRef(BaseModel):
    note_id: str
    title: str | None
    province: str | None
    district: str | None
    pdf_url: str | None = None


class ObsidianAskResponse(BaseModel):
    content: str                            # Markdown answer
    notes_referenced: list[ObsidianNoteRef] = []
    follow_ups: list[str] = []
    metadata: dict = {}


class ObsidianVaultInfo(BaseModel):
    vault_id: str
    name: str
    vault_path: str
    description: str | None
    note_count: int
    indexed_at: str | None


class ObsidianStatusResponse(BaseModel):
    enabled: bool
    vaults: list[ObsidianVaultInfo]
    total_notes: int


class ObsidianIndexRequest(BaseModel):
    vault_id: str = "health_region_10"


class ObsidianIndexResult(BaseModel):
    vault_id: str
    inserted: int
    updated: int
    unchanged: int
    errors: int
    total_files: int
    chunks_written: int = 0
    links_written: int = 0
    elapsed_seconds: float


class ObsidianNoteListItem(BaseModel):
    note_id: str
    title: str | None
    province: str | None
    district: str | None
    note_type: str | None
    tags: list[str] = []
    year: int | None
    source_file: str | None


class ObsidianPdfAsset(BaseModel):
    id: int
    province: str
    note_id: str | None
    filename: str
    minio_path: str
    minio_url: str
    file_size: int | None
    content_type: str
    synced_at: str | None


class ObsidianPdfSyncResult(BaseModel):
    uploaded: int
    skipped: int
    errors: int
    total_files: int
    elapsed_seconds: float


class ObsidianPdfListResponse(BaseModel):
    count: int
    province: str | None
    assets: list[ObsidianPdfAsset]


class ObsidianNotePdfPair(BaseModel):
    note_id: str
    title: str | None
    province: str | None
    district: str | None
    note_type: str | None
    tags: list[str] = []
    year: int | None
    source_file: str | None
    direct_pdfs: list[ObsidianPdfAsset] = []   # via note_id FK (exact match)
    match_type: str = "none"                    # "direct" | "none"


class ObsidianNotePdfPairsResponse(BaseModel):
    count: int
    matched: int        # notes that have ≥1 direct PDF
    unmatched: int      # notes with source_file but no PDF link
    province: str | None
    pairs: list[ObsidianNotePdfPair]
