from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # PostgreSQL
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "musyadata"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "1234"

    # OpenAI
    OPENAI_API_KEY: str = ""

    # Redis (ThaiJo cache)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Google Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    GEMINI_MODEL_PRO: str = "gemini-2.5-pro"
    REPORT_MAX_TOKENS: int = 8192

    # Gemini retry (429 RESOURCE_EXHAUSTED / 503 UNAVAILABLE)
    # ⚠️ base delay เดิม 60s ทำให้ exponential backoff (60→120→240s, cap 300s) รวมสูงสุด
    # ต่อ 1 agent step เกือบ 7 นาที — สำหรับแชท interactive ที่ผู้ใช้รอผลอยู่ ค่านี้นานเกินไป
    # มาก (Google เองแนะนำ backoff เริ่มต้นระดับวินาทีเดียวสำหรับ 503 "ลองใหม่ในไม่ช้า")
    # ลดเหลือ base 8s (พร้อม cap ใน agent_defaults._get_delay ที่ 45s) ให้ผลรวม ~56s
    # ต่อ 3 attempt แทน — ยังกัน thundering-herd ได้แต่ไม่ทำให้ผู้ใช้รอเป็นนาทีต่อ step เดียว
    GEMINI_RETRY_LIMIT: int = 3
    GEMINI_RETRY_DELAY: int = 8

    # Tavily
    TAVILY_API_KEY: str = ""
    TAVILY_MAX_RESULTS: int = 10
    TAVILY_CONTENT_CHARS: int = 1200  # เนื้อหาต่อ 1 ผลลัพธ์ที่ส่งให้ Answer Writer
    TAVILY_COUNTRY: str = "thailand"  # boost ผลลัพธ์จากไทย (ว่าง = ไม่เจาะจงประเทศ)

    # Tavily Research API (async research task → poll → report + sources)
    TAVILY_RESEARCH_MODEL: str = "mini"          # mini | pro | auto
    TAVILY_RESEARCH_OUTPUT_LENGTH: str = "short"  # short | standard | long
    TAVILY_RESEARCH_TIMEOUT: int = 180            # วินาที — เพดานเวลารอ poll ทั้งหมด
    TAVILY_RESEARCH_POLL_INTERVAL: int = 3        # วินาที — เว้นช่วงระหว่าง poll แต่ละครั้ง

    # ThaiJo Research API
    THAIJO_API_URL: str = "https://www.tci-thaijo.org/api"
    THAIJO_MAX_RESULTS: int = 5

    # PubMed (NCBI E-utilities) — ไม่บังคับ, เพิ่ม rate limit จาก 3 → 10 req/s
    NCBI_API_KEY: str = ""
    NCBI_EMAIL: str = ""

    # MinIO
    MINIO_ENDPOINT: str = "localhost"
    MINIO_PORT: int = 9000
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_USE_SSL: bool = False
    MINIO_BUCKET: str = "fileapa"
    PDF_BUCKET: str = "pdf-library"

    @property
    def minio_endpoint_url(self) -> str:
        return f"{self.MINIO_ENDPOINT}:{self.MINIO_PORT}"

    # Obsidian Knowledge Vault
    OBSIDIAN_VAULT_PATH: str = str(Path(__file__).parent / "obsidian_knowledge")
    OBSIDIAN_DEFAULT_VAULT: str = "health_region_10"
    # เพดานตัวอักษรของ context ที่โหลดเข้า Gemini — กัน ContextWindowExceededError
    # (Gemini limit 1,048,576 tokens; ไทย ~1 token/char → 500k chars ปลอดภัย เหลือ headroom)
    OBSIDIAN_MAX_CONTEXT_CHARS: int = 500000
    OBSIDIAN_ENABLED: bool = True
    OBSIDIAN_SEARCH_THRESHOLD: float = 0.1
    OBSIDIAN_RAW_PDF_PATH: str = "obsidian_raw_pdf"
    OBSIDIAN_PDF_MINIO_PREFIX: str = "obsidian-pdfs"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Obsidian / PDF Ingest
    OBSIDIAN_VAULT_PATH: str = "/obsidian_vault/musya knowlads"
    PDF_INGEST_PAGES_PER_CHUNK: int = 20  # หน้าต่อ chunk (เพิ่มจาก 10 → 20 ลด API calls)
    PDF_INGEST_MAX_PARALLEL: int = 4      # จำนวน chunk ที่รันพร้อมกัน

    @property
    def db_dsn(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
