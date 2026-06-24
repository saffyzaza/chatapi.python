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

    # Gemini retry (429 RESOURCE_EXHAUSTED)
    GEMINI_RETRY_LIMIT: int = 3
    GEMINI_RETRY_DELAY: int = 60

    # Tavily
    TAVILY_API_KEY: str = ""

    # ThaiJo Research API
    THAIJO_API_URL: str = "https://www.tci-thaijo.org/api"
    THAIJO_MAX_RESULTS: int = 5

    # MinIO
    MINIO_ENDPOINT: str = "localhost"
    MINIO_PORT: int = 9000
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_USE_SSL: bool = False
    MINIO_BUCKET: str = "fileapa"
    PDF_BUCKET: str = "pdf-library"
    PDF_INGEST_PAGES_PER_CHUNK: int = 10

    @property
    def minio_endpoint_url(self) -> str:
        return f"{self.MINIO_ENDPOINT}:{self.MINIO_PORT}"

    # Obsidian Knowledge Vault
    OBSIDIAN_VAULT_PATH: str = str(Path(__file__).parent / "obsidian_knowledge")
    OBSIDIAN_DEFAULT_VAULT: str = "health_region_10"
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
