-- Migration 027: Obsidian PDF Assets
-- Stores MinIO URLs for raw PDF/DOCX files linked to the Obsidian vault.
-- Province-level grouping: all files under obsidian_raw_pdf/{province}/ share a province tag.

CREATE TABLE IF NOT EXISTS obsidian_pdf_assets (
    id            SERIAL PRIMARY KEY,
    province      TEXT NOT NULL,          -- e.g. "มุกดาหาร" (top-level folder name)
    note_id       TEXT REFERENCES obsidian_notes(note_id) ON DELETE SET NULL,
    filename      TEXT NOT NULL,          -- original filename (with extension)
    minio_path    TEXT NOT NULL UNIQUE,   -- "obsidian-pdfs/มุกดาหาร/1.ดอนตาล.pdf"
    minio_url     TEXT NOT NULL,          -- full http URL to MinIO object
    file_size     BIGINT,                 -- bytes
    content_type  TEXT DEFAULT 'application/pdf',
    synced_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pdf_assets_province ON obsidian_pdf_assets(province);
CREATE INDEX IF NOT EXISTS idx_pdf_assets_note_id  ON obsidian_pdf_assets(note_id);

COMMENT ON TABLE obsidian_pdf_assets IS
    'Raw PDF/DOCX source files uploaded to MinIO, grouped by province. '
    'Surfaced in the Obsidian UI when viewing notes from the same province.';
