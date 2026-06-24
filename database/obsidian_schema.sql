-- Obsidian Knowledge Vault — Database Schema
-- Run once to create tables and seed the vault registry.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ── Vault registry ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS obsidian_vaults (
    vault_id    VARCHAR(100) PRIMARY KEY,
    name        TEXT         NOT NULL,
    vault_path  TEXT         NOT NULL,
    description TEXT,
    note_count  INT          NOT NULL DEFAULT 0,
    indexed_at  TIMESTAMPTZ
);

-- ── Notes store ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS obsidian_notes (
    note_id          TEXT        PRIMARY KEY,
    vault_id         VARCHAR(100) NOT NULL REFERENCES obsidian_vaults(vault_id),
    relative_path    TEXT        NOT NULL,
    title            TEXT,
    province         TEXT,
    district         TEXT,
    note_type        TEXT,
    tags             TEXT[]      DEFAULT '{}',
    source_file      TEXT,
    year             INT,
    content          TEXT,
    content_stripped TEXT,
    indexed_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_obsidian_notes_vault    ON obsidian_notes(vault_id);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_province ON obsidian_notes(province);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_type     ON obsidian_notes(note_type);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_year     ON obsidian_notes(year);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_trgm
    ON obsidian_notes USING gin(content_stripped gin_trgm_ops);

-- ── PDF assets ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS obsidian_pdf_assets (
    id           BIGSERIAL    PRIMARY KEY,
    province     TEXT         NOT NULL,
    note_id      TEXT         REFERENCES obsidian_notes(note_id),
    filename     TEXT         NOT NULL,
    minio_path   TEXT         NOT NULL UNIQUE,
    minio_url    TEXT         NOT NULL,
    file_size    BIGINT,
    content_type TEXT         NOT NULL DEFAULT 'application/pdf',
    synced_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pdf_assets_province ON obsidian_pdf_assets(province);
CREATE INDEX IF NOT EXISTS idx_pdf_assets_note_id  ON obsidian_pdf_assets(note_id);

-- ── Seed vault registry ───────────────────────────────────────────────────────
INSERT INTO obsidian_vaults (vault_id, name, vault_path, description)
VALUES (
    'health_region_10',
    'เขตสุขภาพที่ 10',
    'obsidian_knowledge',
    'ข้อมูลสุขภาพเขตสุขภาพที่ 10 — 5 จังหวัด: อุบลราชธานี ศรีสะเกษ ยโสธร อำนาจเจริญ มุกดาหาร'
)
ON CONFLICT (vault_id) DO UPDATE SET
    name        = EXCLUDED.name,
    description = EXCLUDED.description;
