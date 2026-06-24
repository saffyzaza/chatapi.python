-- Migration 025: Obsidian Knowledge Vault indexing
-- Supports multi-vault, Thai full-text search via pg_trgm

-- Enable trigram extension (handles Thai partial-match without tokenizer)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Vault registry (one row per knowledge base folder)
CREATE TABLE IF NOT EXISTS obsidian_vaults (
    vault_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    vault_path  TEXT NOT NULL,          -- absolute or relative FS path
    description TEXT,
    note_count  INT DEFAULT 0,
    indexed_at  TIMESTAMP,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Notes index (one row per .md file)
CREATE TABLE IF NOT EXISTS obsidian_notes (
    note_id          TEXT PRIMARY KEY,  -- vault_id + "::" + relative_path
    vault_id         TEXT NOT NULL REFERENCES obsidian_vaults(vault_id) ON DELETE CASCADE,
    relative_path    TEXT NOT NULL,     -- path from vault root, e.g. อุบลราชธานี/อ.เมือง/note.md
    title            TEXT,
    province         TEXT,
    district         TEXT,
    note_type        TEXT,              -- MOC | district | report | research | policy
    tags             TEXT[] DEFAULT '{}',
    source_file      TEXT,              -- original PDF source from frontmatter
    year             INT,               -- extracted from filename or frontmatter
    content          TEXT,              -- full markdown (incl. frontmatter)
    content_stripped TEXT,             -- markdown without YAML frontmatter
    indexed_at       TIMESTAMP DEFAULT NOW()
);

-- Indexes for search and filtering
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_content_trgm
    ON obsidian_notes USING GIN(content_stripped gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_title_trgm
    ON obsidian_notes USING GIN(title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_province
    ON obsidian_notes(province);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_district
    ON obsidian_notes(district);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_vault
    ON obsidian_notes(vault_id);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_tags
    ON obsidian_notes USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_type
    ON obsidian_notes(note_type);
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_year
    ON obsidian_notes(year);

-- Seed default vault (Health Region 10)
INSERT INTO obsidian_vaults (vault_id, name, vault_path, description)
VALUES (
    'health_region_10',
    'เขตสุขภาพที่ 10',
    'obsidian_knowledge',
    'ข้อมูลสุขภาพเขตสุขภาพที่ 10 — 5 จังหวัด: อุบลราชธานี ศรีสะเกษ ยโสธร อำนาจเจริญ มุกดาหาร'
)
ON CONFLICT (vault_id) DO NOTHING;

COMMENT ON TABLE obsidian_vaults IS 'Registry of Obsidian knowledge vault sources';
COMMENT ON TABLE obsidian_notes  IS 'Indexed Markdown notes from Obsidian vaults (Thai FTS via pg_trgm)';
