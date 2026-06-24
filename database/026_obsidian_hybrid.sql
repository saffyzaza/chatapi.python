-- Migration 026: Obsidian Hybrid Search — section chunks + wikilink graph
-- Requires: 025_obsidian_knowledge.sql (obsidian_notes table must exist)

-- ── 1. Section-level chunks ───────────────────────────────────────────────────
-- One row per ## header block per note.
-- Replaces "first 600 chars" snippet with the actual matching section.

CREATE TABLE IF NOT EXISTS obsidian_note_chunks (
    chunk_id      TEXT PRIMARY KEY,   -- "{note_id}::{chunk_index}"
    note_id       TEXT NOT NULL REFERENCES obsidian_notes(note_id) ON DELETE CASCADE,
    chunk_index   INT  NOT NULL,      -- 0 = intro block (before first ##)
    header        TEXT,               -- ## heading text; NULL for intro chunk
    chunk_content TEXT NOT NULL,      -- section body (markdown, no frontmatter)
    indexed_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_obsidian_chunks_note_id
    ON obsidian_note_chunks(note_id);

-- GIN trigram index on chunk content — same engine as full-note search
CREATE INDEX IF NOT EXISTS idx_obsidian_chunks_content_trgm
    ON obsidian_note_chunks USING GIN(chunk_content gin_trgm_ops);

-- ── 2. Wikilink graph edges ───────────────────────────────────────────────────
-- One row per resolved [[wikilink]] found in a note.
-- Enables one-hop neighbor expansion during search.

CREATE TABLE IF NOT EXISTS obsidian_note_links (
    source_note_id TEXT NOT NULL REFERENCES obsidian_notes(note_id) ON DELETE CASCADE,
    target_note_id TEXT NOT NULL REFERENCES obsidian_notes(note_id) ON DELETE CASCADE,
    link_text      TEXT,              -- display text after | in [[target|display]]
    PRIMARY KEY (source_note_id, target_note_id)
);

-- Forward traversal: "what does this note link TO?"
CREATE INDEX IF NOT EXISTS idx_obsidian_links_source
    ON obsidian_note_links(source_note_id);

-- Reverse traversal (backlinks): "what notes link TO this note?"
CREATE INDEX IF NOT EXISTS idx_obsidian_links_target
    ON obsidian_note_links(target_note_id);

COMMENT ON TABLE obsidian_note_chunks IS
    'Section-level chunks split by ## headers. Enables precise snippet matching.';
COMMENT ON TABLE obsidian_note_links IS
    'Resolved [[wikilink]] edges for one-hop graph expansion during search.';
