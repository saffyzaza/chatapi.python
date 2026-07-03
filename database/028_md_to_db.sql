-- Migration 028: Store Markdown notes directly in DB (replace filesystem storage)
-- Adds columns needed for DB-native vault management

-- Add file_id link back to MinIO PDF source
ALTER TABLE obsidian_notes
  ADD COLUMN IF NOT EXISTS file_id       TEXT,
  ADD COLUMN IF NOT EXISTS chunk_index   INT DEFAULT 0,
  ADD COLUMN IF NOT EXISTS is_index      BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS updated_at    TIMESTAMP DEFAULT NOW();

-- Index to look up all notes belonging to a PDF upload
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_file_id
  ON obsidian_notes(file_id);

-- Index for ordering chunks within a document
CREATE INDEX IF NOT EXISTS idx_obsidian_notes_chunk
  ON obsidian_notes(file_id, chunk_index);

COMMENT ON COLUMN obsidian_notes.file_id     IS 'MinIO object key (pdf-library bucket) that produced this note';
COMMENT ON COLUMN obsidian_notes.chunk_index IS '0-based chunk position within the source PDF (0 = index/cover note)';
COMMENT ON COLUMN obsidian_notes.is_index    IS 'TRUE for the top-level INDEX note; FALSE for chunk content notes';
COMMENT ON COLUMN obsidian_notes.updated_at  IS 'Last time content was modified via vault editor';
