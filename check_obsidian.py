import sys, os
sys.path.insert(0, "src")

import psycopg2
from pathlib import Path

out = []

try:
    conn = psycopg2.connect(host="localhost", dbname="musyadata", user="postgres", password="1234")
    cur = conn.cursor()

    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'obsidian%' ORDER BY table_name")
    tables = [r[0] for r in cur.fetchall()]
    out.append(f"Tables: {tables}")

    if "obsidian_vaults" in tables:
        cur.execute("SELECT vault_id, note_count, indexed_at FROM obsidian_vaults")
        vaults = cur.fetchall()
        out.append(f"Vaults: {vaults}")

    if "obsidian_notes" in tables:
        cur.execute("SELECT COUNT(*) FROM obsidian_notes")
        cnt = cur.fetchone()[0]
        out.append(f"Notes count: {cnt}")

    conn.close()
except Exception as e:
    out.append(f"DB Error: {e}")

from src.config import get_settings
s = get_settings()
vault_path = Path(s.OBSIDIAN_VAULT_PATH)
out.append(f"OBSIDIAN_VAULT_PATH = {vault_path}")
out.append(f"Vault exists: {vault_path.exists()}")
if vault_path.exists():
    md_files = [p for p in vault_path.rglob("*.md") if ".obsidian" not in p.parts]
    out.append(f"MD files found: {len(md_files)}")

with open(r"h:\chatpython\chat\chatapi.python\check_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Done — see check_result.txt")
