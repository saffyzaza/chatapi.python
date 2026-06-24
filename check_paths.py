import sys
sys.path.insert(0, "src")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from src.config import get_settings
s = get_settings()

vault_path = Path(s.OBSIDIAN_VAULT_PATH)
if not vault_path.is_absolute():
    vault_path = Path.cwd() / vault_path

raw_pdf_root = Path(s.OBSIDIAN_VAULT_PATH).parent / s.OBSIDIAN_RAW_PDF_PATH
if not raw_pdf_root.is_absolute():
    raw_pdf_root = Path.cwd() / raw_pdf_root

print("=== Config ===")
print(f"OBSIDIAN_VAULT_PATH   = {s.OBSIDIAN_VAULT_PATH}")
print(f"OBSIDIAN_RAW_PDF_PATH = {s.OBSIDIAN_RAW_PDF_PATH}")
print(f"OBSIDIAN_ENABLED      = {s.OBSIDIAN_ENABLED}")
print()
print("=== Resolved paths ===")
print(f"vault_path            = {vault_path}")
print(f"vault_path.exists()   = {vault_path.exists()}")
if vault_path.exists():
    md_files = [p for p in vault_path.rglob("*.md") if ".obsidian" not in p.parts]
    print(f"  .md files           = {len(md_files)}")

print()
print(f"raw_pdf_root          = {raw_pdf_root}")
print(f"raw_pdf_root.exists() = {raw_pdf_root.exists()}")
if raw_pdf_root.exists():
    exts = {".pdf", ".docx", ".doc", ".pptx", ".xlsx"}
    files = [p for p in raw_pdf_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    print(f"  supported files     = {len(files)}")

print()
print("=== Actual obsidian/ folder ===")
alt_path = Path.cwd() / "obsidian" / "ข้อมูลสสจ.-20260524T153809Z-3-002" / "ข้อมูลสสจ_"
print(f"alt_path              = {alt_path}")
print(f"alt_path.exists()     = {alt_path.exists()}")
if alt_path.exists():
    exts = {".pdf", ".docx", ".doc", ".pptx", ".xlsx"}
    files = [p for p in alt_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    print(f"  supported files     = {len(files)}")
    provinces = [p.name for p in alt_path.iterdir() if p.is_dir()]
    print(f"  provinces           = {provinces}")
