"""MinIO tools — list/read CSV files + Python executor for CSV analysis agents."""
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse

import minio as minio_lib
import pandas as pd
from crewai.tools import tool

from src.config import get_settings

# ── Client ────────────────────────────────────────────────────────────────────

def _get_client() -> minio_lib.Minio:
    s = get_settings()
    return minio_lib.Minio(
        s.minio_endpoint_url,
        access_key=s.MINIO_ACCESS_KEY,
        secret_key=s.MINIO_SECRET_KEY,
        secure=s.MINIO_USE_SSL,
    )


def _bucket() -> str:
    return get_settings().MINIO_BUCKET


# ── File map (object_name → display_name cache) ───────────────────────────────

_file_map: dict[str, str] = {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def list_csv_files_impl(prefix: str = "") -> str:
    global _file_map
    client = _get_client()
    bucket = _bucket()
    try:
        objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
        files: list[str] = []

        for obj in objects:
            obj_name = obj.object_name

            if obj_name.endswith(".csv"):
                _file_map[obj_name] = obj_name
                files.append(f"[ID:{obj_name}] {obj_name}")
                continue

            try:
                stat = client.stat_object(bucket, obj_name)
                meta = stat.metadata or {}
                meta_lc = {k.lower(): v for k, v in meta.items()}
                ext = meta_lc.get("x-amz-meta-extension", "").strip().lower()
                name_enc = meta_lc.get("x-amz-meta-name", "").strip()
                orig_name = urllib.parse.unquote(name_enc) if name_enc else ""

                if ext == "csv" or orig_name.lower().endswith(".csv"):
                    display = orig_name if orig_name else obj_name
                    _file_map[orig_name] = obj_name
                    _file_map[obj_name] = obj_name
                    files.append(f"[ID:{obj_name}] {display}")
            except Exception:
                pass

        return "\n".join(files) if files else "No CSV files found in bucket"
    except Exception as exc:
        return f"Error listing files: {exc}"


def resolve_file_id(agent_output: str) -> str:
    global _file_map
    stripped = agent_output.strip()
    if not stripped or stripped.startswith("[Agent error:") or stripped == "None":
        return ""

    m = re.search(r'\[ID:([^\]]+)\]', agent_output)
    if m:
        return m.group(1).strip()

    for key, fid in _file_map.items():
        if key and key in agent_output:
            return fid

    m2 = re.search(r'\b(\d{6})\b', agent_output)
    if m2:
        return m2.group(1)

    token = stripped.split()[0] if stripped else ""
    if token.startswith("[") or len(token) < 4:
        return ""
    return token


def fallback_find_file(prompt: str, domain_prefix: str = "") -> str:
    """Keyword-based file finder — used when agent fails."""
    files_text = list_csv_files_impl(domain_prefix)
    if not files_text or files_text.startswith("No CSV") or files_text.startswith("Error"):
        files_text = list_csv_files_impl("")
    if not files_text or files_text.startswith("No CSV") or files_text.startswith("Error"):
        return ""
    lines = [ln.strip() for ln in files_text.split("\n") if ln.strip()]
    if not lines:
        return ""
    prompt_words = set(re.sub(r"[^\w\s]", " ", prompt.lower()).split())

    def score(line: str) -> int:
        line_l = line.lower()
        return sum(1 for w in prompt_words if len(w) > 2 and w in line_l)

    return max(lines, key=score)


def read_csv_schema_impl(file_ref: str) -> str:
    file_id = resolve_file_id(file_ref)
    display_name = next(
        (k for k, v in _file_map.items() if v == file_id and k != file_id),
        file_id,
    )
    client = _get_client()
    try:
        resp = client.get_object(_bucket(), file_id)
        df = pd.read_csv(io.BytesIO(resp.read()))
        return json.dumps(
            {
                "file_id": file_id,
                "file_name": display_name,
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample": df.head(3).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return f"Error reading schema for '{file_id}': {exc}"


def minio_preamble() -> str:
    s = get_settings()
    endpoint = s.minio_endpoint_url
    access_key = s.MINIO_ACCESS_KEY
    secret_key = s.MINIO_SECRET_KEY
    secure = s.MINIO_USE_SSL
    bucket = s.MINIO_BUCKET
    return (
        f"import pandas as pd, io, minio as _m\n"
        f"_c = _m.Minio('{endpoint}', access_key='{access_key}', "
        f"secret_key='{secret_key}', secure={secure})\n"
        f"def load_csv(path): return pd.read_csv(io.BytesIO(_c.get_object('{bucket}', path).read()))\n"
        # ── Analysis helpers available to generated code ──────────────────────
        f"def pct_rank(s):\n"
        f"    \"\"\"Percentile rank 0-100, NaN → 0.\"\"\"\n"
        f"    return s.rank(pct=True, na_option='bottom') * 100\n"
        f"def composite_score(*series):\n"
        f"    \"\"\"Mean percentile rank across series — higher = worse (Red Zone).\"\"\"\n"
        f"    import numpy as _np\n"
        f"    return _np.nanmean([pct_rank(s).values for s in series], axis=0)\n"
    )


def exec_python(code: str, timeout: int = 90) -> str:
    """Execute Python code. timeout=90s (single file) or 180s (multi-file)."""
    full = minio_preamble() + "\n" + code
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
        f.write(full)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout
        if result.stderr:
            out += "\nSTDERR:\n" + result.stderr
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out ({timeout}s)"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── CrewAI Tools ──────────────────────────────────────────────────────────────

@tool("list_csv_files")
def list_csv_files(prefix: str = "") -> str:
    """List all CSV files in MinIO storage. Each line format: [ID:xxxxxx] filename.csv"""
    return list_csv_files_impl(prefix)


@tool("read_csv_schema")
def read_csv_schema(file_path: str) -> str:
    """Read schema, column names, data types, and sample rows of a CSV file from MinIO."""
    return read_csv_schema_impl(file_path)


@tool("execute_python_code")
def execute_python_code(code: str) -> str:
    """Execute Python/Pandas code. Use load_csv(file_id) to load CSV files from MinIO."""
    return exec_python(code)
