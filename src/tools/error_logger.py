"""Agent Error Logger — captures and persists agent/tool failures to error_logs/.

Log format: one .txt file per day  →  error_logs/agent_errors_YYYY-MM-DD.txt
Each entry is a human-readable block separated by a divider line.

Error types classified automatically:
  auth_error      — 403 / API key leaked / PERMISSION_DENIED
  quota_error     — 429 / RESOURCE_EXHAUSTED / quota exceeded
  timeout         — execution or network timeout
  empty_response  — agent returned None or empty string
  tool_error      — tool call inside agent failed
  unknown         — anything else
"""
import json
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

ErrorType = Literal["auth_error", "quota_error", "timeout", "empty_response", "tool_error", "unknown"]

# Root of error_logs/ is next to main.py  →  python-ai/error_logs/
ERROR_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "error_logs"

_DIVIDER = "=" * 72


# ── Classification ─────────────────────────────────────────────────────────────

def classify_error(message: str) -> ErrorType:
    m = (message or "").lower()
    if any(k in m for k in ("403", "permission_denied", "leaked", "api key", "unauthorized")):
        return "auth_error"
    if any(k in m for k in ("429", "quota", "resource_exhausted", "rate limit")):
        return "quota_error"
    if any(k in m for k in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(k in m for k in ("empty", "none or empty", "empty response")):
        return "empty_response"
    if any(k in m for k in ("tool", "function call", "tool_call")):
        return "tool_error"
    return "unknown"


# ── Write ──────────────────────────────────────────────────────────────────────

def log_agent_error(
    error_message: str,
    agent_name: str = "",
    step: str = "",
    domain: str = "",
    prompt: str = "",
    session_id: str = "",
    attempt: int = 0,
) -> None:
    """Append one agent error block to today's .txt log file.

    Safe to call from any thread — file I/O uses append mode.
    """
    ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = ERROR_LOG_DIR / f"agent_errors_{today}.txt"

    error_type = classify_error(error_message)
    entry_id   = str(uuid.uuid4())[:8]
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    block = (
        f"\n{_DIVIDER}\n"
        f"ERROR ID  : {entry_id}\n"
        f"TIMESTAMP : {timestamp}\n"
        f"SESSION   : {session_id or '-'}\n"
        f"AGENT     : {agent_name or '-'}\n"
        f"STEP      : {step or '-'}\n"
        f"DOMAIN    : {domain or '-'}\n"
        f"TYPE      : {error_type}\n"
        f"ATTEMPT   : {attempt}\n"
        f"MESSAGE   : {(error_message or '')[:600]}\n"
        f"PROMPT    : {(prompt or '')[:250]}\n"
        f"{_DIVIDER}\n"
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(block)


# ── Read / Parse ───────────────────────────────────────────────────────────────

def _parse_txt_file(path: Path) -> list[dict]:
    """Parse a .txt log file back into list of dict entries."""
    entries: list[dict] = []
    text = path.read_text(encoding="utf-8")

    # Split on divider lines — blocks are pairs of divider...content...divider
    raw_blocks = re.split(r"={60,}", text)

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        entry: dict = {}
        for line in block.splitlines():
            if " : " in line:
                key, _, value = line.partition(" : ")
                field = key.strip().lower()
                val   = value.strip()
                mapping = {
                    "error id":  "id",
                    "timestamp": "timestamp",
                    "session":   "session_id",
                    "agent":     "agent_name",
                    "step":      "step",
                    "domain":    "domain",
                    "type":      "error_type",
                    "attempt":   "attempt",
                    "message":   "error_message",
                    "prompt":    "prompt_snippet",
                }
                if field in mapping:
                    entry[mapping[field]] = int(val) if field == "attempt" and val.isdigit() else val
        if entry.get("id"):
            entries.append(entry)

    return entries


def read_all_errors(days: int = 7) -> list[dict]:
    """Return all error entries from the last N days, newest first."""
    if not ERROR_LOG_DIR.exists():
        return []

    entries: list[dict] = []
    for log_file in sorted(ERROR_LOG_DIR.glob("agent_errors_*.txt"), reverse=True)[:days]:
        try:
            entries.extend(_parse_txt_file(log_file))
        except Exception:
            pass

    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate_errors(entries: list[dict]) -> dict:
    """Summarise error list into counts and top offenders."""
    by_type:   Counter = Counter()
    by_agent:  Counter = Counter()
    by_step:   Counter = Counter()
    by_domain: Counter = Counter()

    for e in entries:
        by_type[e.get("error_type", "unknown")] += 1
        if e.get("agent_name") and e["agent_name"] != "-":
            by_agent[e["agent_name"]] += 1
        if e.get("step") and e["step"] != "-":
            by_step[e["step"]] += 1
        if e.get("domain") and e["domain"] != "-":
            by_domain[e["domain"]] += 1

    return {
        "total":     len(entries),
        "by_type":   dict(by_type.most_common()),
        "by_agent":  dict(by_agent.most_common(10)),
        "by_step":   dict(by_step.most_common()),
        "by_domain": dict(by_domain.most_common()),
        "latest":    entries[:5],
    }


# ── Clear ─────────────────────────────────────────────────────────────────────

def clear_all_logs() -> int:
    """Delete all .txt log files. Returns number of files deleted."""
    if not ERROR_LOG_DIR.exists():
        return 0
    deleted = 0
    for f in ERROR_LOG_DIR.glob("agent_errors_*.txt"):
        f.unlink()
        deleted += 1
    return deleted
