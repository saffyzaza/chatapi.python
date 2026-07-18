"""PubMed Research Agent — NCBI E-utilities + Gemini keyword extraction.

Flow:
  [Step 1] Keyword Extractor — Gemini: แปลงคำถามไทย/อังกฤษ → English PubMed query
  [Step 2] PubMed Fetcher    — esearch (PMID list) → efetch (abstract + PMC id)

ผลลัพธ์จำกัดเฉพาะบทความที่มี free full text บน PubMed Central (filter คงที่ ห้ามเอาออก):
  free full text[Filter] AND full text[Filter] AND pubmed pmc[sb]

SSE events → queue:
  {"type": "agent_start", "step": "keyword", "agentName": "Keyword Extractor"}
  {"type": "agent_done",  "step": "keyword", "result": "keyword: \"...\""}
  {"type": "agent_start", "step": "fetcher", "agentName": "PubMed Fetcher"}
  {"type": "agent_done",  "step": "fetcher", "result": "พบ N บทความ", "articleCount": N}
  {"type": "text_stream_start", "articleCount": N}
  {"type": "text_chunk", "text": "...chunk..."}
  {"type": "final", "message": "...", "textResult": "...", "articlesText": "...",
   "reportTitle": "...", "articleCount": N, "agentSteps": [...]}
"""
import asyncio
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx
import litellm

from src.tools.error_logger import log_agent_error

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# filter คงที่ — ห้ามเอาออก (เหมือน pubmed/pubmed_ffrft.py ต้นฉบับ):
#   free full text[Filter] = มี full text ให้อ่านฟรี
#   full text[Filter]      = มี full text
#   pubmed pmc[sb]         = จำกัดเฉพาะบทความที่อยู่ใน PubMed Central จริง
FILTERS = ["free full text[Filter]", "full text[Filter]", "pubmed pmc[sb]"]
PMC_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
TIMEOUT = 30


def _build_term(term: str) -> str:
    return " AND ".join([term, *FILTERS])


def _common_params(api_key: str | None, email: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {"db": "pubmed", "tool": "chatapi-pubmed"}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    return params


def _search_pmids(term: str, retmax: int, api_key: str | None, email: str | None) -> list[str]:
    params = _common_params(api_key, email)
    params.update(term=_build_term(term), retmode="json", retmax=retmax)
    resp = httpx.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _abstract_text(article: ET.Element) -> str:
    parts: list[str] = []
    for node in article.findall(".//Abstract/AbstractText"):
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        label = node.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return "\n\n".join(parts)


def _pmc_id(article: ET.Element) -> str | None:
    for aid in article.findall(".//ArticleIdList/ArticleId"):
        if aid.get("IdType") == "pmc" and aid.text:
            return aid.text.strip()
    return None


def _authors_text(article: ET.Element) -> str:
    """รวมรายชื่อผู้แต่งจาก <AuthorList> เป็น "ชื่อ นามสกุล, ชื่อ นามสกุล, ..." —
    ถ้าเกิน 6 คน ตัดเหลือ 6 คนแรก + "et al." ตามธรรมเนียมการอ้างอิงทั่วไป"""
    names: list[str] = []
    for author in article.findall(".//Article/AuthorList/Author"):
        last = author.findtext("LastName")
        fore = author.findtext("ForeName") or author.findtext("Initials")
        if last and fore:
            names.append(f"{fore} {last}")
        elif last:
            names.append(last)
        else:
            collective = author.findtext("CollectiveName")
            if collective:
                names.append(collective)
    if not names:
        return ""
    if len(names) > 6:
        return ", ".join(names[:6]) + ", et al."
    return ", ".join(names)


def _text_of(article: ET.Element, path: str) -> str | None:
    node = article.find(path)
    return "".join(node.itertext()).strip() if node is not None else None


def _fetch_details(pmids: list[str], api_key: str | None, email: str | None) -> list[dict]:
    if not pmids:
        return []
    params = _common_params(api_key, email)
    params.update(id=",".join(pmids), rettype="abstract", retmode="xml")
    resp = httpx.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    results = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _text_of(article, ".//MedlineCitation/PMID")
        pmcid = _pmc_id(article)
        results.append({
            "pmid": pmid,
            "pmcid": pmcid,
            "title": _text_of(article, ".//Article/ArticleTitle"),
            "authors": _authors_text(article),
            "journal": _text_of(article, ".//Article/Journal/Title"),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            "abstract": _abstract_text(article),
            "pmc_free_full_text": PMC_URL.format(pmcid=pmcid) if pmcid else None,
        })
    return results


def search_pubmed(term: str, retmax: int, api_key: str | None = None, email: str | None = None) -> list[dict]:
    """esearch → efetch — คืน list ของบทความ (title/journal/abstract/links)"""
    pmids = _search_pmids(term, retmax, api_key, email)
    return _fetch_details(pmids, api_key, email)


_GENERIC_GEO_TERMS = {
    "thailand", "northeast thailand", "north east thailand", "northeastern thailand",
    "southeast asia", "south east asia", "mekong region", "mekong subregion", "asean",
}


def _strip_generic_geo_terms(query: str) -> str:
    """ตัดคำทั่วไปที่กว้างเกินไป (เช่น "Thailand", "Northeast Thailand") ออกจากกลุ่ม OR ใน
    query — ถ้ากลุ่มไหนตัดแล้วว่างเปล่า (ทุกตัวเลือกเป็นคำทั่วไปหมด) ให้คงกลุ่มเดิมไว้ (ไม่มี
    อะไรเฉพาะเจาะจงกว่าให้เลือก) คืน query ที่แคบกว่าเดิม (หรือเดิมถ้าไม่มีอะไรให้ตัด)"""

    def replace_group(m: re.Match) -> str:
        alts = [a.strip() for a in m.group(1).split(" OR ") if a.strip()]
        specific = [a for a in alts if a.lower() not in _GENERIC_GEO_TERMS]
        if not specific or len(specific) == len(alts):
            return m.group(0)
        if len(specific) == 1:
            return specific[0]
        return "(" + " OR ".join(specific) + ")"

    return re.sub(r"\(([^()]*)\)", replace_group, query)


def _progressive_and_search(
    query: str, retmax: int, api_key: str | None, email: str | None,
) -> tuple[list[dict], str]:
    """ค้นด้วย query เต็มก่อน ถ้าไม่พบบทความเลย (0 ผล) ให้ตัด term ท้ายออกทีละตัวแล้วค้นใหม่
    จนกว่าจะพบหรือเหลือ term เดียว — ป้องกัน query ที่ AND หลาย MeSH term มากเกินไป (เช่น
    โรค + ประเด็น + ชื่อประเทศ) จนไม่มีบทความไหนตรงครบทุกเงื่อนไขพร้อมกัน (ผลลัพธ์ 0 บทความ)
    คืน (articles, query ที่ใช้จริง)"""
    terms = [t.strip() for t in re.split(r"\s+AND\s+", query) if t.strip()] or [query]
    for n in range(len(terms), 0, -1):
        attempt = " AND ".join(terms[:n])
        articles = search_pubmed(attempt, retmax, api_key, email)
        if articles:
            return articles, attempt
    return [], query


def search_pubmed_progressive(
    query: str, retmax: int, api_key: str | None = None, email: str | None = None,
) -> tuple[list[dict], str]:
    """ค้นสองรอบ: (1) ลองด้วย query ที่ตัดคำทั่วไปกว้างๆ (เช่น "Northeast Thailand") ออกจาก
    กลุ่ม OR ก่อน — เฉพาะเจาะจงกว่า (เช่น เหลือแค่ชื่อจังหวัด) (2) ถ้าได้บทความไม่ครบ retmax
    ให้ค้นด้วย query เต็ม (มีคำกว้างๆ ด้วย) เพิ่มเติมเพื่อเติมให้ครบ โดยไม่เอาบทความซ้ำ

    เหตุผล: ถ้าค้นด้วย query เต็มตรงๆ ตั้งแต่แรก PubMed จะถือว่าบทความที่ตรงแค่คำกว้างๆ อย่าง
    "Northeast Thailand" (มีอยู่เพียบ) เท่ากับบทความที่ตรงชื่อจังหวัดเป๊ะ (มีน้อยกว่ามาก) และ
    default sort ของ PubMed ไม่ได้ให้น้ำหนักความเฉพาะเจาะจง ทำให้บทความทั่วไปจำนวนมากกลบ
    บทความเฉพาะเจาะจงจนไม่ถูกดึงมาแสดงเลยทั้งที่มีอยู่จริง — ค้นรอบแคบก่อนแก้ปัญหานี้ตรงๆ
    คืน (articles, query ที่ใช้จริง — เป็น query แคบถ้ารอบแรกเจอผล)
    """
    narrow_query = _strip_generic_geo_terms(query)
    if narrow_query != query:
        narrow_articles, narrow_used = _progressive_and_search(narrow_query, retmax, api_key, email)
        if narrow_articles:
            if len(narrow_articles) < retmax:
                seen_pmids = {a.get("pmid") for a in narrow_articles}
                broad_articles, _ = _progressive_and_search(query, retmax * 3, api_key, email)
                for a in broad_articles:
                    if len(narrow_articles) >= retmax:
                        break
                    if a.get("pmid") not in seen_pmids:
                        narrow_articles.append(a)
                        seen_pmids.add(a.get("pmid"))
            return narrow_articles, narrow_used
    return _progressive_and_search(query, retmax, api_key, email)


# ── Step 0: Keyword Extractor (Gemini) ──────────────────────────────────────

_KEYWORD_SYSTEM = (
    "You convert a user's natural-language request (often Thai) into an "
    "English search query for the PubMed biomedical database.\n"
    "\n"
    "Rules:\n"
    "- Keep the query SHORT: at most 2 concepts joined by AND — the core "
    "biomedical concept (disease/organism/condition) and, if given, ONE "
    "geographic-location concept. Do NOT add extra qualifier terms (e.g. "
    "\"policy\", \"guideline\", \"แนวทาง\", \"มาตรการ\", \"นโยบาย\") as their own "
    "AND term — PubMed articles rarely use those exact words as indexed "
    "text, so adding them just causes zero-result searches. Drop them; the "
    "disease + location terms alone are enough to find relevant literature.\n"
    "- Never drop a location just because it is a place name — a specific "
    "province/place IS the geographic concept, not noise.\n"
    "- If a single specific province/place is named, use it directly. Do "
    "NOT wrap it with broad fallback terms like \"Thailand\" or \"Northeast "
    "Thailand\" — that only makes the query longer without adding value.\n"
    "- PubMed does not understand Thai administrative units. Expand any Thai "
    "'health region' (เขตสุขภาพ / เขต) into its member PROVINCES joined with "
    "OR — this still counts as ONE geographic concept. Health Region 10 "
    "(เขตสุขภาพที่ 10) = Ubon Ratchathani, Sisaket, Yasothon, Amnat Charoen, "
    "Mukdahan.\n"
    "- Translate Thai terms to standard English/scientific equivalents. "
    "Keep scientific names (e.g. Opisthorchis viverrini) unchanged.\n"
    "- Output ONLY the query string. No quotes, no explanation.\n"
    "\n"
    "Examples:\n"
    "User: Opisthorchis viverrini และ อุบลราชธานี\n"
    "Query: Opisthorchis viverrini AND Ubon Ratchathani\n"
    "\n"
    "User: นโยบายโรคพยาธิใบไม้ตับ จังหวัดอุบลราชธานี\n"
    "Query: Opisthorchis viverrini AND Ubon Ratchathani\n"
    "\n"
    "User: Opisthorchis viverrini เฉพาะเขต 10\n"
    "Query: Opisthorchis viverrini AND (Ubon Ratchathani OR Sisaket OR Yasothon "
    "OR Amnat Charoen OR Mukdahan)"
)


def _clean_query_output(text: str) -> str:
    """ทำความสะอาดผลลัพธ์ดิบจาก Gemini — เผื่อโมเดลยังใส่ "Query:" นำหน้า, ครอบด้วย
    เครื่องหมายคำพูด, หรือมี ``` ติดมา ทั้งที่ system prompt สั่งห้ามไว้แล้ว (defense-in-depth)"""
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    cleaned = re.sub(r"^(Query|PubMed query)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().strip('"').strip("'").strip()


def extract_pubmed_query(prompt: str, gemini_key: str) -> dict:
    """ใช้ Gemini แปลงคำถามไทย/อังกฤษ → English PubMed query (plain text ล้วน ไม่ใช่ JSON —
    ดู _KEYWORD_SYSTEM: บังคับรักษาทุก concept ไว้รวมถึงสถานที่ และขยายเขตสุขภาพไทยเป็น
    รายชื่อจังหวัดสมาชิกด้วย OR แทนการทิ้งชื่อสถานที่ไปเฉยๆ) คืน default (ใช้ prompt ตรงๆ)
    ถ้าเรียก Gemini ไม่สำเร็จ"""
    default = {"query": prompt, "keywords": [prompt], "reasoning": ""}
    if not gemini_key:
        return default
    try:
        resp = litellm.completion(
            model="gemini/gemini-2.5-flash-lite",
            api_key=gemini_key,
            messages=[
                {"role": "system", "content": _KEYWORD_SYSTEM},
                {"role": "user", "content": f"User: {prompt}"},
            ],
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        query = _clean_query_output(text)
        if query:
            return {"query": query, "keywords": [], "reasoning": ""}
    except Exception as exc:
        log_agent_error(str(exc), agent_name="Keyword Extractor",
                        step="keyword", domain="pubmed", prompt=prompt)
    return default


def _articles_to_text(articles: list[dict]) -> str:
    if not articles:
        return "[ไม่พบบทความจาก PubMed]"
    lines = []
    for i, a in enumerate(articles, 1):
        ref = a.get("pmc_free_full_text") or a.get("url") or "-"
        lines.append(
            f"--- บทความที่ {i} ---\n"
            f"Title:    {a.get('title') or '-'}\n"
            f"Authors:  {a.get('authors') or '-'}\n"
            f"Journal:  {a.get('journal') or '-'}\n"
            f"PMID:     {a.get('pmid') or '-'}\n"
            f"URL:      {ref}\n"
            f"Abstract: {a.get('abstract') or '-'}"
        )
    return "\n\n".join(lines)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pubmed_pipeline(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    retmax: int = 10,
    history_context: str = "",
) -> None:
    """Stream PubMed research pipeline via SSE queue."""

    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    agent_steps: list[dict] = []
    api_key = os.getenv("NCBI_API_KEY") or None
    email = os.getenv("NCBI_EMAIL") or None

    # ── STEP 0: Keyword Extractor ──────────────────────────────────────────
    put({"type": "agent_start", "step": "keyword", "agentName": "Keyword Extractor"})
    extracted = extract_pubmed_query(prompt, os.getenv("GEMINI_API_KEY", ""))
    query = extracted["query"]
    put({
        "type": "agent_done",
        "step": "keyword",
        "agentName": "Keyword Extractor",
        "result": f'keyword: "{query}"',
        "reasoning": extracted.get("reasoning", ""),
    })
    agent_steps.append({"step": "keyword", "agentName": "Keyword Extractor",
                        "result": f'keyword: "{query}"'})

    # ── STEP 1: PubMed Fetcher ─────────────────────────────────────────────
    # ⚠️ ใช้ search_pubmed_progressive แทนการค้นด้วย query เต็มตรงๆ — ถ้า Keyword
    # Extractor เผลอ AND MeSH term มากเกินไป (เช่น โรค + ประเด็น + ชื่อประเทศ) จะไม่มี
    # บทความไหนตรงครบทุกเงื่อนไข (0 ผล) จึงต้องลองตัด term ท้ายออกทีละตัวจนกว่าจะเจอ
    put({"type": "agent_start", "step": "fetcher", "agentName": "PubMed Fetcher"})
    used_query = query
    try:
        articles, used_query = search_pubmed_progressive(query, retmax, api_key, email)
        if not articles:
            fetcher_result = f"ไม่พบบทความ free full text บน PubMed สำหรับ '{query}'"
        elif used_query != query:
            fetcher_result = f"พบ {len(articles)} บทความสำหรับ '{used_query}' (ขยายคำค้นจาก '{query}' เพราะค้นตรงๆ ไม่พบผล)"
        else:
            fetcher_result = f"พบ {len(articles)} บทความสำหรับ '{query}'"
    except (httpx.HTTPError, ET.ParseError) as exc:
        log_agent_error(str(exc), agent_name="PubMed Fetcher",
                        step="fetcher", domain="pubmed", prompt=query)
        articles = []
        fetcher_result = f"เกิดข้อผิดพลาดขณะค้นหา PubMed: {exc}"

    article_count = len(articles)
    articles_text = _articles_to_text(articles)

    put({"type": "agent_done", "step": "fetcher", "agentName": "PubMed Fetcher",
         "result": fetcher_result, "articleCount": article_count})
    agent_steps.append({"step": "fetcher", "agentName": "PubMed Fetcher",
                        "result": fetcher_result})

    # ── STEP 2: Stream article summaries as text ───────────────────────────
    sep_heavy = "═" * 44 + "\n\n"
    sep_light = "─" * 44 + "\n\n"

    full_text = f"🔬 พบ {article_count} บทความจาก PubMed สำหรับ \"{used_query}\"\n\n{sep_heavy}"
    if articles:
        for i, article in enumerate(articles, 1):
            title = article.get("title") or "(ไม่มีชื่อเรื่อง)"
            authors = article.get("authors")
            journal = article.get("journal")
            abstract = article.get("abstract") or "(ไม่มีบทคัดย่อ)"
            pmid = article.get("pmid")
            free_full_text = article.get("pmc_free_full_text")
            url = article.get("url")

            full_text += f"📄 บทความที่ {i}: {title}\n"
            if authors:
                full_text += f"ผู้แต่ง: {authors}\n"
            if journal:
                full_text += f"วารสาร: {journal}\n"
            if pmid:
                full_text += f"PMID: {pmid}\n"
            full_text += f"\n{abstract}\n\n"
            if free_full_text:
                full_text += f"🔗 Free full text (PMC): {free_full_text}\n"
            if url:
                full_text += f"🔗 PubMed: {url}\n"
            full_text += "\n" + sep_light
    else:
        full_text += "ไม่พบบทความที่เกี่ยวข้อง ลองใช้คำค้นหาอื่น หรือถามให้เจาะจงมากขึ้น\n"

    put({"type": "text_stream_start", "articleCount": article_count})

    chunk_size = 200
    for start in range(0, len(full_text), chunk_size):
        put({"type": "text_chunk", "text": full_text[start:start + chunk_size]})

    # ── FINAL EVENT ────────────────────────────────────────────────────────
    put({
        "type": "final",
        "message": fetcher_result,
        "textResult": full_text,
        "articlesText": articles_text,
        "reportTitle": prompt,
        "articleCount": article_count,
        "agentSteps": agent_steps,
    })
