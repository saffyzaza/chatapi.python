"""Tavily web search tool — ค้นหาข้อมูลภายนอกแบบ real-time."""
from crewai.tools import tool
from src.config import get_settings


def _search(query: str, max_results: int = 2) -> str:
    from tavily import TavilyClient
    api_key = get_settings().TAVILY_API_KEY
    if not api_key:
        return "ไม่พบ TAVILY_API_KEY — กรุณาตั้งค่าใน .env"

    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=2,
        )
    except Exception as exc:
        return f"Tavily search error: {exc}"

    lines = [f"[Tavily Search] query: {query}\n"]

    answer = response.get("answer") or ""
    if answer:
        lines.append(f"คำตอบสรุป: {answer}\n")

    results = response.get("results") or []
    if not results:
        return "\n".join(lines) + "\nไม่พบผลการค้นหา"

    lines.append(f"พบ {len(results)} ผลลัพธ์:\n")
    for i, r in enumerate(results, 1):
        title   = r.get("title", "ไม่มีชื่อ")
        url     = r.get("url", "")
        content = (r.get("content") or "")[:400]
        lines.append(
            f"{i}. {title}\n"
            f"   URL: {url}\n"
            f"   สรุป: {content}\n"
        )

    return "\n".join(lines)


@tool("tavily_search")
def tavily_search(query: str, max_results: int = 2) -> str:
    """ค้นหาข้อมูลภายนอกแบบ real-time ด้วย Tavily (fast, thailand, max 2 results).

    Args:
        query: คำค้นหา (ภาษาไทยหรืออังกฤษ)
        max_results: จำนวนผลลัพธ์สูงสุด (default 2)
    """
    return _search(query, max_results)
