"""Tavily Search tool — ค้นหาข้อมูลจากอินเทอร์เน็ตแบบ real-time.

ใช้ Tavily Search API (search endpoint) — คืนผลลัพธ์ตามจำนวนที่กำหนด
(TAVILY_MAX_RESULTS) แหล่งข้อมูลภาษาไทยได้อยู่แล้วเพราะ search query ถูกบังคับ
เป็นภาษาไทยเสมอ (ดู _search_agent_prompt ใน tavily_pipeline.py)

(เปลี่ยนกลับจาก Research API เพราะ research คุมจำนวนแหล่งอ้างอิงไม่ได้เป๊ะ,
ช้ากว่ามาก และกินเครดิตจนชนเพดานแพลน — search ให้ 10 แหล่งเป๊ะ เร็ว ประหยัด)

⚠️ ไม่ใช้ country="thailand" boost อีกต่อไป — พิสูจน์แล้วว่าพารามิเตอร์นี้ทำให้ Tavily
คืนผลลัพธ์จากแหล่ง aggregator (คล้าย Google News) ที่ url เป็น relative path แบบ
"/goto?url=CAES..." (ไม่มี domain, เปิด/อ้างอิงจริงไม่ได้) แทบทั้งหมด — ทดสอบคำถาม
เดียวกันแบบไม่ใส่ country ได้ผลลัพธ์เป็น absolute URL จริงทุกรายการ (ddc.moph.go.th,
roadsafetythai.org ฯลฯ) และยังคงเป็นแหล่งภาษาไทยตรงประเด็นเหมือนเดิม เพราะ query
เป็นภาษาไทยอยู่แล้วก็เพียงพอต่อการดึงแหล่งไทยโดยไม่ต้องพึ่ง country boost
"""
from crewai.tools import tool
from src.config import get_settings


def _search(query: str, max_results: int | None = None) -> str:
    settings = get_settings()
    api_key = settings.TAVILY_API_KEY
    if not api_key:
        return "ไม่พบ TAVILY_API_KEY — กรุณาตั้งค่าใน .env"
    if max_results is None:
        max_results = settings.TAVILY_MAX_RESULTS

    # ⚠️ import + client construction ต้องอยู่ใน try/except — ถ้า package
    # 'tavily-python' ไม่ได้ติดตั้ง/เวอร์ชันเก่า ข้อผิดพลาดจะหลุดออกไปทำให้ agent
    # พังทั้งกระบวนการ แทนที่จะได้ข้อความอธิบายที่ใช้ต่อได้
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
    except Exception as exc:
        return f"Tavily init error: {exc}"

    # ── ค้นหา: 10 ผลลัพธ์ ────────────────────────────────────────────────────
    # ⚠️ ไม่ส่ง country="thailand" — ดูเหตุผลเต็มใน docstring บนสุดของไฟล์นี้
    # (ทำให้ได้ผลลัพธ์เป็น relative URL ที่อ้างอิงไม่ได้เกือบทั้งหมด)
    base_kwargs = dict(query=query, search_depth="basic", max_results=max_results)
    try:
        response = client.search(**base_kwargs)
    except Exception as exc:
        return f"Tavily search error: {exc}"

    lines = [f"[Tavily Search] query: {query}\n"]

    answer = response.get("answer") or ""
    if answer:
        lines.append(f"คำตอบสรุป: {answer}\n")

    results = response.get("results") or []

    # ⚠️ Tavily บางครั้งคืน url ที่ไม่ใช่ absolute URL (เช่น "/goto?url=CAES..." —
    # relative href ที่ครอบ crawler ไปเจอในหน้า aggregator เช่น Google News แล้วดึง
    # href ดิบมาโดยไม่ resolve เป็น absolute) ลิงก์แบบนี้ไม่มี domain ให้เปิดได้จริง
    # ใช้อ้างอิงในรายงานไม่ได้ (คลิกแล้วพังเพราะไม่มี host) — กรองทิ้งก่อนส่งต่อ
    # ดีกว่าปล่อยให้ agent อื่นเอาไปอ้างอิงเป็นแหล่งที่มาที่เปิดไม่ได้จริง
    valid_results = [r for r in results if (r.get("url") or "").startswith(("http://", "https://"))]
    dropped = len(results) - len(valid_results)
    results = valid_results

    if not results:
        msg = "\n".join(lines) + "\nไม่พบผลการค้นหาที่มี URL ใช้งานได้"
        if dropped:
            msg += f" (คัดกรองออก {dropped} รายการที่ URL ไม่สมบูรณ์)"
        return msg

    lines.append(f"พบ {len(results)} ผลลัพธ์:\n")
    if dropped:
        lines.append(f"(คัดกรองออก {dropped} รายการที่ URL ไม่สมบูรณ์/ไม่ใช่ absolute URL)\n")
    content_chars = settings.TAVILY_CONTENT_CHARS
    for i, r in enumerate(results, 1):
        title   = r.get("title", "ไม่มีชื่อ")
        url     = r.get("url", "")
        content = (r.get("content") or "")[:content_chars]
        lines.append(
            f"{i}. {title}\n"
            f"   URL: {url}\n"
            f"   สรุป: {content}\n"
        )

    return "\n".join(lines)


@tool("tavily_search")
def tavily_search(query: str, max_results: int | None = None) -> str:
    """ค้นหาข้อมูลจากอินเทอร์เน็ตแบบ real-time ด้วย Tavily (เน้นแหล่งข้อมูลไทย).

    Args:
        query: คำค้นหา — **ควรเป็นภาษาไทย** เพื่อให้ได้แหล่งข้อมูลไทย
        max_results: จำนวนผลลัพธ์ (ไม่ระบุ = ใช้ค่าจาก .env TAVILY_MAX_RESULTS = 10)
    """
    return _search(query, max_results)
