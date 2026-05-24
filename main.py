"""Health Analytics API — application entry point."""
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import get_settings
from src.routers.accident_chat import router as accident_chat_router
from src.routers.accident_policy import router as accident_policy_router
from src.routers.analyze import router as analyze_router
from src.routers.error_log import router as error_log_router
from src.routers.thaijo import router as thaijo_router
from src.routers.tools_router import router as tools_router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)

s = get_settings()

app = FastAPI(
    title="Health Analytics API",
    description="CSV Data Analyst + Zone 10 Accident Policy Agent",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=s.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(accident_chat_router)
app.include_router(accident_policy_router)
app.include_router(error_log_router)
app.include_router(thaijo_router)
app.include_router(tools_router)

# ── Static UI pages ──────────────────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "src" / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

_UI_PAGES = [
    ("unified_test_ui.html",        "🤖 Unified Agent Test",       "Multi-pipeline chat + streaming"),
    ("policy_brief_ui.html",        "📋 Policy Brief",             "สร้าง Policy Brief อัตโนมัติ"),
    ("accident_chat_ui.html",       "🚗 Accident Chat",            "วิเคราะห์อุบัติเหตุ Zone 10"),
    ("accident_policy_ui.html",     "📊 Accident Policy",          "ข้อมูลนโยบายอุบัติเหตุ"),
    ("document_agent_test_ui.html", "📄 Document Agent Test",      "ทดสอบ Document Agent"),
    ("document_upload_ui.html",     "📁 Document Upload",          "อัปโหลดและ APA Citation"),
    ("citation_test_ui.html",       "🔖 Citation Test",            "ทดสอบระบบ Citation"),
    ("db_explorer_ui.html",         "🗄️ DB Explorer",             "สำรวจ Database Tables"),
    ("thaijo_research_ui.html",     "🔬 ThaiJO Research",          "ค้นหางานวิจัย ThaiJO"),
    ("test_ui.html",                "🧪 Test UI",                  "ทดสอบ Tools & Pipeline"),
]


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui_index():
    cards = "\n".join(
        f"""
        <a href="/static/{fname}" class="card">
          <div class="card-icon">{icon.split()[0]}</div>
          <div class="card-body">
            <div class="card-title">{" ".join(icon.split()[1:])}</div>
            <div class="card-desc">{desc}</div>
          </div>
          <svg class="card-arrow" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd"/>
          </svg>
        </a>"""
        for fname, icon, desc in _UI_PAGES
    )
    return f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Health Analytics — UI Index</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'IBM Plex Sans Thai', sans-serif;
      background: #f1f5f9;
      min-height: 100vh;
      padding: 2.5rem 1rem;
      color: #1e293b;
    }}
    .wrapper {{ max-width: 720px; margin: 0 auto; }}
    header {{ text-align: center; margin-bottom: 2rem; }}
    header h1 {{ font-size: 1.75rem; font-weight: 700; color: #1e293b; }}
    header p {{ margin-top: 0.4rem; font-size: 0.9rem; color: #64748b; }}
    .grid {{ display: flex; flex-direction: column; gap: 0.75rem; }}
    .card {{
      display: flex; align-items: center; gap: 1rem;
      background: #fff; border: 1px solid #e2e8f0;
      border-radius: 14px; padding: 1rem 1.25rem;
      text-decoration: none; color: inherit;
      transition: box-shadow 0.15s, border-color 0.15s, transform 0.1s;
    }}
    .card:hover {{
      box-shadow: 0 4px 20px rgba(99,102,241,0.12);
      border-color: #6366f1;
      transform: translateY(-1px);
    }}
    .card-icon {{ font-size: 1.75rem; flex-shrink: 0; width: 2.5rem; text-align: center; }}
    .card-body {{ flex: 1; }}
    .card-title {{ font-weight: 600; font-size: 0.97rem; }}
    .card-desc {{ font-size: 0.82rem; color: #64748b; margin-top: 0.15rem; }}
    .card-arrow {{ width: 1.1rem; height: 1.1rem; color: #94a3b8; flex-shrink: 0; }}
    footer {{ text-align: center; margin-top: 2.5rem; font-size: 0.78rem; color: #94a3b8; }}
  </style>
</head>
<body>
  <div class="wrapper">
    <header>
      <h1>🏥 Health Analytics</h1>
      <p>เลือก UI ที่ต้องการใช้งาน</p>
    </header>
    <div class="grid">
      {cards}
    </div>
    <footer>Health Analytics API · <a href="/docs" style="color:#6366f1;">Swagger Docs</a></footer>
  </div>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=s.HOST, port=s.PORT, reload=True, log_level=s.LOG_LEVEL)
