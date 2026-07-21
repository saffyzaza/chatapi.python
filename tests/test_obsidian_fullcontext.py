"""Regression / golden-set tests for src/agents/obsidian_fullcontext.py

เกิดจากบั๊กที่เจอจริงตอนทดสอบระบบผ่านหน้าแชท: ถามคำถามกว้าง ๆ อย่าง
"จังหวัด อุบล เอกสารอะไรบ้าง" แล้วคำตอบมีเนื้อหาดิบของเอกสารต้นฉบับ (บรรทัด
"FILE:", YAML frontmatter, wikilink [[...]]) หลุดเข้ามาในคำตอบตรง ๆ และปุ่ม
คำถามแนะนำ (follow_ups) กลายเป็นหัวข้อ markdown ของคำตอบเอง (เช่น "**สรุปคำตอบ**")

ชุดเทสต์นี้ล็อกพฤติกรรมที่ถูกต้องไว้ไม่ให้ regression กลับไปเป็นแบบเดิมอีก —
ตรงตามที่ Traceability Matrix ของวอลต์เอกสารเองก็ทำเครื่องหมายไว้ว่าเป็นช่องโหว่
("ไม่มี automated test suite" ปนอยู่แทบทุก FR-CHAT/FR-KB)
"""
import json
import types

import litellm
import pytest

from src.agents import obsidian_fullcontext as ofc


class FakeSettings:
    GEMINI_API_KEY = "test-key"
    GEMINI_MODEL_PRO = "gemini-2.5-pro"
    REPORT_MAX_TOKENS = 4096
    OBSIDIAN_MAX_CONTEXT_CHARS = 500_000


def _completion_response(content: str):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def _fake_notes(rows):
    return lambda *_args, **_kwargs: rows


CLEAN_ANSWER = (
    "**สรุปคำตอบ**\n"
    "จังหวัดอุบลราชธานีมีรายงานตรวจราชการหลายฉบับ ครอบคลุมประเด็นสุขภาพต่าง ๆ\n\n"
    "**ข้อมูลจากคลังความรู้**\n"
    "- ปีงบประมาณ 2566 และ 2568\n\n"
    "<<<FOLLOWUPS>>>\n"
    '["มีข้อมูลปีล่าสุดไหม?", "แยกรายอำเภอได้ไหม?"]\n'
    "<<<END_FOLLOWUPS>>>"
)

# จำลองบั๊กจริงที่เจอ: FILE: marker + YAML frontmatter + wikilink หลุดเข้าคำตอบ
LEAKED_ANSWER = (
    "**สรุปคำตอบ**\n"
    "มีเอกสารดังนี้\n\n"
    "## FILE: เขต10/อุบลราชธานี/R_รายงานตรวจราชการ-อุบลราชธานี-2566-ส่วนที่01.md\n\n"
    "---\n"
    'title: "R_รายงานตรวจราชการ-อุบลราชธานี-2566 (ส่วนที่ 1/20)"\n'
    'province: "อุบลราชธานี"\n'
    "---\n\n"
    "[[เขต10/INDEX|เขต 10]] > [[เขต10/อุบลราชธานี/INDEX|อุบลราชธานี]]\n\n"
    "<<<FOLLOWUPS>>>\n"
    '["อยากรู้เพิ่มไหม?"]\n'
    "<<<END_FOLLOWUPS>>>"
)


# ── 1. Anti-leak guard: regex detection ─────────────────────────────────────

class TestContainsLeak:
    def test_detects_file_marker(self):
        assert ofc._contains_leak("## FILE: foo/bar.md\n\nเนื้อหา") is True

    def test_detects_yaml_frontmatter_block(self):
        text = '---\ntitle: "x"\nprovince: "y"\n---\n\nเนื้อหาต่อ'
        assert ofc._contains_leak(text) is True

    def test_detects_yaml_fenced_code_block(self):
        text = '```yaml\ntitle: "x"\n```\n\nเนื้อหาต่อ'
        assert ofc._contains_leak(text) is True

    def test_detects_wikilink(self):
        assert ofc._contains_leak("ดูที่ [[เขต10/INDEX|เขต 10]] ก่อน") is True

    def test_clean_text_is_not_flagged(self):
        clean = (
            "จังหวัดอุบลราชธานีมีรายงานตรวจราชการหลายฉบับ ครอบคลุมประเด็นสุขภาพ "
            "ต่าง ๆ เช่น การคัดกรองมะเร็งปากมดลูก และการดูแลผู้สูงอายุ"
        )
        assert ofc._contains_leak(clean) is False

    def test_full_leaked_sample_is_flagged(self):
        content, _ = ofc._extract_and_strip_followups(LEAKED_ANSWER)
        assert ofc._contains_leak(content) is True

    def test_full_clean_sample_is_not_flagged(self):
        content, _ = ofc._extract_and_strip_followups(CLEAN_ANSWER)
        assert ofc._contains_leak(content) is False


class TestStripLeakedBlocks:
    def test_removes_all_markers_best_effort(self):
        content, _ = ofc._extract_and_strip_followups(LEAKED_ANSWER)
        cleaned = ofc._strip_leaked_blocks(content)
        assert not ofc._contains_leak(cleaned)
        assert "FILE:" not in cleaned
        assert "[[" not in cleaned
        assert "---" not in cleaned


# ── 2. Structured follow_ups extraction ─────────────────────────────────────

class TestExtractAndStripFollowups:
    def test_extracts_valid_json_array(self):
        content, follow_ups = ofc._extract_and_strip_followups(CLEAN_ANSWER)
        assert follow_ups == ["มีข้อมูลปีล่าสุดไหม?", "แยกรายอำเภอได้ไหม?"]
        assert "<<<FOLLOWUPS>>>" not in content
        assert "<<<END_FOLLOWUPS>>>" not in content

    def test_missing_block_yields_empty_list(self):
        content, follow_ups = ofc._extract_and_strip_followups("คำตอบธรรมดาไม่มีบล็อกท้าย")
        assert follow_ups == []
        assert content == "คำตอบธรรมดาไม่มีบล็อกท้าย"

    def test_rejects_items_without_question_mark(self):
        text = '<<<FOLLOWUPS>>>\n["สรุปคำตอบ", "อยากรู้เพิ่มไหม?"]\n<<<END_FOLLOWUPS>>>'
        _, follow_ups = ofc._extract_and_strip_followups(text)
        assert follow_ups == ["อยากรู้เพิ่มไหม?"]

    def test_rejects_items_with_markdown_bold(self):
        """ล็อกบั๊กที่เจอจริง: ปุ่มแนะนำเคยกลายเป็น '**สรุปคำตอบ**' '**ข้อมูลจากคลังความรู้**' """
        text = (
            '<<<FOLLOWUPS>>>\n'
            '["**สรุปคำตอบ**", "**ข้อมูลจากคลังความรู้**", "อยากรู้เพิ่มไหม?"]\n'
            '<<<END_FOLLOWUPS>>>'
        )
        _, follow_ups = ofc._extract_and_strip_followups(text)
        assert follow_ups == ["อยากรู้เพิ่มไหม?"]

    def test_malformed_json_does_not_crash(self):
        text = "<<<FOLLOWUPS>>>\nนี่ไม่ใช่ json เลย\n<<<END_FOLLOWUPS>>>"
        content, follow_ups = ofc._extract_and_strip_followups(text)
        assert follow_ups == []
        assert isinstance(content, str)

    def test_caps_at_three_items(self):
        items = [f"คำถามที่ {i}?" for i in range(10)]
        text = f"<<<FOLLOWUPS>>>\n{json.dumps(items, ensure_ascii=False)}\n<<<END_FOLLOWUPS>>>"
        _, follow_ups = ofc._extract_and_strip_followups(text)
        assert len(follow_ups) == 3


# ── 2b. Note-reference title cleanup (สำหรับ dedupe เอกสารที่ถูกตัดแบ่งหลาย .md) ──

class TestCleanDocTitle:
    def test_strips_part_number_suffix(self):
        assert ofc._clean_doc_title(
            "R_รายงานการตรวจราชการและนิเทศงานกรณีปกติ-อุบลราชธานี-2567-ส่วนที่05"
        ) == "R_รายงานการตรวจราชการและนิเทศงานกรณีปกติ-อุบลราชธานี-2567"

    def test_strips_index_suffix(self):
        assert ofc._clean_doc_title(
            "R_รายงานการตรวจราชการและนิเทศงานกรณีปกติ-อุบลราชธานี-2567-INDEX"
        ) == "R_รายงานการตรวจราชการและนิเทศงานกรณีปกติ-อุบลราชธานี-2567"

    def test_leaves_title_without_suffix_unchanged(self):
        assert ofc._clean_doc_title("R_รายงานตรวจราชการ-อุบลราชธานี-2566") == \
            "R_รายงานตรวจราชการ-อุบลราชธานี-2566"

    def test_never_returns_empty_string(self):
        # กันเคส edge ที่ regex อาจกินจนเหลือสตริงว่าง — ต้อง fallback กลับค่าเดิม
        assert ofc._clean_doc_title("ส่วนที่01") != ""


# ── 3. Streaming guard: หยุดส่งสดทันทีที่พบร่องรอยเนื้อหาดิบ ──────────────────

class TestCallGeminiStreamingGuard:
    def test_stops_forwarding_once_leak_detected_but_keeps_full_text(self, monkeypatch):
        chunks = ["สรุปคำตอบ: มีเอกสารดังนี้\n", "## FILE: foo.md\n", "เนื้อหาต่อจากนี้"]

        def fake_stream(*_args, **_kwargs):
            for c in chunks:
                yield types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=c))]
                )

        monkeypatch.setattr(litellm, "completion", lambda *a, **k: fake_stream())

        forwarded: list[str] = []
        full_text = ofc._call_gemini("sys", "user", FakeSettings(), on_delta=forwarded.append)

        assert full_text == "".join(chunks)  # เก็บครบทุก token ไว้ในหน่วยความจำเสมอ
        assert forwarded == ["สรุปคำตอบ: มีเอกสารดังนี้\n"]  # หยุดส่งสดทันทีที่เจอ FILE:

    def test_forwards_all_chunks_when_clean(self, monkeypatch):
        chunks = ["สวัสดีครับ ", "นี่คือคำตอบ ", "ที่สะอาดดี"]

        def fake_stream(*_args, **_kwargs):
            for c in chunks:
                yield types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=c))]
                )

        monkeypatch.setattr(litellm, "completion", lambda *a, **k: fake_stream())

        forwarded: list[str] = []
        full_text = ofc._call_gemini("sys", "user", FakeSettings(), on_delta=forwarded.append)

        assert full_text == "".join(chunks)
        assert forwarded == chunks

    def test_non_streaming_path_unaffected(self, monkeypatch):
        monkeypatch.setattr(litellm, "completion", lambda *a, **k: _completion_response("คำตอบปกติ"))
        result = ofc._call_gemini("sys", "user", FakeSettings(), on_delta=None)
        assert result == "คำตอบปกติ"


# ── 4. Golden-set: end-to-end pipeline (mocked DB + Gemini) ─────────────────

class TestRunObsidianAskFullcontextGoldenSet:
    """จำลอง Gemini ตอบเนื้อหาดิบหลุดในความพยายามแรก (เหมือนบั๊กจริงที่เจอ) แล้ว
    ยืนยันว่า guard+retry ทำงาน — เอาต์พุตสุดท้ายที่ผู้ใช้เห็นต้องสะอาดเสมอ"""

    def _patch_common(self, monkeypatch, notes):
        monkeypatch.setattr(ofc, "get_settings", lambda: FakeSettings())
        monkeypatch.setattr(ofc, "query_db", _fake_notes(notes))

    def test_leak_on_first_attempt_triggers_retry_and_clean_final_output(self, monkeypatch):
        self._patch_common(monkeypatch, [
            {"relative_path": "เขต10/อุบลราชธานี/R_รายงาน-01.md",
             "content": "เนื้อหาเอกสารตัวอย่างเกี่ยวกับการคัดกรองมะเร็ง", "file_id": None},
        ])
        calls = {"n": 0}

        def fake_completion(*_args, **_kwargs):
            calls["n"] += 1
            return _completion_response(LEAKED_ANSWER if calls["n"] == 1 else CLEAN_ANSWER)

        monkeypatch.setattr(litellm, "completion", fake_completion)

        result = ofc.run_obsidian_ask_fullcontext(
            "จังหวัด อุบล เอกสารอะไรบ้าง", province="อุบลราชธานี",
        )

        assert calls["n"] == 2, "ต้อง retry อีกครั้งเมื่อพบเนื้อหาดิบหลุดในความพยายามแรก"
        assert not ofc._contains_leak(result.content)
        assert "<<<FOLLOWUPS>>>" not in result.content
        assert "FILE:" not in result.content
        assert "[[" not in result.content
        for q in result.follow_ups:
            assert q.endswith("?")
            assert "**" not in q

    def test_clean_first_attempt_does_not_retry(self, monkeypatch):
        self._patch_common(monkeypatch, [
            {"relative_path": "เขต10/อุบลราชธานี/R_รายงาน-01.md",
             "content": "เนื้อหาเอกสารตัวอย่าง", "file_id": None},
        ])
        calls = {"n": 0}

        def fake_completion(*_args, **_kwargs):
            calls["n"] += 1
            return _completion_response(CLEAN_ANSWER)

        monkeypatch.setattr(litellm, "completion", fake_completion)

        result = ofc.run_obsidian_ask_fullcontext(
            "รายงานสถานการณ์โรคและภัยสุขภาพ", province="อุบลราชธานี",
        )

        assert calls["n"] == 1, "คำตอบสะอาดตั้งแต่แรกไม่ควรมีการ retry"
        assert not ofc._contains_leak(result.content)
        assert result.follow_ups == ["มีข้อมูลปีล่าสุดไหม?", "แยกรายอำเภอได้ไหม?"]

    def test_leak_persists_after_retry_falls_back_to_best_effort_strip(self, monkeypatch):
        """ถ้า retry แล้วยังหลุดอีก (edge case สุด ๆ) ต้องไม่ปล่อยเนื้อหาดิบออกไปให้
        ผู้ใช้เห็นเด็ดขาด — ระบบต้องตัดออกเองแบบ best-effort เป็นด่านสุดท้าย"""
        self._patch_common(monkeypatch, [
            {"relative_path": "เขต10/อุบลราชธานี/R_รายงาน-01.md",
             "content": "เนื้อหาเอกสารตัวอย่าง", "file_id": None},
        ])
        monkeypatch.setattr(litellm, "completion", lambda *a, **k: _completion_response(LEAKED_ANSWER))

        result = ofc.run_obsidian_ask_fullcontext(
            "จังหวัด อุบล เอกสารอะไรบ้าง", province="อุบลราชธานี",
        )

        assert not ofc._contains_leak(result.content), (
            "ผู้ใช้ต้องไม่เห็นเนื้อหาดิบไม่ว่ากรณีใด แม้ retry จะยังหลุดอีกก็ตาม"
        )

    def test_streaming_on_delta_is_forwarded_during_clean_generation(self, monkeypatch):
        self._patch_common(monkeypatch, [
            {"relative_path": "เขต10/อุบลราชธานี/R_รายงาน-01.md",
             "content": "เนื้อหาเอกสารตัวอย่าง", "file_id": None},
        ])
        chunks = [p + "\n" for p in CLEAN_ANSWER.split("\n") if p]

        def fake_stream(*_args, **_kwargs):
            for c in chunks:
                yield types.SimpleNamespace(
                    choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=c))]
                )

        monkeypatch.setattr(litellm, "completion", lambda *a, **k: fake_stream())

        streamed: list[str] = []
        result = ofc.run_obsidian_ask_fullcontext(
            "รายงานสถานการณ์โรคและภัยสุขภาพ", province="อุบลราชธานี",
            on_delta=streamed.append,
        )

        assert "".join(streamed) == "".join(chunks)
        assert not ofc._contains_leak(result.content)

    def test_notes_referenced_dedupe_by_pdf_file_id(self, monkeypatch):
        """เอกสาร PDF ต้นฉบับที่ถูกตัดแบ่งเป็นหลาย .md 'ส่วน' ตอน ingest (ทุกส่วนชี้
        minio file_id เดียวกัน) ต้องโผล่ในการอ้างอิงเป็น 'ลิงก์เดียว' ไม่ใช่ซ้ำทุกส่วน
        — ล็อกบั๊กที่เจอจริง: ถามเจอเอกสาร 20 ส่วนของรายงานเดียวกัน แต่โชว์ป้าย 20 อัน"""
        self._patch_common(monkeypatch, [
            {"relative_path": "เขต10/อุบลราชธานี/R_รายงาน-2567-ส่วนที่01.md",
             "content": "เนื้อหาส่วนที่ 1", "file_id": "783979"},
            {"relative_path": "เขต10/อุบลราชธานี/R_รายงาน-2567-ส่วนที่02.md",
             "content": "เนื้อหาส่วนที่ 2", "file_id": "783979"},
            {"relative_path": "เขต10/อุบลราชธานี/R_รายงาน-2567-INDEX.md",
             "content": "สารบัญ", "file_id": "783979"},
            {"relative_path": "เขต10/อุบลราชธานี/R_รายงาน-2566-ส่วนที่01.md",
             "content": "เนื้อหาปี 2566", "file_id": "815316"},
            {"relative_path": "เขต10/อุบลราชธานี/หมายเหตุอื่นๆ.md",
             "content": "โน้ตที่ไม่มี PDF ผูก", "file_id": None},
        ])
        monkeypatch.setattr(litellm, "completion", lambda *a, **k: _completion_response(CLEAN_ANSWER))

        result = ofc.run_obsidian_ask_fullcontext(
            "จังหวัด อุบล เอกสารอะไรบ้าง", province="อุบลราชธานี",
        )

        # 5 notes ดิบ (3 ส่วนของไฟล์เดียวกัน + 1 ไฟล์อื่น + 1 ไม่มี PDF) → เหลือ 3 อ้างอิง
        assert len(result.notes_referenced) == 3
        pdf_urls = [n.pdf_url for n in result.notes_referenced]
        assert pdf_urls.count("/api/pdf/view/783979") == 1
        assert pdf_urls.count("/api/pdf/view/815316") == 1
        assert None in pdf_urls
        # ชื่อที่โชว์ต้องตัดคำต่อท้าย "-ส่วนที่NN"/"-INDEX" ออกแล้ว
        titles = [n.title for n in result.notes_referenced]
        assert not any(t.endswith(("ส่วนที่01", "ส่วนที่02", "INDEX")) for t in titles)
