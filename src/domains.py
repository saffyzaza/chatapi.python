"""Domain definitions — health analysis domains d0–d8."""
from dataclasses import dataclass


@dataclass
class Domain:
    code: str
    name_th: str
    name_en: str
    folder_prefix: str
    expertise: str


DOMAINS: dict[str, Domain] = {
    "d0": Domain(
        code="d0",
        name_th="ทั่วไป",
        name_en="General Advisor",
        folder_prefix="",
        expertise="ผู้เชี่ยวชาญด้านสุขภาพและข้อมูลสาธารณสุขทั่วไป วิเคราะห์ได้ทุกประเด็น",
    ),
    "d1": Domain(
        code="d1",
        name_th="อุบัติเหตุทางถนน",
        name_en="Road Accidents",
        folder_prefix="D1_Road",
        expertise="ผู้เชี่ยวชาญด้านอุบัติเหตุทางถนน การบาดเจ็บ การเสียชีวิต และความปลอดภัยบนท้องถนน",
    ),
    "d2": Domain(
        code="d2",
        name_th="สุขภาพจิต",
        name_en="Mental Health",
        folder_prefix="D2_Mental",
        expertise="ผู้เชี่ยวชาญด้านสุขภาพจิต การฆ่าตัวตาย ภาวะซึมเศร้า และบริการจิตเวช",
    ),
    "d3": Domain(
        code="d3",
        name_th="โรคไม่ติดต่อ",
        name_en="NCDs",
        folder_prefix="D3_NCD",
        expertise="ผู้เชี่ยวชาญด้านโรคไม่ติดต่อเรื้อรัง เช่น เบาหวาน ความดันโลหิตสูง โรคหัวใจ โรคหลอดเลือดสมอง",
    ),
    "d4": Domain(
        code="d4",
        name_th="โภชนาการ",
        name_en="Nutrition",
        folder_prefix="D4_Nutrition",
        expertise="ผู้เชี่ยวชาญด้านโภชนาการ ภาวะทุพโภชนาการ โรคอ้วน และความมั่นคงทางอาหาร",
    ),
    "dt": Domain(
        code="dt",
        name_th="วิจัย ThaiJo",
        name_en="ThaiJo Research",
        folder_prefix="",
        expertise=(
            "ผู้เชี่ยวชาญด้านการสังเคราะห์งานวิจัยทางวิชาการ "
            "ค้นหาและสรุปบทความจากฐานข้อมูล ThaiJo สร้างรายงานวิชาการอัตโนมัติ"
        ),
    ),
    "obsidian": Domain(
        code="obsidian",
        name_th="คลังความรู้สุขภาพ เขต 10",
        name_en="Obsidian Knowledge Vault",
        folder_prefix="",
        expertise=(
            "ผู้เชี่ยวชาญด้านข้อมูลสุขภาพเขตสุขภาพที่ 10 (อุบลราชธานี ศรีสะเกษ ยโสธร อำนาจเจริญ มุกดาหาร) "
            "ค้นหาและตอบคำถามจาก Obsidian Knowledge Vault ซึ่งเป็นฐานความรู้ "
            "ด้านนโยบาย รายงาน และข้อมูลสุขภาพของเขตสุขภาพที่ 10"
        ),
    ),
}

DOMAIN_LIST_TEXT = "\n".join(
    f"- {d.code}: {d.name_th} ({d.name_en})"
    for d in DOMAINS.values()
)
