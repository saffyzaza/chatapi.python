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
    "d5": Domain(
        code="d5",
        name_th="ผู้สูงอายุ",
        name_en="Elderly Care",
        folder_prefix="D5_Elderly",
        expertise="ผู้เชี่ยวชาญด้านสุขภาพผู้สูงอายุ การดูแลระยะยาว ภาวะพึ่งพิง และสังคมผู้สูงวัย",
    ),
    "d6": Domain(
        code="d6",
        name_th="โรคติดต่อ",
        name_en="Communicable Disease",
        folder_prefix="D6_Communicable",
        expertise="ผู้เชี่ยวชาญด้านโรคติดต่อ เช่น ไข้เลือดออก มาลาเรีย วัณโรค และการระบาดของโรค",
    ),
    "d7": Domain(
        code="d7",
        name_th="มะเร็ง",
        name_en="Cancer",
        folder_prefix="D7_Cancer",
        expertise="ผู้เชี่ยวชาญด้านโรคมะเร็ง อัตราการเกิดโรค การตรวจคัดกรอง และสถิติการรักษา",
    ),
    "d8": Domain(
        code="d8",
        name_th="ประชากร",
        name_en="Population",
        folder_prefix="D8_Population",
        expertise="ผู้เชี่ยวชาญด้านประชากรศาสตร์ การเติบโต การกระจายตัว และโครงสร้างประชากร",
    ),
}

DOMAIN_LIST_TEXT = "\n".join(
    f"- {d.code}: {d.name_th} ({d.name_en})"
    for d in DOMAINS.values()
)
