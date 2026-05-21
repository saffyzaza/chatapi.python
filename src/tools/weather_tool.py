"""Weather Tool — ดึงข้อมูลสภาพอากาศจาก Open-Meteo (ฟรี ไม่ต้อง API key)"""
import json
import urllib.request
import urllib.parse
from crewai.tools import tool


_WMO_CODES = {
    0: "ท้องฟ้าแจ่มใส", 1: "แจ่มใสเป็นส่วนใหญ่", 2: "มีเมฆบางส่วน", 3: "มีเมฆมาก",
    45: "หมอกลง", 48: "หมอกเยือกแข็ง",
    51: "ฝนละออง (เบา)", 53: "ฝนละออง (ปานกลาง)", 55: "ฝนละออง (หนัก)",
    61: "ฝนตก (เบา)", 63: "ฝนตก (ปานกลาง)", 65: "ฝนตก (หนัก)",
    71: "หิมะตก (เบา)", 73: "หิมะตก (ปานกลาง)", 75: "หิมะตก (หนัก)",
    80: "ฝนฟ้าคะนอง (เบา)", 81: "ฝนฟ้าคะนอง (ปานกลาง)", 82: "ฝนฟ้าคะนอง (หนัก)",
    95: "พายุฝนฟ้าคะนอง", 96: "พายุฝนลูกเห็บ (เบา)", 99: "พายุฝนลูกเห็บ (หนัก)",
}

_DAY_TH = ["จันทร์", "อังคาร", "พุธ", "พฤหัส", "ศุกร์", "เสาร์", "อาทิตย์"]


def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as res:
        return json.loads(res.read().decode())


def _geocode(location: str) -> tuple[float, float, str]:
    """แปลงชื่อสถานที่เป็น lat/lon"""
    q = urllib.parse.quote(location)
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=th&format=json"
    data = _fetch(url)
    results = data.get("results")
    if not results:
        raise ValueError(f"ไม่พบสถานที่ '{location}' — ลองใช้ชื่อภาษาอังกฤษ")
    r = results[0]
    name = r.get("name", location)
    country = r.get("country", "")
    label = f"{name}, {country}" if country else name
    return float(r["latitude"]), float(r["longitude"]), label


def _weather_code_label(code: int) -> str:
    return _WMO_CODES.get(code, f"รหัส {code}")


def _get_weather_data(location: str) -> str:
    try:
        lat, lon, place = _geocode(location)
    except Exception as exc:
        return f"ไม่สามารถค้นหาสถานที่ได้: {exc}"

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
        f"wind_speed_10m,wind_direction_10m,weather_code,precipitation,cloud_cover"
        f"&daily=weather_code,temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,wind_speed_10m_max"
        f"&timezone=Asia%2FBangkok&forecast_days=3"
    )

    try:
        data = _fetch(url)
    except Exception as exc:
        return f"ไม่สามารถดึงข้อมูลอากาศได้: {exc}"

    cur = data.get("current", {})
    daily = data.get("daily", {})

    # ── Current weather ──────────────────────────────────────────────────────
    temp       = cur.get("temperature_2m", "N/A")
    feels_like = cur.get("apparent_temperature", "N/A")
    humidity   = cur.get("relative_humidity_2m", "N/A")
    wind_spd   = cur.get("wind_speed_10m", "N/A")
    wind_dir   = cur.get("wind_direction_10m", "N/A")
    precip     = cur.get("precipitation", 0)
    cloud      = cur.get("cloud_cover", "N/A")
    wcode      = cur.get("weather_code", 0)
    condition  = _weather_code_label(wcode)

    lines = [
        f"[Weather] สภาพอากาศ — {place}",
        f"  สภาพ:          {condition}",
        f"  อุณหภูมิ:       {temp}°C (รู้สึกเหมือน {feels_like}°C)",
        f"  ความชื้น:       {humidity}%",
        f"  ลม:            {wind_spd} km/h ทิศ {wind_dir}°",
        f"  ฝน (ชม.นี้):    {precip} mm",
        f"  เมฆปกคลุม:      {cloud}%",
    ]

    # ── 3-day forecast ────────────────────────────────────────────────────────
    dates    = daily.get("time", [])
    wmax     = daily.get("temperature_2m_max", [])
    wmin     = daily.get("temperature_2m_min", [])
    wcodes   = daily.get("weather_code", [])
    precips  = daily.get("precipitation_sum", [])
    winds    = daily.get("wind_speed_10m_max", [])

    if dates:
        lines.append("\n  พยากรณ์ 3 วัน:")
        lines.append(f"  {'วันที่':<12} {'สภาพ':<22} {'สูงสุด':>6} {'ต่ำสุด':>6} {'ฝน':>6} {'ลมสูงสุด':>10}")
        lines.append("  " + "-" * 70)
        for i, date in enumerate(dates[:3]):
            label     = _weather_code_label(wcodes[i] if i < len(wcodes) else 0)
            t_max     = wmax[i]    if i < len(wmax)    else "N/A"
            t_min     = wmin[i]    if i < len(wmin)    else "N/A"
            rain      = precips[i] if i < len(precips) else 0
            wind_max  = winds[i]   if i < len(winds)   else "N/A"
            lines.append(
                f"  {date:<12} {label:<22} {t_max:>5}°C {t_min:>5}°C "
                f"{rain:>5.1f}mm {wind_max:>7}km/h"
            )

    return "\n".join(lines)


@tool("get_weather")
def get_weather(location: str) -> str:
    """ดึงข้อมูลสภาพอากาศปัจจุบันและพยากรณ์ 3 วัน จาก Open-Meteo (ฟรี ไม่ต้อง key)

    Args:
        location: ชื่อเมือง/จังหวัด เช่น 'อุบลราชธานี', 'Bangkok', 'เชียงใหม่'
    """
    return _get_weather_data(location)
