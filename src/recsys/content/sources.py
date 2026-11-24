from __future__ import annotations

import re
import xml.etree.ElementTree as ET


class UnsupportedSource(Exception):
    pass


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_from_html(html: str) -> str:
    html = _SCRIPT_RE.sub(" ", html)
    return _clean(_HTML_TAG_RE.sub(" ", html))


def extract_from_scorm(xml_text: str) -> dict:
    ns_stripped = re.sub(r'xmlns="[^"]+"', "", xml_text)
    root = ET.fromstring(ns_stripped)
    title_el = root.find(".//title")
    objectives = [
        _clean(el.text or "")
        for el in root.findall(".//objective/description")
        if el.text
    ]
    return {
        "title": _clean(title_el.text or "") if title_el is not None else "",
        "objectives": [o for o in objectives if o],
        "text": _clean(" ".join(t.strip() for t in root.itertext() if t.strip()))[
            :20000
        ],
    }


def extract_from_aicc(crs_text: str) -> dict:
    pairs: list[tuple[str, str]] = []
    for line in crs_text.splitlines():
        line = line.strip()
        if line and not line.startswith("[") and "=" in line:
            key, value = line.split("=", 1)
            pairs.append((key.strip(), value.strip()))
    title = ""
    for key, value in pairs:
        if key.upper().endswith("TITLE") or key.upper() == "COURSE":
            title = _clean(value)
            break
    description = _clean(" ".join(v for k, v in pairs if "DESCRIP" in k.upper()))
    return {
        "title": title,
        "objectives": [description] if description else [],
        "text": _clean(" ".join(v for _, v in pairs)),
    }


def parse_captions(vtt_or_srt: str) -> str:
    lines = []
    for line in vtt_or_srt.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line or line.isdigit():
            continue
        lines.append(line)
    return _clean(" ".join(lines))


def extract(source_type: str, payload: str) -> dict:
    st = (source_type or "").lower()
    if st == "html":
        return {"title": "", "objectives": [], "text": extract_from_html(payload)}
    if st == "scorm":
        return extract_from_scorm(payload)
    if st == "aicc":
        return extract_from_aicc(payload)
    if st in ("srt", "vtt", "transcript"):
        return {
            "title": "",
            "objectives": [],
            "text": parse_captions(payload)
            if st in ("srt", "vtt")
            else _clean(payload),
        }
    if st == "pdf":
        try:
            from pdfminer.high_level import extract_text

            return {
                "title": "",
                "objectives": [],
                "text": _clean(extract_text(payload)),
            }
        except ImportError as exc:
            raise UnsupportedSource("pdfminer.six not installed") from exc
    raise UnsupportedSource(f"unsupported source type '{source_type}'")


def estimate_duration_minutes(text: str) -> int:
    words = len(text.split())
    return max(1, round(words / 130))


MODALITY_BY_SOURCE = {
    "scorm": "elearning",
    "aicc": "elearning",
    "html": "multimedia",
    "video": "video",
    "pdf": "reading",
    "srt": "video",
    "vtt": "video",
    "transcript": "reading",
}
