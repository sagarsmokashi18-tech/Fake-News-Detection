import re


def normalize_text(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
