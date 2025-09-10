# utils_text.py
import re

_URL_RE   = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_NUM_RE   = re.compile(r"\b\d+(\.\d+)?\b")
_PUNC_RE  = re.compile(r"[^\w\s]")   # keep letters/numbers/space, drop punctuation
_WS_RE    = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = _URL_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    s = _NUM_RE.sub(" ", s)
    s = _PUNC_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s
