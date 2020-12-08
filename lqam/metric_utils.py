import re
import string

RE_DET = re.compile(rf"\b(?:an?|the)\b|[{re.escape(string.punctuation)}]")


def _normalize_label(label: str) -> str:
    return RE_DET.sub("", label.lower()).strip()


def exact_match(label1: str, label2: str) -> bool:
    return _normalize_label(label1) == _normalize_label(label2)
