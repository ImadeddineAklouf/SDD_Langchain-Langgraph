"""Découpage déterministe (sans LLM) d'un texte en énoncés sources numérotés."""
from __future__ import annotations

import re

from .models import SourceStatement

_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])")


def split_statements(text: str, prefix: str, origin: str, start: int = 1) -> list[SourceStatement]:
    out: list[SourceStatement] = []
    n = start
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = _BULLET.sub("", line)
        for sent in _SENT.split(line):
            sent = sent.strip()
            if sent:
                out.append(SourceStatement(id=f"{prefix}-{n:03d}", text=sent, origin=origin))
                n += 1
    return out
