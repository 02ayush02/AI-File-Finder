"""Shared data structures and small text utilities used by both the indexer and the agent."""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger("filefinder")
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(h)
        root.setLevel(logging.INFO)
        root.propagate = False
    return root.getChild(name)


# --------------------------------------------------------------------------- indexing types
@dataclass
class Segment:
    """A unit of text coming out of a parser, with a human-readable location.

    unit:  "Page" | "Slide" | "Lines" | "Rows" | "Section" | "Sheet" | "Image"
    start/end: inclusive numbers in that unit (e.g. page 3..3, lines 1..40)
    label: extra locator text, e.g. a sheet name or a heading
    """
    text: str
    unit: str = ""
    start: int = 0
    end: int = 0
    label: str = ""


@dataclass
class FileMeta:
    file_path: str
    file_name: str
    file_extension: str
    created_ts: float
    modified_ts: float
    size_bytes: int
    sha256: str = ""

    @property
    def modified_date(self) -> str:
        return datetime.fromtimestamp(self.modified_ts).strftime("%Y-%m-%d")

    @property
    def created_date(self) -> str:
        return datetime.fromtimestamp(self.created_ts).strftime("%Y-%m-%d")


@dataclass
class ParsedFile:
    meta: FileMeta
    segments: list[Segment]
    warnings: list[str] = field(default_factory=list)
    ocr_pages: int = 0


@dataclass
class Chunk:
    index: int
    text: str            # raw chunk text (what the user sees as evidence)
    location: str        # e.g. "Pages 3-4", "Slide 2", "Sheet 'Q3' rows 1-50"
    tokens: int


# --------------------------------------------------------------------------- query / result types
@dataclass
class Filters:
    modified_after: Optional[float] = None     # unix ts, inclusive
    modified_before: Optional[float] = None
    file_types: list[str] = field(default_factory=list)    # extensions with dot, e.g. [".pdf", ".docx"]
    path_contains: Optional[str] = None

    def is_empty(self) -> bool:
        return not (self.modified_after or self.modified_before or self.file_types or self.path_contains)

    def describe(self) -> str:
        parts = []
        if self.modified_after:
            parts.append("modified >= " + datetime.fromtimestamp(self.modified_after).strftime("%Y-%m-%d"))
        if self.modified_before:
            parts.append("modified <= " + datetime.fromtimestamp(self.modified_before).strftime("%Y-%m-%d"))
        if self.file_types:
            parts.append("types: " + ", ".join(self.file_types))
        if self.path_contains:
            parts.append(f"path contains '{self.path_contains}'")
        return "; ".join(parts) or "none"


@dataclass
class QueryPlan:
    original: str
    semantic_query: str
    exact_keywords: list[str] = field(default_factory=list)
    filters: Filters = field(default_factory=Filters)
    planner: str = "heuristic"       # which planner produced it
    notes: str = ""


@dataclass
class ChunkHit:
    chunk_id: int
    file_path: str
    location: str
    text: str
    score: float = 0.0               # final (reranker) score, 0..1
    fusion_score: float = 0.0        # RRF score before reranking
    snippet: str = ""


@dataclass
class FileMatch:
    file_path: str
    file_name: str
    extension: str
    modified: str
    score: float                      # file-level relevance, 0..1
    hits: list[ChunkHit]              # best chunks first
    confidence: Optional[float] = None    # verifier confidence 0..1
    verified: Optional[bool] = None
    reason: str = ""
    evidence: str = ""                # short quote proving the match
    evidence_location: str = ""

    @property
    def open_uri(self) -> str:
        return Path(self.file_path).as_uri()


@dataclass
class Attempt:
    plan: QueryPlan
    matches: list[FileMatch]
    confident: bool
    top_confidence: float
    verifier: str = ""
    note: str = ""


@dataclass
class SearchResponse:
    query: str
    matches: list[FileMatch]
    confident: bool
    message: str
    attempts: list[Attempt] = field(default_factory=list)
    elapsed_s: float = 0.0


# --------------------------------------------------------------------------- text helpers
STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i
if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours out over
own same she should so some such than that the their theirs them then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your yours
find show get give need want looking look search locate document file files doc docs thing stuff something
one ones please ok where's what's remember recall called named contains containing contain includes including
""".split())


def est_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Good enough for chunk sizing without a tokenizer download."""
    return max(1, (len(text) + 3) // 4)


_WORD_RE = re.compile(r"[A-Za-z0-9\u00C0-\u024F]+(?:[._\-/][A-Za-z0-9\u00C0-\u024F]+)*")


def light_stem(w: str) -> str:
    for suf in ("ations", "ation", "ingly", "ing", "edly", "ies", "es", "ed", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def content_terms(text: str, stem: bool = True) -> list[str]:
    """Lower-cased content words (stopwords removed, optionally stemmed)."""
    out = []
    for tok in re.findall(r"[A-Za-z0-9\u00C0-\u024F]+", text.lower()):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        out.append(light_stem(tok) if stem else tok)
    return out


def code_like_tokens(text: str) -> list[str]:
    """Tokens that look like identifiers/part numbers/equations: contain digits, or hyphen/underscore/dot joins,
    or are ALL CAPS (>=2 chars)."""
    out = []
    for m in _WORD_RE.finditer(text):
        tok = m.group(0)
        has_digit = any(c.isdigit() for c in tok)
        joined = any(c in tok for c in "._-/")
        caps = tok.isupper() and len(tok) >= 2 and tok.isalpha()
        if (has_digit and len(tok) >= 2) or (joined and len(tok) >= 4) or caps:
            out.append(tok)
    return out


def snippet_around_terms(text: str, terms: list[str], max_chars: int = 320) -> str:
    """Pick the sentence window of `text` that overlaps most with `terms` and return it trimmed."""
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= max_chars:
        return flat
    sentences = re.split(r"(?<=[.!?;])\s+|\s{2,}", flat)
    if not sentences:
        return flat[:max_chars]
    stems = {light_stem(t.lower()) for t in terms if t}
    best_i, best_score = 0, -1.0
    for i, s in enumerate(sentences):
        toks = {light_stem(t) for t in re.findall(r"[a-z0-9]+", s.lower())}
        score = len(stems & toks)
        if score > best_score:
            best_i, best_score = i, score
    window = sentences[best_i]
    lo = hi = best_i
    while len(window) < max_chars * 0.7 and (lo > 0 or hi < len(sentences) - 1):
        if hi < len(sentences) - 1:
            hi += 1
            window = window + " " + sentences[hi]
        if len(window) >= max_chars * 0.7:
            break
        if lo > 0:
            lo -= 1
            window = sentences[lo] + " " + window
    if len(window) > max_chars:
        window = window[:max_chars].rsplit(" ", 1)[0] + "…"
    prefix = "… " if lo > 0 else ""
    suffix = " …" if hi < len(sentences) - 1 and not window.endswith("…") else ""
    return prefix + window + suffix
