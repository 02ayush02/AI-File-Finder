"""
Query planner: natural-language request -> structured search plan.

    "Find that document from last month where I calculated the stress tensor for the gear shaft"
 -> semantic_query : "calculated stress tensor gear shaft"
    exact_keywords : ["stress tensor"]
    filters        : modified_after = first day of last month

Two implementations behind one interface:
  * LLM planner    (Ollama, JSON output, today's date injected so "last month" resolves correctly)
  * Heuristic      (regex/rule based, no model needed; also used if the LLM call fails or returns garbage)

Whatever the LLM returns is *validated and sanitised* in Python (dates parsed, types normalised, lengths capped) -
a small local model that hallucinates a field cannot break retrieval.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from common import STOPWORDS, Filters, QueryPlan, code_like_tokens, get_logger

log = get_logger("planner")

# --------------------------------------------------------------------------- file type vocabulary
TYPE_WORDS: dict[str, list[str]] = {
    "pdf": [".pdf"], "pdfs": [".pdf"],
    "word": [".docx"], "docx": [".docx"],
    "excel": [".xlsx", ".xlsm"], "xlsx": [".xlsx", ".xlsm"], "workbook": [".xlsx", ".xlsm"],
    "spreadsheet": [".xlsx", ".xlsm", ".csv", ".tsv"], "spreadsheets": [".xlsx", ".xlsm", ".csv", ".tsv"],
    "csv": [".csv", ".tsv"], "sheet": [".xlsx", ".xlsm", ".csv"], "sheets": [".xlsx", ".xlsm", ".csv"],
    "powerpoint": [".pptx"], "pptx": [".pptx"], "slides": [".pptx"], "slide": [".pptx"],
    "presentation": [".pptx"], "deck": [".pptx"],
    "image": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"],
    "photo": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
    "picture": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
    "screenshot": [".png", ".jpg", ".jpeg"],
    "jpg": [".jpg", ".jpeg"], "jpeg": [".jpg", ".jpeg"], "png": [".png"],
    "markdown": [".md"], "txt": [".txt"],
    "python": [".py"], "script": [".py", ".js", ".sh", ".ts"],
}
_CODE_EXTS = [".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".sql", ".sh"]
TYPE_WORDS["code"] = _CODE_EXTS
# words that name a type but are *also* normal English: only treat as a type filter in "<word> file/document"
_STRICT_TYPES = {"word"}   # only a type when followed by file/document/doc ("word doc")
_AMBIGUOUS_TYPES = {"word", "deck", "slide", "slides", "script", "python", "image", "picture", "sheet", "sheets"}

_EXT_ALIASES = {
    "pdf": [".pdf"], "word": [".docx"], "doc": [".docx"], "docx": [".docx"], "excel": [".xlsx", ".xlsm"],
    "xlsx": [".xlsx", ".xlsm"], "spreadsheet": [".xlsx", ".xlsm", ".csv", ".tsv"], "csv": [".csv", ".tsv"],
    "powerpoint": [".pptx"], "pptx": [".pptx"], "ppt": [".pptx"], "presentation": [".pptx"], "slides": [".pptx"],
    "image": TYPE_WORDS["image"], "photo": TYPE_WORDS["photo"], "png": [".png"], "jpg": [".jpg", ".jpeg"],
    "text": [".txt", ".md"], "txt": [".txt"], "markdown": [".md"], "md": [".md"], "code": _CODE_EXTS,
    "python": [".py"], "py": [".py"],
}

UNITS = {"mpa", "kpa", "gpa", "kw", "mw", "nm", "rpm", "hz", "khz", "mhz", "ghz", "psi", "kn", "mm", "cm", "kwh"}
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_NOT_NAMES = {"i", "i'm", "i've", "excel", "word", "powerpoint", "pdf", "python", "monday", "tuesday", "wednesday",
              "thursday", "friday", "saturday", "sunday", "find", "show", "where", "what", "which", "the",
              "q1", "q2", "q3", "q4"} | set(MONTHS)


def normalize_file_types(types) -> list[str]:
    out: list[str] = []
    for t in types or []:
        if not isinstance(t, str):
            continue
        t = t.strip().lower().lstrip("*")
        if not t:
            continue
        exts = _EXT_ALIASES.get(t.lstrip("."))
        if exts is None:
            exts = [t if t.startswith(".") else "." + t]
        for e in exts:
            if e not in out:
                out.append(e)
    return out


# --------------------------------------------------------------------------- date helpers
def _sod(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_start(year: int, month: int) -> datetime:
    return datetime(year, month, 1)


def _month_end(year: int, month: int) -> datetime:
    nxt = datetime(year + (month == 12), (month % 12) + 1, 1)
    return nxt - timedelta(seconds=1)


def _shift_months(d: datetime, n: int) -> datetime:
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return d.replace(year=y, month=m, day=day)


def _last_month_start(now: datetime) -> datetime:
    y, m = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
    return datetime(y, m, 1)


def parse_time_expressions(text: str, now: datetime) -> tuple[Optional[float], Optional[float], str, list[str]]:
    """Return (after_ts, before_ts, text_without_time_phrases, notes).

    "last/this <period>" and "past N units" give a forgiving open-ended `after` bound (people say "last month"
    loosely); explicit months/years/ISO dates give closed ranges.
    """
    t = text
    after: Optional[datetime] = None
    before: Optional[datetime] = None
    notes: list[str] = []

    def cut(m: re.Match) -> None:
        nonlocal t
        t = t[:m.start()] + " " + t[m.end():]

    def search(pattern: str) -> Optional[re.Match]:
        return re.search(pattern, t, flags=re.I)

    # explicit ISO dates with from/since/before
    m = search(r"\b(?:from|since|after|on or after)\s+(\d{4}-\d{2}-\d{2})\b")
    if m:
        after = datetime.strptime(m.group(1), "%Y-%m-%d"); cut(m); notes.append(f"after {m.group(1)}")
    m = search(r"\b(?:before|until|up to|by)\s+(\d{4}-\d{2}-\d{2})\b")
    if m:
        before = datetime.strptime(m.group(1), "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
        cut(m); notes.append(f"before {m.group(1)}")
    m = search(r"\b(?:on\s+)?(\d{4}-\d{2}-\d{2})\b")
    if m and after is None and before is None:
        d = datetime.strptime(m.group(1), "%Y-%m-%d")
        after, before = d, d + timedelta(days=1) - timedelta(seconds=1); cut(m); notes.append(f"on {m.group(1)}")

    # "since March (2025)" / "after march"
    m = search(rf"\b(?:since|after)\s+({_MONTH_RE})\b(?:\s+(\d{{4}}))?")
    if m and after is None:
        mon = MONTHS[m.group(1).lower()]
        yr = int(m.group(2)) if m.group(2) else (now.year if mon <= now.month else now.year - 1)
        after = _month_start(yr, mon); cut(m); notes.append(f"since {m.group(1)} {yr}")

    # "in October (2025)", "during march", "from october 2025"
    m = search(rf"\b(?:in|during|from|of|back in)\s+({_MONTH_RE})\b(?:\s+(\d{{4}}))?") \
        or search(rf"\b({_MONTH_RE})\s+(\d{{4}})\b")
    if m and after is None and before is None:
        mon = MONTHS[m.group(1).lower()]
        yr = int(m.group(2)) if m.group(2) else (now.year if mon <= now.month else now.year - 1)
        after, before = _month_start(yr, mon), _month_end(yr, mon); cut(m); notes.append(f"in {m.group(1)} {yr}")

    # "in 2025"
    m = search(r"\b(?:in|during|from|of)\s+(20\d{2}|19\d{2})\b")
    if m and after is None and before is None:
        yr = int(m.group(1))
        after, before = datetime(yr, 1, 1), datetime(yr, 12, 31, 23, 59, 59); cut(m); notes.append(f"in {yr}")

    # "past/last N days|weeks|months|years"
    m = search(r"\b(?:in the\s+)?(?:past|last|previous|within the last)\s+(\d+)\s+(day|week|month|year)s?\b")
    if m and after is None:
        n, unit = int(m.group(1)), m.group(2)
        after = (now - timedelta(days=n * {"day": 1, "week": 7, "year": 365}[unit])) if unit != "month" \
            else _shift_months(now, -n)
        cut(m); notes.append(f"past {n} {unit}s")

    # "N days|weeks|months ago"
    m = search(r"\b(\d+)\s+(day|week|month|year)s?\s+ago\b")
    if m and after is None:
        n, unit = int(m.group(1)), m.group(2)
        days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        after, before = now - timedelta(days=days * 1.5 + 1), now - timedelta(days=days * 0.5)
        cut(m); notes.append(f"about {n} {unit}s ago")

    # relative single phrases
    rel = [
        (r"\b(?:from\s+|since\s+|during\s+|in\s+)?yesterday\b", lambda: _sod(now - timedelta(days=1))),
        (r"\b(?:from\s+|since\s+|during\s+)?today\b", lambda: _sod(now)),
        (r"\b(?:from\s+|since\s+|during\s+|in\s+)?last\s+week\b", lambda: _sod(now - timedelta(days=now.weekday() + 7))),
        (r"\b(?:from\s+|since\s+|during\s+|in\s+)?this\s+week\b", lambda: _sod(now - timedelta(days=now.weekday()))),
        (r"\b(?:from\s+|since\s+|during\s+|in\s+)?last\s+month\b", lambda: _last_month_start(now)),
        (r"\b(?:from\s+|since\s+|during\s+|in\s+)?this\s+month\b", lambda: _month_start(now.year, now.month)),
        (r"\b(?:from\s+|since\s+|during\s+|in\s+)?last\s+year\b", lambda: datetime(now.year - 1, 1, 1)),
        (r"\b(?:from\s+|since\s+|during\s+|in\s+)?this\s+year\b", lambda: datetime(now.year, 1, 1)),
        (r"\b(?:very\s+)?recent(?:ly)?\b|\blately\b", lambda: now - timedelta(days=30)),
    ]
    for pat, fn in rel:
        m = search(pat)
        if m and after is None:
            after = fn(); cut(m); notes.append(m.group(0).strip())

    return (after.timestamp() if after else None, before.timestamp() if before else None,
            re.sub(r"\s+", " ", t).strip(), notes)


# =========================================================================== planners
class HeuristicPlanner:
    name = "heuristic"

    def plan(self, query: str, now: Optional[datetime] = None) -> QueryPlan:
        now = now or datetime.now()
        notes: list[str] = []
        text = query.strip()

        # 1) quoted strings are exact keywords
        exact: list[str] = []
        for m in re.finditer(r"[\"\u201c\u201d]([^\"\u201c\u201d]{2,80})[\"\u201c\u201d]", text):
            exact.append(m.group(1).strip())
        text = re.sub(r"[\"\u201c\u201d]", " ", text)

        # 2) time expressions (use the corrected 'last month' logic)
        after, before, text, tnotes = parse_time_expressions(text, now)
        notes += tnotes

        # 3) file types
        types: list[str] = []
        words = re.findall(r"[A-Za-z0-9\u00C0-\u024F]+(?:[._\-/][A-Za-z0-9\u00C0-\u024F]+)*", text)
        low = [w.lower() for w in words]
        drop_idx: set[int] = set()
        for i, w in enumerate(low):
            key = w.lstrip(".")
            if key in TYPE_WORDS:
                nxt = low[i + 1] if i + 1 < len(low) else ""
                prev = low[i - 1] if i > 0 else ""
                if key in _AMBIGUOUS_TYPES:
                    explicit = nxt in ("file", "files", "document", "doc", "docs", "format", "version")
                    article = prev in ("a", "the", "my", "an") and key not in _STRICT_TYPES
                    if not (explicit or article):
                        continue
                for e in TYPE_WORDS[key]:
                    if e not in types:
                        types.append(e)
                drop_idx.add(i)
        if types:
            notes.append("file types " + ",".join(types))

        # 4) exact keywords: code-like tokens, units, proper-noun phrases
        for tok in code_like_tokens(" ".join(w for j, w in enumerate(words) if j not in drop_idx)):
            if tok.lower() in ("i",) or (tok.isdigit() and len(tok) < 4):
                continue
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", tok):
                continue
            exact.append(tok)
        for w in words:
            if w.lower() in UNITS:
                exact.append(w)
        exact.extend(self._proper_noun_phrases(query))

        # 5) semantic query: content words, minus time/type phrases
        kept = [w for j, w in enumerate(words) if j not in drop_idx and w.lower() not in STOPWORDS and len(w) > 1]
        semantic = " ".join(kept).strip() or re.sub(r"\s+", " ", text).strip() or query.strip()

        dedup: list[str] = []
        seen: set[str] = set()
        for k in exact:
            kk = k.strip()
            if kk and kk.lower() not in seen and len(kk) <= 60:
                seen.add(kk.lower()); dedup.append(kk)
        return QueryPlan(original=query, semantic_query=semantic, exact_keywords=dedup[:8],
                         filters=Filters(modified_after=after, modified_before=before, file_types=types),
                         planner="heuristic", notes="; ".join(notes))

    @staticmethod
    def _proper_noun_phrases(query: str) -> list[str]:
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]*|[^\sA-Za-z]+", query)
        out, cur = [], []
        words = [t for t in toks if re.match(r"[A-Za-z]", t)]
        for idx, w in enumerate(words):
            is_cap = w[0].isupper() and not w.isupper() and w.lower() not in _NOT_NAMES and w.lower() not in STOPWORDS
            if is_cap and idx > 0:
                cur.append(w)
            else:
                if cur:
                    out.append(" ".join(cur))
                cur = []
        if cur:
            out.append(" ".join(cur))
        return [re.sub(r"['\u2019]s$", "", o) for o in out]


PLANNER_SYSTEM = """You convert a user's description of a file they want to find into a JSON search plan.
Today's date is {today} (weekday: {weekday}). Resolve relative dates against it.
Return ONLY a JSON object with exactly these keys:
{{
  "semantic_query": string,        // concise natural-language query: the concepts/content to match (no dates, no filler words)
  "exact_keywords": [string],      // up to 6 literal strings that should appear verbatim: part numbers, codes, formulas, names, units, quoted phrases. [] if none
  "modified_after": string|null,   // YYYY-MM-DD or null
  "modified_before": string|null,  // YYYY-MM-DD or null
  "file_types": [string],          // from: pdf, docx, xlsx, csv, pptx, image, txt, md, code. [] if the user did not say
  "path_contains": string|null     // folder/file-name fragment only if the user explicitly mentions one
}}
Rules: only set date or type filters when the user clearly states them. "last month" means modified_after = first day of the previous month.
Do not invent keywords that are not in the request. No explanations, JSON only.

Example 1
User: Find that document from last month where I calculated the stress tensor for the gear shaft
JSON: {{"semantic_query": "stress tensor calculation gear shaft mechanical design", "exact_keywords": ["stress tensor"], "modified_after": "{last_month_start}", "modified_before": null, "file_types": [], "path_contains": null}}

Example 2
User: the PDF quote that lists part XJ-4471-B
JSON: {{"semantic_query": "supplier quote bearing part number", "exact_keywords": ["XJ-4471-B"], "modified_after": null, "modified_before": null, "file_types": ["pdf"], "path_contains": null}}
"""


class LLMPlanner:
    name = "llm"

    def __init__(self, llm, fallback: Optional[HeuristicPlanner] = None):
        self.llm = llm
        self.fallback = fallback or HeuristicPlanner()

    def plan(self, query: str, now: Optional[datetime] = None) -> QueryPlan:
        now = now or datetime.now()
        system = PLANNER_SYSTEM.format(today=now.strftime("%Y-%m-%d"), weekday=now.strftime("%A"),
                                       last_month_start=_last_month_start(now).strftime("%Y-%m-%d"))
        try:
            raw = self.llm.chat_json(system, query)
            plan = self._sanitize(raw, query)
            plan.planner = f"llm:{getattr(self.llm, 'model', '?')}"
            return plan
        except Exception as e:
            log.warning("LLM planning failed (%s: %s); using rule-based planner.", type(e).__name__, e)
            p = self.fallback.plan(query, now)
            p.notes = (p.notes + "; " if p.notes else "") + f"LLM planner failed: {type(e).__name__}"
            return p

    @staticmethod
    def _sanitize(raw: dict, query: str) -> QueryPlan:
        sq = raw.get("semantic_query")
        sq = sq.strip() if isinstance(sq, str) and sq.strip() else query.strip()
        kws = raw.get("exact_keywords") or []
        if isinstance(kws, str):
            kws = [kws]
        exact, seen = [], set()
        for k in kws:
            if isinstance(k, str):
                k = k.strip()
                if 1 < len(k) <= 60 and k.lower() not in seen:
                    seen.add(k.lower()); exact.append(k)

        def to_ts(v, end=False):
            if not isinstance(v, str) or not v.strip():
                return None
            try:
                d = datetime.strptime(v.strip()[:10], "%Y-%m-%d")
            except ValueError:
                return None
            return (d + timedelta(days=1) - timedelta(seconds=1)).timestamp() if end else d.timestamp()

        after, before = to_ts(raw.get("modified_after")), to_ts(raw.get("modified_before"), end=True)
        if after and before and after > before:
            after, before = before, after
        ftypes = raw.get("file_types") or []
        if isinstance(ftypes, str):
            ftypes = [ftypes]
        pc = raw.get("path_contains")
        pc = pc.strip() if isinstance(pc, str) and pc.strip() else None
        return QueryPlan(original=query, semantic_query=sq, exact_keywords=exact[:8],
                         filters=Filters(modified_after=after, modified_before=before,
                                         file_types=normalize_file_types(ftypes), path_contains=pc))


class QueryPlanner:
    """Facade: picks the LLM planner when an LLM is available, otherwise the heuristic one."""

    def __init__(self, llm=None):
        self.heuristic = HeuristicPlanner()
        self.impl = LLMPlanner(llm, self.heuristic) if llm is not None else self.heuristic

    def plan(self, query: str, now: Optional[datetime] = None) -> QueryPlan:
        return self.impl.plan(query, now)
