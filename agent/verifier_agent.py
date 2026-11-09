"""
Verifier: "does this file actually contain what the user described?"  + agentic self-correction.

LLM mode   The top candidates and their best excerpts are shown to the local LLM, which returns per-file
           {matches, confidence, reason, evidence_quote}. Quotes are checked against the real chunk text in
           Python (an LLM that invents a quote gets it replaced by a real snippet) - so the evidence shown to the
           user is always genuine text from the file.
Heuristic  No LLM available: confidence = reranker score adjusted by whether the exact keywords really occur in the
           candidate's chunks.

If nothing is confident, `rewrite()` proposes a changed plan (LLM-suggested rewrite, or a relaxation ladder:
drop date filter -> drop type/path filters -> broaden the query). The pipeline retries with it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from common import (FileMatch, Filters, QueryPlan, content_terms, get_logger, light_stem, snippet_around_terms)

log = get_logger("verifier")


@dataclass
class Verdict:
    confident: bool
    top_confidence: float
    verifier: str
    note: str = ""
    rewrite: Optional[QueryPlan] = None     # suggestion coming straight from the LLM verifier


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


VERIFIER_SYSTEM = """You verify file-search results for a user who is looking for a file they only remember by its content.
You get the user's request and numbered candidate files, each with excerpts taken from the file.
For EACH candidate decide whether the file plausibly contains what the user described.
Be strict: related topic is not enough - the described content/specifics must be present in the excerpts.
Return ONLY JSON:
{
  "results": [
    {"id": 1, "matches": true|false, "confidence": 0.0-1.0, "reason": "one short sentence",
     "evidence_quote": "an EXACT quote (max 25 words) copied from that candidate's excerpt that proves the match, or \\"\\" if none"}
  ],
  "retry": {"semantic_query": "better search query or empty", "exact_keywords": ["..."], "drop_filters": true|false}
}
Fill "retry" only when no candidate matches: suggest a rephrased query (synonyms, different wording) and whether the
date/type filters may be the reason nothing was found. Otherwise set retry to null."""


class Verifier:
    def __init__(self, settings, llm=None):
        self.s = settings
        self.llm = llm

    # ------------------------------------------------------------------ verify
    def verify(self, plan: QueryPlan, matches: list[FileMatch]) -> Verdict:
        if not matches:
            return Verdict(False, 0.0, "none", note="no candidates retrieved")
        self._heuristic(plan, matches)          # always compute: used directly or as the fallback
        if self.llm is not None:
            try:
                return self._llm_verify(plan, matches)
            except Exception as e:
                log.warning("LLM verification failed (%s: %s); using heuristic confidence.", type(e).__name__, e)
                v = self._finish(matches, "heuristic")
                v.note = f"LLM verifier failed ({type(e).__name__}); heuristic used"
                return v
        return self._finish(matches, "heuristic")

    def _finish(self, matches: list[FileMatch], name: str) -> Verdict:
        matches.sort(key=lambda m: ((m.verified is True), m.confidence or 0.0, m.score), reverse=True)
        top = max((m.confidence or 0.0) for m in matches)
        confident = any(m.verified for m in matches)
        return Verdict(confident, float(top), name)

    # ------------------------------------------------------------------ heuristic
    def _heuristic(self, plan: QueryPlan, matches: list[FileMatch]) -> None:
        q_terms = sorted(set(content_terms(plan.semantic_query)))
        kws = [k for k in plan.exact_keywords if _alnum(k)]
        thr = self.s.confidence_threshold
        for m in matches:
            text = "\n".join(h.text for h in m.hits)
            flat = _alnum(text)
            stems = set(content_terms(text))
            found_terms = [t for t in q_terms if t in stems]
            found_kw = [k for k in kws if _alnum(k) in flat]
            missing_kw = [k for k in kws if k not in found_kw]
            conf = m.score
            if kws:
                conf *= 0.55 + 0.45 * (len(found_kw) / len(kws))
            m.confidence = float(min(1.0, conf))
            m.verified = m.confidence >= thr
            bits = [f"covers {len(found_terms)}/{len(q_terms)} key terms" if q_terms else "no key terms"]
            if kws:
                bits.append(f"exact match: {', '.join(found_kw) or 'none'}"
                            + (f" (missing: {', '.join(missing_kw)})" if missing_kw else ""))
            m.reason = "; ".join(bits)
            best = m.hits[0]
            m.evidence = best.snippet
            m.evidence_location = best.location

    # ------------------------------------------------------------------ LLM
    def _llm_verify(self, plan: QueryPlan, matches: list[FileMatch]) -> Verdict:
        cands = matches[:5]
        terms = content_terms(plan.semantic_query, stem=False) + plan.exact_keywords
        blocks = []
        for i, m in enumerate(cands, start=1):
            ex = []
            for h in m.hits[:2]:
                ex.append(f'  excerpt ({h.location or "text"}): "{snippet_around_terms(h.text, terms, 900)}"')
            blocks.append(f"[{i}] {m.file_path} (modified {m.modified})\n" + "\n".join(ex))
        user = f"USER REQUEST: {plan.original}\n\nCANDIDATES:\n" + "\n\n".join(blocks)
        raw = self.llm.chat_json(VERIFIER_SYSTEM, user)
        results = raw.get("results")
        if not isinstance(results, list):
            raise ValueError("verifier JSON missing 'results'")

        seen: set[int] = set()
        for r in results:
            if not isinstance(r, dict):
                continue
            try:
                idx = int(r.get("id")) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= idx < len(cands) or idx in seen:
                continue
            seen.add(idx)
            m = cands[idx]
            try:
                conf = float(r.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            ok = bool(r.get("matches")) and conf >= self.s.confidence_threshold
            m.confidence, m.verified = conf, ok
            reason = str(r.get("reason") or "").strip()
            if reason:
                m.reason = reason
            quote = str(r.get("evidence_quote") or "").strip()
            if quote:
                for h in m.hits:                      # accept the quote only if it is really in the file text
                    if _norm_ws(quote) and _norm_ws(quote) in _norm_ws(h.text):
                        m.evidence, m.evidence_location = quote, h.location
                        break
                else:
                    m.reason += " (quote not found verbatim; showing the best matching passage instead)"
        for i, m in enumerate(cands):
            if i not in seen:                         # LLM skipped it: demote
                m.confidence = (m.confidence or 0.0) * 0.5
                m.verified = False
        for m in matches[5:]:
            m.confidence = (m.confidence or 0.0) * 0.5
            m.verified = False

        v = self._finish(matches, f"llm:{getattr(self.llm, 'model', '?')}")
        retry = raw.get("retry")
        if not v.confident and isinstance(retry, dict):
            sq = retry.get("semantic_query")
            kws = retry.get("exact_keywords") or []
            sq = sq.strip() if isinstance(sq, str) and sq.strip() else plan.semantic_query
            kws = [k.strip() for k in kws if isinstance(k, str) and k.strip()][:6] if isinstance(kws, list) else []
            filters = Filters() if retry.get("drop_filters") else plan.filters
            v.rewrite = QueryPlan(original=plan.original, semantic_query=sq, exact_keywords=kws or plan.exact_keywords,
                                  filters=filters, planner=f"rewrite:{v.verifier}",
                                  notes="LLM-suggested rewrite" + (" (filters dropped)" if retry.get("drop_filters") else ""))
        return v

    # ------------------------------------------------------------------ self-correction ladder
    def relax(self, plan: QueryPlan) -> Optional[QueryPlan]:
        """Deterministic fallback rewrite: loosen the constraints most likely to have excluded the right file."""
        f = plan.filters
        if f.modified_after or f.modified_before:
            return QueryPlan(plan.original, plan.semantic_query, plan.exact_keywords,
                             Filters(file_types=f.file_types, path_contains=f.path_contains),
                             planner="relaxed", notes="dropped the date filter (your memory of the date may be off)")
        if f.file_types or f.path_contains:
            return QueryPlan(plan.original, plan.semantic_query, plan.exact_keywords, Filters(),
                             planner="relaxed", notes="dropped file-type / path filters")
        terms = sorted(set(content_terms(plan.semantic_query, stem=False)), key=len, reverse=True)[:3]
        broadened = " ".join(plan.exact_keywords + terms).strip()
        if broadened and broadened.lower() != plan.semantic_query.lower():
            return QueryPlan(plan.original, broadened, plan.exact_keywords, Filters(),
                             planner="relaxed", notes="broadened to the most distinctive terms")
        return None
