"""
Rerankers score (query, passage) pairs jointly, which is far more precise than comparing independent vectors.

    CrossEncoderReranker   BAAI/bge-reranker-v2-m3 via sentence-transformers (probability 0..1)
    LexicalReranker        offline fallback: stem coverage + bigram adjacency + exact-keyword presence (0..1)

Both return scores on a 0..1 scale so downstream confidence thresholds mean roughly the same thing.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from common import content_terms, get_logger, light_stem

log = get_logger("rerank")


class Reranker(ABC):
    name: str = ""

    @abstractmethod
    def score(self, query: str, passages: Sequence[str], keywords: Sequence[str] = ()) -> list[float]: ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model: str, max_length: int = 1024):
        from sentence_transformers import CrossEncoder  # heavy import, only when actually used
        log.info("Loading cross-encoder %s (first use downloads the weights)", model)
        self.name = f"cross_encoder:{model}"
        self._ce = CrossEncoder(model, max_length=max_length)

    def score(self, query: str, passages: Sequence[str], keywords: Sequence[str] = ()) -> list[float]:
        if not passages:
            return []
        raw = self._ce.predict([(query, p) for p in passages], batch_size=8, show_progress_bar=False)
        vals = [float(x) for x in raw]
        if any(v < 0.0 or v > 1.0 for v in vals):     # raw logits -> probabilities
            vals = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, v)))) for v in vals]
        return vals


def _norm_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


class LexicalReranker(Reranker):
    name = "lexical"

    def score(self, query: str, passages: Sequence[str], keywords: Sequence[str] = ()) -> list[float]:
        q_terms = content_terms(query)
        q_set = set(q_terms)
        q_bigrams = {(a, b) for a, b in zip(q_terms, q_terms[1:])}
        kw_norm = [_norm_alnum(k) for k in keywords if _norm_alnum(k)]
        out: list[float] = []
        for p in passages:
            p_terms = content_terms(p)
            p_set = set(p_terms)
            cov = (len(q_set & p_set) / len(q_set)) if q_set else 0.0
            p_bigrams = set(zip(p_terms, p_terms[1:]))
            big = (len(q_bigrams & p_bigrams) / len(q_bigrams)) if q_bigrams else cov
            if kw_norm:
                flat = _norm_alnum(p)
                kwscore = sum(1 for k in kw_norm if k in flat) / len(kw_norm)
                s = 0.55 * cov + 0.20 * big + 0.25 * kwscore
            else:
                s = 0.75 * cov + 0.25 * big
            out.append(min(1.0, s))
        return out


def get_reranker(settings) -> Optional[Reranker]:
    backend = settings.reranker_backend.lower()
    if backend == "none":
        return None
    if backend == "lexical":
        return LexicalReranker()
    if backend not in ("auto", "cross_encoder"):
        raise ValueError(f"Unknown reranker_backend: {settings.reranker_backend}")
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        if backend == "cross_encoder":
            raise RuntimeError("reranker_backend=cross_encoder needs: pip install sentence-transformers")
        log.warning("sentence-transformers not installed -> using the lexical reranker. "
                    "For best precision: pip install sentence-transformers (uses %s).", settings.reranker_model)
        return LexicalReranker()
    try:
        return CrossEncoderReranker(settings.reranker_model, settings.reranker_max_length)
    except Exception as e:
        if backend == "cross_encoder":
            raise
        log.warning("Could not load cross-encoder %s (%s: %s) -> using the lexical reranker.",
                    settings.reranker_model, type(e).__name__, e)
        return LexicalReranker()
