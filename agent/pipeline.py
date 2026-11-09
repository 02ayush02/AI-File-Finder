"""
FileFinder = the whole system behind one object.

    finder = FileFinder()                         # loads models per config.py (with graceful fallbacks)
    finder.index_all()                            # (re)scan watched folders; unchanged files are skipped
    resp = finder.search("Find the doc with the Q3 pump efficiency calculations")
    for m in resp.matches: print(m.file_path, m.evidence_location, m.evidence)

Search = Plan -> Hybrid retrieve -> Rerank -> Verify -> (self-correct and retry) -> answer.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from agent.llm import get_llm
from agent.query_planner import QueryPlanner
from agent.reranker import get_reranker
from agent.retriever import Retriever
from agent.verifier_agent import Verifier
from common import Attempt, QueryPlan, SearchResponse, get_logger
from config import Settings, settings as default_settings
from indexer.embed_store import IndexStore, ScanStats
from indexer.embedders import get_embedder

log = get_logger("pipeline")


def _signature(plan: QueryPlan) -> tuple:
    f = plan.filters
    return (plan.semantic_query.lower(), tuple(sorted(k.lower() for k in plan.exact_keywords)),
            f.modified_after, f.modified_before, tuple(f.file_types), f.path_contains)


class FileFinder:
    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or default_settings
        self.embedder = get_embedder(self.s)
        self.store = IndexStore(self.s, self.embedder)
        self.llm = get_llm(self.s)
        self.planner = QueryPlanner(self.llm)
        self.reranker = get_reranker(self.s)
        self.retriever = Retriever(self.s, self.store, self.embedder, self.reranker)
        self.verifier = Verifier(self.s, self.llm)
        log.info("Ready. embedder=%s | reranker=%s | llm=%s", self.embedder.id,
                 getattr(self.reranker, "name", "none"), getattr(self.llm, "model", "rule-based"))

    # ------------------------------------------------------------------ indexing passthroughs
    def index_all(self, progress: Optional[Callable] = None, force: bool = False) -> ScanStats:
        return self.store.scan(self.s.watch_dirs, progress=progress, force=force)

    def index_file(self, path: str, force: bool = False):
        return self.store.index_file(path, force=force)

    def status(self) -> dict:
        d = self.store.stats()
        d.update({
            "reranker": getattr(self.reranker, "name", "none"),
            "llm": getattr(self.llm, "model", "rule-based"),
            "semantic_embeddings": self.embedder.is_semantic,
            "watch_dirs": self.s.watch_dirs,
        })
        return d

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: Optional[int] = None, verify: bool = True,
               max_retries: Optional[int] = None) -> SearchResponse:
        t0 = time.time()
        top_k = top_k or self.s.final_files
        retries = self.s.max_retries if max_retries is None else max_retries
        query = (query or "").strip()
        if not query:
            return SearchResponse(query, [], False, "Please describe the file you are looking for.")
        if self.store.stats()["files"] == 0:
            return SearchResponse(query, [], False,
                                  "The index is empty. Add folders to WATCH_DIRS and run indexing first.")

        plan = self.planner.plan(query)
        seen = {_signature(plan)}
        attempts: list[Attempt] = []

        for attempt_no in range(retries + 1):
            matches, debug = self.retriever.retrieve(plan, top_k)
            note = (f"candidates: dense={debug['dense']} bm25={debug['bm25']} exact={debug['exact']} "
                    f"-> pool {debug['pool']} reranked by {debug['reranker']}")

            if verify:
                verdict = self.verifier.verify(plan, matches)
                att = Attempt(plan, list(matches), verdict.confident, verdict.top_confidence,
                              verdict.verifier, note + (f"; {verdict.note}" if verdict.note else ""))
            else:
                top = matches[0].score if matches else 0.0
                for m in matches:
                    m.confidence = m.score
                conf = top >= self.s.confidence_threshold
                att = Attempt(plan, list(matches), conf, top, "off", note)
            attempts.append(att)
            if att.confident:
                break
            if attempt_no >= retries:
                break

            # ---- agentic self-correction: LLM-suggested rewrite first, deterministic relaxation otherwise
            nxt = (verdict.rewrite if verify else None) or self.verifier.relax(plan)
            if nxt is not None and _signature(nxt) in seen:
                nxt = self.verifier.relax(plan)
                if nxt is not None and _signature(nxt) in seen:
                    nxt = None
            if nxt is None:
                break
            seen.add(_signature(nxt))
            log.info("Attempt %d not confident (%.2f) -> retrying: %s", attempt_no + 1, att.top_confidence, nxt.notes)
            plan = nxt

        best = max(attempts, key=lambda a: (a.confident, a.top_confidence))
        resp = SearchResponse(query=query, matches=best.matches, confident=best.confident,
                              message="", attempts=attempts, elapsed_s=time.time() - t0)
        if not best.matches:
            resp.message = "No indexed file matches that description."
        elif best.confident:
            n_ok = sum(1 for m in best.matches if m.verified or (m.confidence or 0) >= self.s.confidence_threshold)
            extra = f" (after {len(attempts) - 1} self-correction round(s): {best.plan.notes})" if len(attempts) > 1 else ""
            resp.message = f"Found {n_ok} likely match(es){extra}."
        else:
            resp.message = ("No file clearly matches. These are the closest candidates - "
                            "try adding a distinctive word, number, or name you remember from inside the file.")
        return resp

    def close(self) -> None:
        self.store.close()
