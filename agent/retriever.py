"""
Hybrid retrieval + reranking + chunk-to-file aggregation.

    plan --> dense vector search  ┐
         --> BM25 (all terms)     ├─ Reciprocal Rank Fusion --> top `rerank_pool` chunks --> reranker
         --> BM25 (exact phrases) ┘                                                        |
                                                  group by parent file_path <---------------┘
                                                  file score = 0.7 * best chunk + 0.3 * mean(top-N chunks)

Dense search catches concepts ("thermal dissipation" ~ "heat loss"); BM25 catches literal strings (part numbers,
formulas, names); the third list rewards chunks that contain the planner's *exact keywords* as phrases.
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

from agent.reranker import Reranker
from common import (ChunkHit, FileMatch, QueryPlan, content_terms, get_logger, snippet_around_terms)
from indexer.embed_store import IndexStore, build_fts_query
from indexer.embedders import Embedder

log = get_logger("retriever")


def reciprocal_rank_fusion(ranked_lists: list[tuple[list[int], float]], k: int = 60) -> dict[int, float]:
    """ranked_lists: [(ids best-first, weight), ...] -> {id: fused score}."""
    fused: dict[int, float] = defaultdict(float)
    for ids, weight in ranked_lists:
        for rank, cid in enumerate(ids, start=1):
            fused[cid] += weight / (k + rank)
    return dict(fused)


class Retriever:
    def __init__(self, settings, store: IndexStore, embedder: Embedder, reranker: Optional[Reranker]):
        self.s = settings
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    # ------------------------------------------------------------------
    def retrieve(self, plan: QueryPlan, top_k: Optional[int] = None) -> tuple[list[FileMatch], dict]:
        s = self.s
        top_k = top_k or s.final_files
        debug: dict = {"dense": 0, "bm25": 0, "exact": 0, "pool": 0, "reranker": getattr(self.reranker, "name", "none")}

        sem = plan.semantic_query.strip() or plan.original
        kw = [k for k in plan.exact_keywords if k]
        dense_text = sem if all(k.lower() in sem.lower() for k in kw) else f"{sem} {' '.join(kw)}"

        # --- three candidate lists --------------------------------------------------------------
        lists: list[tuple[list[int], float]] = []
        dense = self.store.dense_search(self.embedder.embed_query(dense_text), s.dense_candidates, plan.filters)
        debug["dense"] = len(dense)
        if dense:
            lists.append(([i for i, _ in dense], 1.0))

        terms = content_terms(sem, stem=False)
        bm25_q = build_fts_query(terms=terms + [t for k in kw for t in k.split()], phrases=kw)
        bm25 = self.store.sparse_search(bm25_q, s.sparse_candidates, plan.filters)
        debug["bm25"] = len(bm25)
        if bm25:
            lists.append(([i for i, _ in bm25], 1.0))

        if kw:
            exact = self.store.sparse_search(build_fts_query(phrases=kw), s.sparse_candidates, plan.filters)
            debug["exact"] = len(exact)
            if exact:
                lists.append(([i for i, _ in exact], 0.8))

        if not lists:
            return [], debug

        fused = reciprocal_rank_fusion(lists, s.rrf_k)
        pool_ids = [cid for cid, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:s.rerank_pool]]
        rows = self.store.get_chunks(pool_ids)
        pool_ids = [i for i in pool_ids if i in rows]          # drop vectors whose chunk row vanished
        debug["pool"] = len(pool_ids)
        if not pool_ids:
            return [], debug

        # --- rerank -----------------------------------------------------------------------------
        rerank_query = sem if all(k.lower() in sem.lower() for k in kw) else f"{sem} {' '.join(kw)}"
        if self.reranker is not None:
            passages = [f"{os.path.basename(rows[i]['file_path'])} | {rows[i]['location']}\n{rows[i]['text']}"
                        for i in pool_ids]
            scores = self.reranker.score(rerank_query, passages, keywords=kw)
        else:
            mx = max(fused[i] for i in pool_ids) or 1.0
            scores = [fused[i] / mx for i in pool_ids]

        # --- group by parent file ----------------------------------------------------------------
        snippet_terms = terms + kw
        by_file: dict[str, list[ChunkHit]] = defaultdict(list)
        for cid, sc in zip(pool_ids, scores):
            r = rows[cid]
            by_file[r["file_path"]].append(ChunkHit(
                chunk_id=cid, file_path=r["file_path"], location=r["location"] or "", text=r["text"],
                score=float(sc), fusion_score=float(fused[cid]),
                snippet=snippet_around_terms(r["text"], snippet_terms)))

        matches: list[FileMatch] = []
        for path, hits in by_file.items():
            if not os.path.exists(path):         # stale entry: never recommend a dead link
                log.info("Dropping stale index entry for missing file: %s", path)
                self.store.remove_file(path)
                continue
            hits.sort(key=lambda h: h.score, reverse=True)
            best = hits[0].score
            top_n = [h.score for h in hits[:s.file_score_top_chunks]]
            file_score = 0.7 * best + 0.3 * (sum(top_n) / len(top_n))
            frow = self.store.get_file(path)
            matches.append(FileMatch(
                file_path=path, file_name=os.path.basename(path),
                extension=(frow["ext"] if frow else os.path.splitext(path)[1].lower()),
                modified=(datetime.fromtimestamp(frow["modified_ts"]).strftime("%Y-%m-%d") if frow else ""),
                score=float(file_score), hits=hits[:3]))
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k], debug
