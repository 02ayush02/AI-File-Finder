"""
Chunk -> embed -> upsert, plus all read paths used by the retriever.

Two stores kept in sync, both keyed by the same integer chunk id:

  SQLite  (storage/index.sqlite)      files table (path, sha256, mtime, ...), chunks table (raw text, location,
                                      parent file_path) and an FTS5 table = the BM25 keyword index.
  Chroma  (storage/vector_db/)        one dense vector per chunk (+ small metadata for filtering).

Every chunk row stores its parent `file_path`, so any hit can be traced back to a real file.

Engineering rules implemented here (see README "Edge cases"):
  * No duplicate re-indexing: a file is skipped when (size, mtime) are unchanged, and re-hashed (SHA-256) when
    they differ; unchanged content is never re-embedded (only its date metadata is refreshed).
  * Deleted / moved files: `remove_file`, `move_file`, and `remove_missing` purge their chunks from BOTH stores,
    so the agent never recommends a dead link.
  * Embedder changes (different model/dimension) are detected and the derived index is rebuilt automatically.
  * Crash safety: embeddings are computed *before* any destructive write; the `files` row is written last, so an
    interrupted file is simply re-indexed on the next scan.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from common import Filters, ParsedFile, get_logger
from indexer.chunker import chunk_segments
from indexer.embedders import Embedder
from indexer.parser import file_meta, file_sha256, iter_files, parse_file, should_index

log = get_logger("store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, name TEXT, ext TEXT, size INTEGER, created_ts REAL, modified_ts REAL,
    sha256 TEXT, n_chunks INTEGER, indexed_at REAL, ocr_pages INTEGER DEFAULT 0, warnings TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, file_path TEXT NOT NULL, chunk_idx INTEGER, location TEXT,
    text TEXT, tokens INTEGER, modified_ts REAL, ext TEXT
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS chunks_mod ON chunks(modified_ts);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(body, tokenize = 'porter unicode61 remove_diacritics 2');
"""


@dataclass
class IndexResult:
    path: str
    status: str                 # indexed | unchanged | metadata_updated | skipped | failed | removed
    n_chunks: int = 0
    ocr_pages: int = 0
    message: str = ""


@dataclass
class ScanStats:
    seen: int = 0
    indexed: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.seen} files seen: {self.indexed} indexed, {self.unchanged} unchanged, "
                f"{self.skipped} skipped, {self.failed} failed, {self.removed} removed "
                f"({self.seconds:.1f}s)")


def fts_escape(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def build_fts_query(terms: Iterable[str] = (), phrases: Iterable[str] = ()) -> str:
    """OR-query for FTS5. `terms` are single words; `phrases` are exact multi-token strings like 'AB-1234'
    (tokenised the same way FTS tokenises, then matched as an adjacent phrase)."""
    parts: list[str] = []
    seen: set[str] = set()
    for p in phrases:
        toks = re.findall(r"[A-Za-z0-9\u00C0-\u024F]+", p)
        if not toks:
            continue
        q = fts_escape(" ".join(toks))
        if q not in seen:
            seen.add(q)
            parts.append(q)
    for t in terms:
        for tok in re.findall(r"[A-Za-z0-9\u00C0-\u024F]+", t):
            q = fts_escape(tok)
            if q not in seen:
                seen.add(q)
                parts.append(q)
    return " OR ".join(parts)


class IndexStore:
    def __init__(self, settings, embedder: Embedder):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings
        self.embedder = embedder
        self._lock = threading.RLock()
        self.version_file = Path(settings.storage_dir) / ".index_version"

        Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(settings.sqlite_path, check_same_thread=False, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

        cs = ChromaSettings(anonymized_telemetry=False)
        if settings.chroma_mode == "http":
            self.chroma = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port, settings=cs)
        else:
            self.chroma = chromadb.PersistentClient(path=settings.chroma_path, settings=cs)

        self._check_embedder()
        self.collection = self._get_collection(embedder.collection_name)

    # ------------------------------------------------------------------ setup helpers
    def _get_collection(self, name: str):
        try:
            return self.chroma.get_or_create_collection(name=name, configuration={"hnsw": {"space": "cosine"}})
        except TypeError:   # older chromadb without `configuration`
            return self.chroma.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def _get_meta(self, key: str) -> Optional[str]:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None

    def _set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, value))

    def _check_embedder(self) -> None:
        """If the embedding model changed, old vectors are meaningless -> rebuild the derived index."""
        stored = self._get_meta("embedder_id")
        if stored and stored != self.embedder.id:
            log.warning("Embedding backend changed (%s -> %s). Rebuilding the index from scratch.",
                        stored, self.embedder.id)
            with self._lock:
                self.db.execute("DELETE FROM files")
                self.db.execute("DELETE FROM chunks")
                self.db.execute("DELETE FROM chunks_fts")
                self.db.commit()
            try:
                for c in self.chroma.list_collections():
                    nm = c if isinstance(c, str) else c.name
                    if nm.startswith("chunks_") and nm != self.embedder.collection_name:
                        self.chroma.delete_collection(nm)
            except Exception as e:  # pragma: no cover
                log.warning("Could not delete old vector collections: %s", e)
        with self._lock:
            self._set_meta("embedder_id", self.embedder.id)
            self.db.commit()

    def _bump_version(self) -> None:
        try:
            self.version_file.write_text(str(time.time()))
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            try:
                self.db.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ write path
    @staticmethod
    def _prefix(meta, location: str) -> str:
        loc = f" | {location}" if location else ""
        return f"[File: {meta.file_name} | Modified: {meta.modified_date}{loc}] "

    def index_file(self, path: str, force: bool = False) -> IndexResult:
        path = str(Path(path))
        try:
            if not os.path.isfile(path):
                return IndexResult(path, "skipped", message="not a file")
            if not should_index(path, self.settings, self.settings.watch_dirs):
                # extension/size/ignore rules; make sure nothing stale remains
                if self._file_row(path):
                    self.remove_file(path)
                return IndexResult(path, "skipped", message="filtered (type/size/ignored)")
            meta = file_meta(path)
            row = self._file_row(path)

            # 1) cheap check: same size and mtime => unchanged
            if row and not force and row["size"] == meta.size_bytes and abs(row["modified_ts"] - meta.modified_ts) < 1e-3:
                return IndexResult(path, "unchanged", n_chunks=row["n_chunks"])

            # 2) content hash: touched/copied-but-identical files are not re-embedded
            meta.sha256 = file_sha256(path)
            if row and not force and row["sha256"] == meta.sha256:
                self._refresh_file_dates(path, meta)
                return IndexResult(path, "metadata_updated", n_chunks=row["n_chunks"], message="content identical")

            # 3) parse + chunk + embed (slow, outside the DB lock, before anything destructive happens)
            parsed: ParsedFile = parse_file(path, self.settings)
            chunks = chunk_segments(parsed.segments, self.settings.chunk_tokens, self.settings.chunk_overlap_ratio)
            embed_texts = [self._prefix(meta, c.location) + c.text for c in chunks]
            vectors = self.embedder.embed_documents(embed_texts) if embed_texts else []

            # a file with no extractable text is still findable by name
            if not chunks:
                name_text = f"{meta.file_name} {Path(meta.file_name).stem.replace('_', ' ').replace('-', ' ')}"
                from common import Chunk
                chunks = [Chunk(index=0, text=name_text, location="file name", tokens=len(name_text) // 4 + 1)]
                embed_texts = [self._prefix(meta, "file name") + name_text]
                vectors = self.embedder.embed_documents(embed_texts)

            self._write_file(meta, parsed, chunks, embed_texts, vectors)
            msg = "; ".join(parsed.warnings[:3])
            return IndexResult(path, "indexed", n_chunks=len(chunks), ocr_pages=parsed.ocr_pages, message=msg)
        except Exception as e:
            log.error("Indexing failed for %s: %s: %s", path, type(e).__name__, e)
            return IndexResult(path, "failed", message=f"{type(e).__name__}: {e}")

    def _file_row(self, path: str):
        with self._lock:
            return self.db.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()

    def _refresh_file_dates(self, path: str, meta) -> None:
        with self._lock:
            self.db.execute("UPDATE files SET size=?, created_ts=?, modified_ts=? WHERE path=?",
                            (meta.size_bytes, meta.created_ts, meta.modified_ts, path))
            self.db.execute("UPDATE chunks SET modified_ts=? WHERE file_path=?", (meta.modified_ts, path))
            ids = [str(r["id"]) for r in self.db.execute("SELECT id FROM chunks WHERE file_path=?", (path,))]
            self.db.commit()
        if ids:
            try:
                for i in range(0, len(ids), 500):
                    self.collection.update(ids=ids[i:i + 500],
                                           metadatas=[{"modified_ts": float(meta.modified_ts)}] * len(ids[i:i + 500]))
            except Exception as e:
                log.warning("Could not refresh vector metadata for %s: %s", path, e)
        self._bump_version()

    def _write_file(self, meta, parsed: ParsedFile, chunks, embed_texts, vectors) -> None:
        path = meta.file_path
        with self._lock:
            old_ids = [str(r["id"]) for r in self.db.execute("SELECT id FROM chunks WHERE file_path=?", (path,))]
            try:
                self.db.execute("BEGIN")
                self.db.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_path=?)", (path,))
                self.db.execute("DELETE FROM chunks WHERE file_path=?", (path,))
                new_ids: list[str] = []
                for c, et in zip(chunks, embed_texts):
                    cur = self.db.execute(
                        "INSERT INTO chunks(file_path, chunk_idx, location, text, tokens, modified_ts, ext) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (path, c.index, c.location, c.text, c.tokens, meta.modified_ts, meta.file_extension))
                    cid = cur.lastrowid
                    self.db.execute("INSERT INTO chunks_fts(rowid, body) VALUES (?,?)", (cid, et))
                    new_ids.append(str(cid))
                self.db.execute(
                    "INSERT INTO files(path,name,ext,size,created_ts,modified_ts,sha256,n_chunks,indexed_at,ocr_pages,warnings) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET name=excluded.name, ext=excluded.ext, "
                    "size=excluded.size, created_ts=excluded.created_ts, modified_ts=excluded.modified_ts, "
                    "sha256=excluded.sha256, n_chunks=excluded.n_chunks, indexed_at=excluded.indexed_at, "
                    "ocr_pages=excluded.ocr_pages, warnings=excluded.warnings",
                    (path, meta.file_name, meta.file_extension, meta.size_bytes, meta.created_ts, meta.modified_ts,
                     meta.sha256, len(chunks), time.time(), parsed.ocr_pages, "; ".join(parsed.warnings)[:2000]))

                # vectors: upsert new first, then drop old (new ids are never reused thanks to AUTOINCREMENT)
                metas = [{"file_path": path, "ext": meta.file_extension, "modified_ts": float(meta.modified_ts)}
                         for _ in new_ids]
                for i in range(0, len(new_ids), 256):
                    self.collection.upsert(ids=new_ids[i:i + 256], embeddings=vectors[i:i + 256],
                                           metadatas=metas[i:i + 256])
                for i in range(0, len(old_ids), 500):
                    self.collection.delete(ids=old_ids[i:i + 500])
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
        self._bump_version()

    def remove_file(self, path: str) -> bool:
        path = str(Path(path))
        with self._lock:
            ids = [str(r["id"]) for r in self.db.execute("SELECT id FROM chunks WHERE file_path=?", (path,))]
            existed = self.db.execute("SELECT 1 FROM files WHERE path=?", (path,)).fetchone() is not None
            self.db.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_path=?)", (path,))
            self.db.execute("DELETE FROM chunks WHERE file_path=?", (path,))
            self.db.execute("DELETE FROM files WHERE path=?", (path,))
            self.db.commit()
        try:
            for i in range(0, len(ids), 500):
                self.collection.delete(ids=ids[i:i + 500])
        except Exception as e:
            log.warning("Vector delete failed for %s: %s", path, e)
        if existed or ids:
            self._bump_version()
        return existed

    def remove_folder(self, folder: str) -> int:
        """Remove everything under a directory (used when a whole folder is deleted/moved)."""
        prefix = str(Path(folder)).rstrip(os.sep) + os.sep
        with self._lock:
            paths = [r["path"] for r in self.db.execute("SELECT path FROM files WHERE path LIKE ? ESCAPE '\\'",
                                                         (prefix.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_") + "%",))]
        for p in paths:
            self.remove_file(p)
        return len(paths)

    def move_file(self, old: str, new: str) -> IndexResult:
        self.remove_file(old)
        return self.index_file(new)

    def remove_missing(self, roots: list[str]) -> int:
        """Drop index entries whose file no longer exists (or no longer passes the ignore rules)."""
        roots_ok = [str(Path(r).resolve()) for r in roots if os.path.isdir(r)]
        roots_all = [str(Path(r).resolve()) for r in roots]
        removed = 0
        with self._lock:
            paths = [r["path"] for r in self.db.execute("SELECT path FROM files")]
        for p in paths:
            rp = str(Path(p).resolve()) if os.path.exists(p) else p
            under_missing_root = any(
                (p == r or p.startswith(r + os.sep)) for r in roots_all if r not in roots_ok)
            if not os.path.exists(p):
                if under_missing_root:      # drive probably just unmounted: keep entries
                    continue
                self.remove_file(p)
                removed += 1
            elif not any(rp == r or rp.startswith(r + os.sep) for r in roots_all):
                self.remove_file(p)          # root no longer watched
                removed += 1
            elif not should_index(p, self.settings, roots):
                self.remove_file(p)
                removed += 1
        return removed

    def scan(self, roots: list[str], progress: Optional[Callable[[int, int, str, str], None]] = None,
             force: bool = False) -> ScanStats:
        t0 = time.time()
        stats = ScanStats()
        files = list(iter_files(roots, self.settings))
        total = len(files)
        for i, p in enumerate(files, start=1):
            res = self.index_file(p, force=force)
            stats.seen += 1
            if res.status == "indexed":
                stats.indexed += 1
            elif res.status in ("unchanged", "metadata_updated"):
                stats.unchanged += 1
            elif res.status == "failed":
                stats.failed += 1
                stats.errors.append(f"{p}: {res.message}")
            else:
                stats.skipped += 1
            if progress:
                progress(i, total, p, res.status)
        stats.removed = self.remove_missing(roots)
        stats.seconds = time.time() - t0
        return stats

    # ------------------------------------------------------------------ read path
    def stats(self) -> dict:
        with self._lock:
            nf = self.db.execute("SELECT COUNT(*) c FROM files").fetchone()["c"]
            nc = self.db.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
            by_ext = {r["ext"]: r["c"] for r in self.db.execute(
                "SELECT ext, COUNT(*) c FROM files GROUP BY ext ORDER BY c DESC")}
            last = self.db.execute("SELECT MAX(indexed_at) m FROM files").fetchone()["m"]
        return {"files": nf, "chunks": nc, "by_extension": by_ext, "last_indexed_at": last,
                "embedder": self.embedder.id}

    def list_files(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM files ORDER BY path")]

    def get_chunks(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        out: dict[int, sqlite3.Row] = {}
        with self._lock:
            for i in range(0, len(ids), 500):
                part = ids[i:i + 500]
                q = f"SELECT * FROM chunks WHERE id IN ({','.join('?' * len(part))})"
                for r in self.db.execute(q, part):
                    out[r["id"]] = r
        return out

    def get_file(self, path: str) -> Optional[sqlite3.Row]:
        return self._file_row(path)

    def _sql_filters(self, f: Filters) -> tuple[str, list]:
        clauses, params = [], []
        if f.modified_after:
            clauses.append("c.modified_ts >= ?")
            params.append(f.modified_after)
        if f.modified_before:
            clauses.append("c.modified_ts <= ?")
            params.append(f.modified_before)
        if f.file_types:
            clauses.append(f"c.ext IN ({','.join('?' * len(f.file_types))})")
            params.extend(f.file_types)
        if f.path_contains:
            esc = f.path_contains.lower().replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
            clauses.append("lower(c.file_path) LIKE ? ESCAPE '\\'")
            params.append(f"%{esc}%")
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    def sparse_search(self, fts_query: str, n: int, filters: Optional[Filters] = None) -> list[tuple[int, float]]:
        """BM25 over chunk text (FTS5). Returns [(chunk_id, score)] best first, score > 0."""
        if not fts_query:
            return []
        where, params = self._sql_filters(filters or Filters())
        sql = ("SELECT c.id AS id, bm25(chunks_fts) AS s FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
               f"WHERE chunks_fts MATCH ?{where} ORDER BY s LIMIT ?")
        with self._lock:
            try:
                rows = self.db.execute(sql, [fts_query, *params, n]).fetchall()
            except sqlite3.OperationalError as e:
                log.warning("FTS query failed (%s): %r", e, fts_query[:120])
                return []
        return [(r["id"], -float(r["s"])) for r in rows]

    def dense_search(self, query_vec: list[float], n: int, filters: Optional[Filters] = None) -> list[tuple[int, float]]:
        """Vector similarity search. Returns [(chunk_id, cosine_similarity)] best first."""
        count = self.collection.count()
        if count == 0:
            return []
        filters = filters or Filters()
        conds = []
        if filters.modified_after:
            conds.append({"modified_ts": {"$gte": float(filters.modified_after)}})
        if filters.modified_before:
            conds.append({"modified_ts": {"$lte": float(filters.modified_before)}})
        if filters.file_types:
            conds.append({"ext": {"$in": list(filters.file_types)}})
        where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
        want = min(count, n * (3 if filters.path_contains else 1))
        res = self.collection.query(query_embeddings=[query_vec], n_results=want, where=where,
                                    include=["distances"])
        ids = [int(i) for i in res["ids"][0]]
        sims = [1.0 - float(d) for d in res["distances"][0]]
        out = list(zip(ids, sims))
        if filters.path_contains and out:
            rows = self.get_chunks([i for i, _ in out])
            needle = filters.path_contains.lower()
            out = [(i, s) for i, s in out if i in rows and needle in rows[i]["file_path"].lower()]
        return out[:n]

    def consistency_check(self, repair: bool = False) -> dict:
        """Compare SQLite chunk ids with Chroma ids; optionally delete orphan vectors."""
        with self._lock:
            sql_ids = {str(r["id"]) for r in self.db.execute("SELECT id FROM chunks")}
        chroma_ids = set(self.collection.get(include=[])["ids"])
        orphans = sorted(chroma_ids - sql_ids)
        missing = sorted(sql_ids - chroma_ids)
        if repair and orphans:
            for i in range(0, len(orphans), 500):
                self.collection.delete(ids=orphans[i:i + 500])
        return {"orphan_vectors": len(orphans), "chunks_without_vector": len(missing), "repaired": repair}
