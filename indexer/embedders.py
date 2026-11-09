"""
Embedding backends.

    ollama                 nomic-embed-text / bge-m3 / mxbai-embed-large ... served by a local Ollama
    sentence_transformers  BAAI/bge-m3 etc. loaded in-process (downloads weights on first use)
    hashing                dependency-free feature-hashing embedder; works fully offline. Lower quality (it is
                           lexical, not semantic) but lets the whole system run with zero model downloads.

`get_embedder(settings)` implements the "auto" policy: Ollama if reachable and the model is pulled, otherwise
the hashing fallback (with a loud warning so you know semantic quality is reduced).
"""
from __future__ import annotations

import hashlib
import math
import re
import zlib
from abc import ABC, abstractmethod
from typing import Optional

from common import STOPWORDS, code_like_tokens, get_logger, light_stem
import ollama_client

log = get_logger("embed")


def default_prefixes(model: str) -> tuple[str, str]:
    """(document_prefix, query_prefix) that each model family was trained with."""
    m = model.lower()
    if "nomic-embed" in m:
        return "search_document: ", "search_query: "
    if "mxbai-embed" in m:
        return "", "Represent this sentence for searching relevant passages: "
    if "e5" in m and "bge" not in m:
        return "passage: ", "query: "
    return "", ""


class Embedder(ABC):
    backend: str = ""
    model: str = ""
    dim: int = 0

    @property
    def id(self) -> str:
        return f"{self.backend}:{self.model}:{self.dim}"

    @property
    def collection_name(self) -> str:
        return "chunks_" + hashlib.sha1(self.id.encode()).hexdigest()[:12]

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    @property
    def is_semantic(self) -> bool:
        return True


# --------------------------------------------------------------------------- hashing fallback
class HashingEmbedder(Embedder):
    """Signed feature hashing of stemmed unigrams, bigrams, char 4-grams and code-like tokens.

    Stateless (no fitting), deterministic across processes (crc32, not Python's salted hash()), L2-normalised.
    """
    backend = "hashing"
    model = "feature-hash-v1"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    @property
    def is_semantic(self) -> bool:
        return False

    def _add(self, vec: list[float], feat: str, w: float) -> None:
        h = zlib.crc32(feat.encode("utf-8"))
        idx = h % self.dim
        sign = 1.0 if (h >> 16) & 1 else -1.0
        vec[idx] += sign * w

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        low = text.lower()
        words = re.findall(r"[a-z0-9\u00c0-\u024f]+", low)
        content = [light_stem(w) for w in words if w not in STOPWORDS and len(w) > 1]
        tf: dict[str, int] = {}
        for w in content:
            tf[w] = tf.get(w, 0) + 1
        for w, c in tf.items():
            self._add(vec, "u:" + w, 1.0 + math.log(c))
        for a, b in zip(content, content[1:]):
            self._add(vec, f"b:{a}_{b}", 0.6)
        for w in set(content):
            if len(w) >= 5:
                padded = f"#{w}#"
                for i in range(len(padded) - 3):
                    self._add(vec, "c:" + padded[i:i + 4], 0.15)
        for tok in set(code_like_tokens(text)):
            self._add(vec, "k:" + tok.lower(), 1.5)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# --------------------------------------------------------------------------- Ollama
class OllamaEmbedder(Embedder):
    backend = "ollama"

    def __init__(self, model: str, url: str, batch_size: int = 32,
                 doc_prefix: Optional[str] = None, query_prefix: Optional[str] = None):
        self.model, self.url, self.batch = model, url, batch_size
        d, q = default_prefixes(model)
        self.doc_prefix = d if doc_prefix is None else doc_prefix
        self.query_prefix = q if query_prefix is None else query_prefix
        self.dim = len(ollama_client.embed(url, model, ["dimension probe"])[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            batch = [self.doc_prefix + t for t in texts[i:i + self.batch]]
            out.extend(ollama_client.embed(self.url, self.model, batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return ollama_client.embed(self.url, self.model, [self.query_prefix + text])[0]


# --------------------------------------------------------------------------- sentence-transformers
class SentenceTransformerEmbedder(Embedder):
    backend = "sentence_transformers"

    def __init__(self, model: str, batch_size: int = 16,
                 doc_prefix: Optional[str] = None, query_prefix: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # heavy import, only when requested
        log.info("Loading sentence-transformers model %s (first use downloads the weights)", model)
        self.model = model
        self._m = SentenceTransformer(model)
        self.batch = batch_size
        d, q = default_prefixes(model)
        self.doc_prefix = d if doc_prefix is None else doc_prefix
        self.query_prefix = q if query_prefix is None else query_prefix
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embs = self._m.encode([self.doc_prefix + t for t in texts], batch_size=self.batch,
                              normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embs]

    def embed_query(self, text: str) -> list[float]:
        return self._m.encode([self.query_prefix + text], normalize_embeddings=True,
                              show_progress_bar=False)[0].tolist()


# --------------------------------------------------------------------------- factory
def get_embedder(settings) -> Embedder:
    backend = settings.embed_backend.lower()
    if backend not in ("auto", "ollama", "sentence_transformers", "hashing"):
        raise ValueError(f"Unknown embed_backend: {settings.embed_backend}")

    if backend == "hashing":
        return HashingEmbedder(settings.hashing_dim)

    if backend in ("auto", "ollama"):
        if ollama_client.has_model(settings.ollama_url, settings.embed_model):
            try:
                emb = OllamaEmbedder(settings.embed_model, settings.ollama_url, settings.embed_batch_size,
                                     settings.embed_doc_prefix, settings.embed_query_prefix)
                log.info("Embeddings: Ollama model '%s' (%d dims)", emb.model, emb.dim)
                return emb
            except Exception as e:
                if backend == "ollama":
                    raise
                log.warning("Ollama embedding failed (%s); falling back.", e)
        elif backend == "ollama":
            raise RuntimeError(
                f"Ollama model '{settings.embed_model}' not available at {settings.ollama_url}. "
                f"Start Ollama and run: ollama pull {settings.embed_model}")

    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embed_model, settings.embed_batch_size,
                                           settings.embed_doc_prefix, settings.embed_query_prefix)

    log.warning(
        "Ollama embedding model '%s' not reachable at %s -> using the offline HASHING embedder. "
        "Search still works (hybrid BM25 + hashed vectors) but is lexical, not semantic. "
        "For semantic search: install Ollama, `ollama pull %s`, restart.",
        settings.embed_model, settings.ollama_url, settings.embed_model)
    return HashingEmbedder(settings.hashing_dim)
