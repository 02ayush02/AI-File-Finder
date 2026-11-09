"""
Central configuration.

Edit the defaults below, or override anything with environment variables
(prefix FILEFINDER_). Examples:

    export FILEFINDER_WATCH_DIRS="$HOME/Documents:$HOME/Desktop:$HOME/Downloads"
    export FILEFINDER_EMBED_BACKEND=ollama
    export FILEFINDER_EMBED_MODEL=bge-m3
    export FILEFINDER_LLM_MODEL=qwen2.5:7b

On Windows separate WATCH_DIRS entries with ';' (os.pathsep).

Backends ("auto" = use the best thing that is actually available, otherwise
fall back to a dependency-free local alternative and log a warning):

    EMBED_BACKEND     auto | ollama | sentence_transformers | hashing
    RERANKER_BACKEND  auto | cross_encoder | lexical | none
    LLM_BACKEND       auto | ollama | heuristic
    OCR_BACKEND       auto | tesseract | easyocr | none
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

_PREFIX = "FILEFINDER_"


def _default_watch_dirs() -> list[str]:
    home = Path.home()
    candidates = [home / "Documents", home / "Desktop", home / "Downloads"]
    return [str(p) for p in candidates if p.exists()]


@dataclass
class Settings:
    # ------------------------------------------------------------------ what to index
    watch_dirs: list[str] = field(default_factory=_default_watch_dirs)
    supported_extensions: tuple[str, ...] = (
        ".pdf", ".docx", ".pptx", ".xlsx", ".xlsm", ".csv", ".tsv",
        ".txt", ".md", ".rst", ".log", ".json", ".yaml", ".yml", ".toml", ".ini",
        ".html", ".htm", ".xml",
        ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".sql", ".sh", ".m", ".tex",
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif",
    )
    ignore_dir_names: tuple[str, ...] = (
        ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv", "env",
        ".idea", ".vscode", "site-packages", "$RECYCLE.BIN", ".Trash", ".cache",
    )
    ignore_file_patterns: tuple[str, ...] = (
        "~$*", "*.tmp", "*.part", "*.crdownload", "*.swp", ".DS_Store", "Thumbs.db", "desktop.ini",
    )
    max_file_size_mb: int = 100
    index_hidden: bool = False

    # ------------------------------------------------------------------ storage
    storage_dir: str = str(PROJECT_ROOT / "storage")
    chroma_mode: str = "persistent"          # persistent | http  (http => run `chroma run --path ...`)
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # ------------------------------------------------------------------ chunking
    chunk_tokens: int = 600                  # target 500-800
    chunk_overlap_ratio: float = 0.17        # 15-20 %

    # ------------------------------------------------------------------ OCR
    ocr_backend: str = "auto"
    ocr_languages: str = "eng"               # tesseract codes, e.g. "eng+deu"
    ocr_min_chars_per_pdf_page: int = 30     # fewer chars => treat page as scanned
    ocr_max_pages_per_file: int = 150
    ocr_dpi: int = 200

    # ------------------------------------------------------------------ embeddings
    embed_backend: str = "auto"
    embed_model: str = "nomic-embed-text"    # or bge-m3 (Ollama) / BAAI/bge-m3 (sentence_transformers)
    embed_batch_size: int = 32
    embed_doc_prefix: str | None = None      # None = pick sensible default for the model
    embed_query_prefix: str | None = None
    hashing_dim: int = 1024                  # dimension of the offline fallback embedder

    # ------------------------------------------------------------------ retrieval
    dense_candidates: int = 40
    sparse_candidates: int = 40
    rrf_k: int = 60
    rerank_pool: int = 25                    # chunks passed to the reranker
    reranker_backend: str = "auto"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_length: int = 1024
    final_files: int = 5
    file_score_top_chunks: int = 3           # file score = blend of best and mean-of-top-N chunk scores

    # ------------------------------------------------------------------ LLM agent
    llm_backend: str = "auto"
    llm_model: str = "qwen2.5:7b"            # or llama3.1:8b
    llm_timeout_s: int = 120
    max_retries: int = 2                     # agentic self-correction loops after the first attempt
    confidence_threshold: float = 0.45

    # ------------------------------------------------------------------ Ollama
    ollama_url: str = "http://localhost:11434"

    # ------------------------------------------------------------------ watcher
    watch_debounce_s: float = 2.0
    watch_polling: bool = False              # True for network drives / filesystems without native events
    initial_scan_on_start: bool = True

    # ------------------------------------------------------------------ derived paths
    @property
    def sqlite_path(self) -> str:
        return str(Path(self.storage_dir) / "index.sqlite")

    @property
    def chroma_path(self) -> str:
        return str(Path(self.storage_dir) / "vector_db")

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


def _coerce(value: str, current):
    if isinstance(current, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, tuple):
        return tuple(v.strip() for v in value.split(",") if v.strip())
    if isinstance(current, list):
        return [v for v in value.split(os.pathsep) if v]
    if current is None:
        return value or None
    return value


def load_settings(**overrides) -> Settings:
    """Defaults < environment variables < explicit keyword overrides."""
    s = Settings()
    for f in fields(s):
        env = os.environ.get(_PREFIX + f.name.upper())
        if env is not None:
            setattr(s, f.name, _coerce(env, getattr(s, f.name)))
    for k, v in overrides.items():
        if not hasattr(s, k):
            raise AttributeError(f"Unknown setting: {k}")
        setattr(s, k, v)
    Path(s.storage_dir).mkdir(parents=True, exist_ok=True)
    Path(s.chroma_path).mkdir(parents=True, exist_ok=True)
    return s


settings = load_settings()
