# AI File Finder: agentic, local-first file discovery

Describe a file the way you remember it ("the doc from last month where I calculated the stress tensor for the gear
shaft") and get back the **exact file path, the page/section, and a quote proving the match**. Everything runs on
your machine; no file content leaves it.

```
[ PHASE 1: BACKGROUND INDEXING ]
Local files (PDF, DOCX, PPTX, XLSX, CSV, TXT/MD/code, images)
   -> Parser + OCR            PyMuPDF / python-docx / python-pptx / openpyxl, Tesseract for scans & images
   -> Recursive chunker       ~600-token chunks, 17 % overlap, each chunk remembers page/slide/rows/lines
   -> Embeddings + BM25       Ollama (nomic-embed-text / bge-m3)  +  SQLite FTS5 (BM25 keyword index)
   -> Local stores            ChromaDB (vectors)  +  SQLite (chunks, file hashes) - every chunk -> its file_path

[ PHASE 2: LIVE AGENTIC SEARCH ]
"Find the doc with the Q3 pump efficiency calculations"
   -> Query planner (LLM)     semantic query + exact keywords + date / file-type filters (JSON)
   -> Hybrid search           dense vectors + BM25 + exact-phrase BM25, fused with Reciprocal Rank Fusion
   -> Cross-encoder reranker  bge-reranker-v2-m3 scores the top 25 chunks 0..1
   -> Group by file           file score = 0.7 * best chunk + 0.3 * mean(top 3 chunks)
   -> LLM verifier            "does this file really contain it?"  + real-quote check
   -> Self-correction         not confident? rewrite the query / relax filters and retry
   -> Answer                  file path, page/section, evidence quote, open button / file:// link
```

---------------------------------------------------------------------------------------------------------

## 1. Quick start (about 10 minutes)

### 1.1 Prerequisites

| Need | Why | Install |
|---|---|---|
| **Python 3.10+** (tested on 3.12) | runs everything | python.org / your package manager |
| **Tesseract OCR** (system binary) | text from scanned PDFs and images | Ubuntu/Debian: `sudo apt install tesseract-ocr` · macOS: `brew install tesseract` · Windows: installer from the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki), then add its folder to `PATH` |
| **Ollama** (recommended) | local embeddings + LLM agent | [ollama.com/download](https://ollama.com/download) |
| ~10 GB free disk, 8-16 GB RAM | models | (see 1.3 for lighter options) |

Check: `python3 --version`, `tesseract --version`, `ollama --version`.

### 1.2 Install

```bash
cd ai-file-finder
python3 -m venv .venv
source .venv/bin/activate              # Windows PowerShell:  .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-optional.txt     # recommended: enables the bge cross-encoder reranker
```

### 1.3 Pull the local models

Make sure Ollama is running (`ollama serve`, or the desktop app), then:

```bash
ollama pull nomic-embed-text      # embeddings, 274 MB, 768 dims   (alternative: bge-m3, 1.2 GB, multilingual)
ollama pull qwen2.5:7b            # the planner / verifier agent, ~4.7 GB (alternative: llama3.1:8b)
```
Low on RAM? Use `qwen2.5:3b` and set `FILEFINDER_LLM_MODEL=qwen2.5:3b`. The reranker
(`BAAI/bge-reranker-v2-m3`, ~2.2 GB) downloads automatically from Hugging Face on first search.

### 1.4 Tell it which folders to index

```bash
# macOS / Linux
export FILEFINDER_WATCH_DIRS="$HOME/Documents:$HOME/Desktop:$HOME/Downloads"
# Windows PowerShell (separator is ;)
$env:FILEFINDER_WATCH_DIRS = "C:\Users\you\Documents;C:\Users\you\Desktop"
```
(or edit `watch_dirs` in `config.py`. By default it uses your Documents / Desktop / Downloads folders if they exist.)

### 1.5 Run it

```bash
streamlit run app.py                  # opens http://localhost:8501
```
The first launch scans your folders in the background (you can search while it runs). After that only new or
changed files are processed, and the built-in watcher keeps the index live.

### 1.6 Try it without any setup

```bash
python scripts/demo.py                # creates sample files, indexes them with offline fallbacks, runs 6 searches
pip install reportlab                 # (only if the demo complains: it is used to generate a sample PDF)
```

---------------------------------------------------------------------------------------------------------

## 2. Everyday use

### Web UI (`streamlit run app.py`)
Type a description in the chat box. Each result shows confidence, why it matched, the **evidence quote with its
page/section**, and **Open / Folder / file link** buttons. The sidebar shows index size, active models, recent
indexing activity, a re-scan button, and an option to display *how the search was done* (planner output, filters,
retries).

### Command line
```bash
python cli.py index                    # scan watched folders now (--force re-embeds everything, --dirs A B overrides folders)
python cli.py watch                    # headless: initial scan + live file watcher (use this OR the UI, see note below)
python cli.py search "Q3 pump efficiency calculations" --trace
python cli.py search "quote with part XJ-4471-B" --json
python cli.py chat                     # interactive prompt
python cli.py open "my lease agreement"        # search and open the best hit
python cli.py status                   # counts, active embedder / reranker / LLM
python cli.py check --repair           # verify SQLite <-> vector DB consistency, remove orphan vectors
```

### Python API
```python
from agent.pipeline import FileFinder
finder = FileFinder()
finder.index_all()
resp = finder.search("the doc from last month with the stress tensor of the gear shaft")
for m in resp.matches:
    print(m.file_path, m.evidence_location, m.confidence)
    print("  ", m.evidence)
```

### Example queries that work well
* `Find the doc with the Q3 pump efficiency calculations`
* `the PDF quote that lists part XJ-4471-B`      (exact codes are handled by BM25 phrase matching)
* `spreadsheet from October 2026 about sales by region`    (month + file type become filters)
* `slides where I mention "runway" since 2026-09-01`
* `photo of a receipt from the hardware store`   (images are OCR'd)

### Keeping it running in the background
Run **one** process that writes to the index. The Streamlit app already embeds the watcher; if you prefer a
headless service, run `python cli.py watch` and use the UI/CLI in **HTTP mode** (below) so they can share the vector
DB safely.

*Linux (systemd user service)*: `~/.config/systemd/user/filefinder.service`
```ini
[Service]
WorkingDirectory=/path/to/ai-file-finder
Environment=FILEFINDER_WATCH_DIRS=/home/you/Documents:/home/you/Desktop
ExecStart=/path/to/ai-file-finder/.venv/bin/python cli.py watch
Restart=on-failure
[Install]
WantedBy=default.target
```
`systemctl --user enable --now filefinder`. On Windows use Task Scheduler ("at log on" -> `python cli.py watch`);
on macOS use a launchd agent.

**Multi-process / shared vector DB:** ChromaDB's embedded mode is not safe for several processes at once. Run
`chroma run --path ./storage/vector_db --port 8000` and set `FILEFINDER_CHROMA_MODE=http`
(`FILEFINDER_CHROMA_HOST`, `FILEFINDER_CHROMA_PORT`). SQLite (WAL mode) is already multi-process safe. Tested here.

---------------------------------------------------------------------------------------------------------

## 3. Configuration reference

Set via environment variable `FILEFINDER_<NAME>` (upper-case) or edit `config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `WATCH_DIRS` | Documents/Desktop/Downloads | folders to index (`:` separated, `;` on Windows) |
| `SUPPORTED_EXTENSIONS` | pdf docx pptx xlsx csv txt md code images … | comma-separated list |
| `MAX_FILE_SIZE_MB` | 100 | larger files are skipped |
| `STORAGE_DIR` | `./storage` | SQLite file + vector DB location |
| `CHUNK_TOKENS` / `CHUNK_OVERLAP_RATIO` | 600 / 0.17 | chunk size (500-800) and overlap (15-20 %) |
| `EMBED_BACKEND` | `auto` | `auto` / `ollama` / `sentence_transformers` / `hashing` |
| `EMBED_MODEL` | `nomic-embed-text` | e.g. `bge-m3` (Ollama) or `BAAI/bge-m3` (sentence_transformers) |
| `RERANKER_BACKEND` | `auto` | `auto` / `cross_encoder` / `lexical` / `none` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | lower `RERANKER_MAX_LENGTH` (e.g. 512) to speed up on CPU |
| `LLM_BACKEND` / `LLM_MODEL` | `auto` / `qwen2.5:7b` | planner + verifier (`heuristic` = no LLM) |
| `OCR_BACKEND` / `OCR_LANGUAGES` | `auto` / `eng` | `tesseract`, `easyocr`, `none`; e.g. `eng+deu` |
| `RERANK_POOL` / `FINAL_FILES` | 25 / 5 | chunks reranked / files returned |
| `MAX_RETRIES` | 2 | self-correction rounds after the first attempt |
| `CONFIDENCE_THRESHOLD` | 0.45 | below this a file is reported as "closest candidate", not a match |
| `WATCH_DEBOUNCE_S` / `WATCH_POLLING` | 2.0 / false | debounce for bursts of events; polling for network drives |
| `OLLAMA_URL` | `http://localhost:11434` | |

**`auto` means "use the best thing available".** If Ollama / sentence-transformers are missing the system falls
back (with a clear warning in the log and the UI sidebar) to: hashed-vector embeddings + BM25, a lexical reranker
and a rule-based planner/verifier. It keeps working, but see "Quality" in section 6.

If you change the embedding model or dimension, the index detects it and rebuilds itself automatically.

---------------------------------------------------------------------------------------------------------

## 4. How each stage maps to the blueprint

| Blueprint stage | Where | Notes |
|---|---|---|
| 1. File watcher & ingestion | `indexer/file_watcher.py`, `indexer/parser.py` | watchdog; debounced; waits for files to finish copying; OCR for scanned PDF pages & images; metadata (`file_path, file_name, extension, created, modified`) stored per file |
| 2. Chunking + metadata | `indexer/chunker.py`, `embed_store.py` | recursive paragraph -> line -> sentence -> word splitting, overlap built from whole sentences; each chunk is embedded as `[File: report_v2.pdf \| Modified: 2026-10-12 \| Page 3] <text>` and stores its parent `file_path` |
| 3. Hybrid index | `indexer/embed_store.py` | dense = ChromaDB; sparse = **SQLite FTS5 `bm25()`** (persistent and incremental, no in-memory rebuild); fusion = RRF |
| 4. Query translator | `agent/query_planner.py` | LLM emits JSON (`semantic_query`, `exact_keywords`, date/type/path filters) with today's date injected; output is validated in Python; rule-based planner is the fallback |
| 5. Retrieval + reranking | `agent/retriever.py`, `agent/reranker.py` | 3 candidate lists (dense, BM25, exact-phrase BM25) -> RRF -> top 25 -> cross-encoder -> group by `file_path` |
| 6. Verify + self-correct | `agent/verifier_agent.py`, `agent/pipeline.py` | LLM verdict per candidate; quotes are **checked against the real chunk text** (a hallucinated quote is replaced by genuine text); retry with LLM rewrite or relaxation ladder (drop date filter -> drop type filters -> broaden) |

Where this deliberately differs from the blueprint: **vector DB** is ChromaDB only (Qdrant is not implemented);
**BM25** uses SQLite FTS5 instead of a separate `rank_bm25` index; the UI is Streamlit only (no Chainlit);
**parsing** uses PyMuPDF/python-docx/etc. (no Unstructured.io/Docling).

### Edge cases handled
* **No duplicate re-indexing**: (size, mtime) check, then SHA-256; unchanged content is never re-embedded, and a
  touched/copied-but-identical file only refreshes its date metadata.
* **Deleted / moved files and folders**: purged from SQLite, FTS and the vector DB; results are also existence-checked
  at query time so a dead link is never shown (stale entries are removed lazily).
* **Unmounted drive**: if a whole watch root is missing, its entries are kept (not wiped).
* **Chunk -> file aggregation**: results are distinct files ranked by best chunk blended with the mean of their top 3.
* **Scanned PDFs**: pages with almost no text layer *and* images are rendered and OCR'd (page-level).
* **Corrupt / password-protected / empty files**: logged as warnings; a file with no extractable text is still
  findable by its name.
* **Crash safety**: embedding happens before any destructive write; the `files` row is written last.
* **Temp/lock files** (`~$x.docx`, `*.tmp`, `*.crdownload`), hidden folders, `node_modules`, `.git`, venvs: ignored.

---------------------------------------------------------------------------------------------------------

## 5. Tests

```bash
pip install -r requirements-optional.txt
python -m pytest -q          # 55 tests, about 1 minute
```
Covers: chunking (size, overlap, sentence integrity, locations), query planning (dates, types, codes, LLM output
sanitising and fallback), the store (dedup, touch/modify/delete, filters, embedder change, unmounted roots),
parsers incl. OCR and corrupt files, end-to-end retrieval on a generated messy corpus (exact part numbers, scanned
PDF, date filters, unanswerable query), the watcher (create/modify/rename/move/delete/folder delete), the real
Ollama code paths against a **fake Ollama server** (prefixes, JSON planning, quote validation, rewrite retry,
garbage-LLM fallback), the cross-encoder path with a stubbed model, and the Streamlit UI (`AppTest`).

**What was *not* run in the build environment** (it has no access to Ollama or Hugging Face): the real
`nomic-embed-text` / `bge-m3` embeddings, the real `bge-reranker-v2-m3` weights, and a real 7B LLM. Those code paths
are exercised only against fakes/stubs (protocol and parsing correctness), so **retrieval quality with the real
models is unmeasured**; expect it to be better than the offline fallback, but verify on your own files.

---------------------------------------------------------------------------------------------------------

## 6. Quality, speed, and limits

* **Quality with the offline fallback** (hashing vectors + lexical reranker + rule planner): excellent on exact
  terms, part numbers, names, dates and file types; weak on paraphrase ("money we're raising" vs "seed extension").
  With Ollama embeddings + the cross-encoder + the LLM verifier, paraphrase queries are the main improvement.
* **Speed** (measured offline: 300 files / 3,000 chunks): ~20 files/s indexing (text files; OCR and real embeddings
  are slower), 0.2 s to re-scan unchanged files, 20-40 ms per search. With the real stack, expect indexing to be
  bounded by embedding throughput, and 2-10 s per search on CPU (cross-encoder + LLM calls); a GPU helps a lot.
* **Scale**: dense search (HNSW) and FTS5 scale well past 100k chunks; the first scan of a large drive can take hours
  with OCR.
* **Not supported**: legacy `.doc/.ppt/.xls` (convert to docx/pptx/xlsx), `.pages/.numbers`, password-protected files,
  text inside images embedded in DOCX/PPTX, handwriting, audio/video.
* Scores are relative: confidence from the LLM verifier is the model's own judgement, not a calibrated probability.
* No access-control layer: it indexes whatever the current OS user can read. Do not point a shared instance at
  folders other users should not see.

---------------------------------------------------------------------------------------------------------

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar says "Offline hashing embeddings are active" | Ollama not running or model not pulled: `ollama serve`, `ollama pull nomic-embed-text`, restart the app (index rebuilds itself) |
| `tesseract is not installed` / scanned PDFs return nothing | install the Tesseract binary and make sure `tesseract --version` works in the same shell |
| First search is very slow | the reranker is downloading (~2 GB) / loading; later searches are faster. Lower `RERANKER_MAX_LENGTH` or set `RERANKER_BACKEND=lexical` |
| Results ignore a date I mentioned | the planner treats "last month" loosely (everything since the 1st of last month); check "How I searched" in the UI. After a failed attempt the agent drops the date filter automatically |
| New files are not picked up on a network drive | `FILEFINDER_WATCH_POLLING=true` |
| Linux: `inotify watch limit reached` | `echo fs.inotify.max_user_watches=524288 \| sudo tee -a /etc/sysctl.conf && sudo sysctl -p` |
| Chroma errors when running UI and `cli.py watch` together | use HTTP mode (section 2) or run only one of them |
| "FTS query failed" warning in logs | a keyword query the BM25 index could not parse; the dense (vector) results still answer |
| The "file link" in the UI does nothing | browsers often block `file://` links from `http://` pages; use the **Open** / **Folder** buttons (they run on the machine hosting the app, so use the UI locally) |
| Want to start over | stop everything, delete `storage/index.sqlite*` and `storage/vector_db/*` (a derived index; your files are never modified) |

---------------------------------------------------------------------------------------------------------

## 8. Project layout

```
ai-file-finder/
├── config.py                 # watched folders, model names, DB settings (env overridable)
├── common.py                 # shared dataclasses + text helpers
├── ollama_client.py          # minimal Ollama HTTP client (embed + JSON chat)
├── opener.py                 # open file / reveal in folder (Windows/macOS/Linux)
├── indexer/
│   ├── file_watcher.py       # watchdog: created/modified/deleted/moved, debounced
│   ├── parser.py             # text + page/slide/row locations + OCR, per file type
│   ├── chunker.py            # recursive chunking with overlap + location tracking
│   ├── embedders.py          # Ollama / sentence-transformers / offline hashing
│   └── embed_store.py        # SQLite (+FTS5 BM25) and ChromaDB, hashing, dedup, deletes
├── agent/
│   ├── query_planner.py      # prompt -> semantic query + keywords + filters (LLM or rules)
│   ├── retriever.py          # hybrid search + RRF + rerank + group by file
│   ├── reranker.py           # bge cross-encoder / lexical fallback
│   ├── verifier_agent.py     # verification, quote checking, self-correction
│   ├── llm.py                # Ollama LLM wrapper
│   └── pipeline.py           # FileFinder: plan -> retrieve -> verify -> retry
├── storage/                  # index.sqlite + vector_db/ (created at runtime)
├── app.py                    # Streamlit UI
├── cli.py                    # index / watch / search / chat / status / check / open
├── scripts/                  # demo.py, make_sample_corpus.py
└── tests/                    # pytest suite (+ fake Ollama server)
```
