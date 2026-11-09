"""
Document parsing: file -> ParsedFile(meta, segments).

Every parser returns `Segment`s that carry a location (page / slide / lines / sheet rows / section) so the final
answer can say "Page 3" or "Sheet 'Q3' rows 41-80" instead of just a file name.

Supported: PDF (PyMuPDF, with per-page OCR fallback for scanned pages), DOCX (paragraphs + tables in document order,
split by headings), PPTX (slides, tables, notes), XLSX/CSV/TSV (row blocks, header repeated), plain text / markdown /
source code / HTML / JSON / YAML (line blocks), images (OCR).

Everything is defensive: a broken page/slide/sheet becomes a warning, never an exception for the whole file.
"""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import io
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable, Optional

from common import FileMeta, ParsedFile, Segment, get_logger

log = get_logger("parser")

try:  # PyMuPDF >= 1.24 exposes `pymupdf`; older versions expose `fitz`
    import pymupdf as _fitz
except Exception:  # pragma: no cover
    try:
        import fitz as _fitz  # type: ignore
    except Exception:
        _fitz = None


# =========================================================================== file discovery helpers
def file_meta(path: str) -> FileMeta:
    st = os.stat(path)
    created = getattr(st, "st_birthtime", None) or min(st.st_ctime, st.st_mtime)
    p = Path(path)
    return FileMeta(
        file_path=str(p), file_name=p.name, file_extension=p.suffix.lower(),
        created_ts=float(created), modified_ts=float(st.st_mtime), size_bytes=int(st.st_size),
    )


def file_sha256(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def should_index(path: str, settings, roots: Optional[Iterable[str]] = None) -> bool:
    """Extension supported, not ignored/hidden/temporary, size within limits."""
    p = Path(path)
    if p.suffix.lower() not in settings.supported_extensions:
        return False
    name = p.name
    if any(fnmatch.fnmatch(name, pat) for pat in settings.ignore_file_patterns):
        return False
    # never index our own storage directory
    try:
        if Path(settings.storage_dir).resolve() in p.resolve().parents:
            return False
    except Exception:
        pass
    # Directory-level rules (ignored/hidden folders) apply to the path *relative to a watched root*.
    # A file outside every root is judged by its own name only, so system path parts never block it.
    rel_parts = [p.name]
    if roots:
        for r in roots:
            try:
                rel_parts = list(p.resolve().relative_to(Path(r).resolve()).parts)
                break
            except Exception:
                continue
    for part in rel_parts[:-1]:
        if part in settings.ignore_dir_names:
            return False
        if not settings.index_hidden and part.startswith("."):
            return False
    if not settings.index_hidden and name.startswith("."):
        return False
    try:
        if p.stat().st_size > settings.max_file_size_bytes or p.stat().st_size == 0:
            return False
    except OSError:
        return False
    return True


def iter_files(roots: Iterable[str], settings) -> Iterable[str]:
    roots = [r for r in roots if r]
    for root in roots:
        if not os.path.isdir(root):
            log.warning("Watch directory does not exist, skipping: %s", root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in settings.ignore_dir_names and (settings.index_hidden or not d.startswith("."))
            ]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                if should_index(full, settings, roots):
                    yield full


# =========================================================================== OCR
class OcrEngine:
    """Thin wrapper so the rest of the code does not care which OCR library is used."""

    name = "none"

    def available(self) -> bool:
        return False

    def image_to_text(self, pil_image) -> str:  # pragma: no cover - interface
        return ""


class TesseractOcr(OcrEngine):
    name = "tesseract"

    def __init__(self, lang: str = "eng"):
        self.lang = lang
        self._ok: Optional[bool] = None

    def available(self) -> bool:
        if self._ok is None:
            try:
                import pytesseract  # noqa: F401
                self._ok = shutil.which("tesseract") is not None or bool(
                    getattr(__import__("pytesseract").pytesseract, "tesseract_cmd", None)
                    and os.path.exists(__import__("pytesseract").pytesseract.tesseract_cmd)
                )
            except Exception:
                self._ok = False
        return bool(self._ok)

    def image_to_text(self, pil_image) -> str:
        import pytesseract
        from PIL import ImageOps

        img = pil_image.convert("L")
        if img.width < 1000:  # tesseract does better on larger glyphs
            scale = 2
            img = img.resize((img.width * scale, img.height * scale))
        img = ImageOps.autocontrast(img)
        return pytesseract.image_to_string(img, lang=self.lang) or ""


class EasyOcr(OcrEngine):
    name = "easyocr"
    _LANG_MAP = {"eng": "en", "deu": "de", "fra": "fr", "spa": "es", "ita": "it", "por": "pt", "nld": "nl",
                 "rus": "ru", "chi_sim": "ch_sim", "chi_tra": "ch_tra", "jpn": "ja", "kor": "ko", "hin": "hi"}

    def __init__(self, lang: str = "eng"):
        self.langs = [self._LANG_MAP.get(l, l) for l in lang.split("+")]
        self._reader = None
        self._ok: Optional[bool] = None

    def available(self) -> bool:
        if self._ok is None:
            try:
                import easyocr  # noqa: F401
                self._ok = True
            except Exception:
                self._ok = False
        return bool(self._ok)

    def image_to_text(self, pil_image) -> str:
        import easyocr
        import numpy as np

        if self._reader is None:
            self._reader = easyocr.Reader(self.langs, gpu=False, verbose=False)
        out = self._reader.readtext(np.array(pil_image.convert("RGB")), detail=0, paragraph=True)
        return "\n".join(out)


_ocr_cache: dict[tuple, OcrEngine] = {}


def get_ocr(settings) -> OcrEngine:
    key = (settings.ocr_backend, settings.ocr_languages)
    if key in _ocr_cache:
        return _ocr_cache[key]
    backend = settings.ocr_backend.lower()
    engine: OcrEngine = OcrEngine()
    if backend in ("auto", "tesseract"):
        t = TesseractOcr(settings.ocr_languages)
        if t.available():
            engine = t
        elif backend == "tesseract":
            log.warning("OCR backend 'tesseract' requested but the tesseract binary / pytesseract was not found.")
    if engine.name == "none" and backend in ("auto", "easyocr"):
        e = EasyOcr(settings.ocr_languages)
        if e.available():
            engine = e
        elif backend == "easyocr":
            log.warning("OCR backend 'easyocr' requested but easyocr is not installed.")
    if engine.name == "none" and backend != "none":
        log.warning("No OCR engine available: scanned PDFs and images will be indexed without their text. "
                    "Install tesseract (apt/brew/winget) or `pip install easyocr`.")
    _ocr_cache[key] = engine
    return engine


# =========================================================================== individual parsers
def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t\u00A0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: str, settings, ocr: OcrEngine) -> tuple[list[Segment], list[str], int]:
    if _fitz is None:
        return [], ["PyMuPDF is not installed (pip install pymupdf)"], 0
    segs: list[Segment] = []
    warns: list[str] = []
    ocr_pages = 0
    doc = _fitz.open(path)
    try:
        if doc.needs_pass:
            return [], ["PDF is password protected; skipped"], 0
        title = (doc.metadata or {}).get("title", "") if doc.metadata else ""
        if title and title.strip():
            segs.append(Segment(text=f"Title: {title.strip()}", unit="Page", start=1, end=1))
        for i, page in enumerate(doc, start=1):
            try:
                text = page.get_text("text", sort=True) or ""
                if len(text.strip()) < settings.ocr_min_chars_per_pdf_page and page.get_images():
                    if ocr.available() and ocr_pages < settings.ocr_max_pages_per_file:
                        from PIL import Image
                        pix = page.get_pixmap(dpi=settings.ocr_dpi)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        o = ocr.image_to_text(img)
                        ocr_pages += 1
                        if len(o.strip()) > len(text.strip()):
                            text = o
                    elif not ocr.available():
                        warns.append(f"page {i} looks scanned but no OCR engine is available")
                text = _clean(text)
                if text:
                    segs.append(Segment(text=text, unit="Page", start=i, end=i))
            except Exception as e:  # one bad page must not kill the file
                warns.append(f"page {i}: {type(e).__name__}: {e}")
    finally:
        doc.close()
    return segs, warns, ocr_pages


def parse_docx(path: str, settings, ocr: OcrEngine) -> tuple[list[Segment], list[str], int]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = Document(path)
    segs: list[Segment] = []
    warns: list[str] = []

    state = {"label": "", "lines": [], "n": 0}

    def flush():
        text = _clean("\n".join(state["lines"]))
        if text:
            state["n"] += 1
            segs.append(Segment(text=text, unit="Section", start=state["n"], end=state["n"], label=state["label"]))
        state["lines"] = []

    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        try:
            if tag == "p":
                p = Paragraph(child, d)
                t = p.text.strip()
                if not t:
                    continue
                style = (p.style.name if p.style is not None else "") or ""
                if style.startswith("Heading") or style == "Title":
                    flush()
                    state["label"] = t[:80]
                state["lines"].append(t)
            elif tag == "tbl":
                tbl = Table(child, d)
                for row in tbl.rows:
                    seen, cells = set(), []
                    for c in row.cells:  # merged cells repeat; de-duplicate by identity
                        if id(c._tc) in seen:
                            continue
                        seen.add(id(c._tc))
                        cells.append(c.text.strip().replace("\n", " "))
                    if any(cells):
                        state["lines"].append(" | ".join(cells))
        except Exception as e:
            warns.append(f"{tag}: {type(e).__name__}: {e}")
    flush()
    # core properties title
    try:
        title = (d.core_properties.title or "").strip()
        if title:
            segs.insert(0, Segment(text=f"Title: {title}", unit="Section", start=0, end=0, label="Document properties"))
    except Exception:
        pass
    return segs, warns, 0


def _iter_shapes(shapes):
    for sh in shapes:
        yield sh
        if getattr(sh, "shape_type", None) == 6 and hasattr(sh, "shapes"):  # group
            yield from _iter_shapes(sh.shapes)


def parse_pptx(path: str, settings, ocr: OcrEngine) -> tuple[list[Segment], list[str], int]:
    from pptx import Presentation

    prs = Presentation(path)
    segs: list[Segment] = []
    warns: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        try:
            lines: list[str] = []
            for sh in _iter_shapes(slide.shapes):
                if getattr(sh, "has_text_frame", False) and sh.has_text_frame:
                    t = sh.text_frame.text.strip()
                    if t:
                        lines.append(t)
                if getattr(sh, "has_table", False) and sh.has_table:
                    for row in sh.table.rows:
                        cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                        if any(cells):
                            lines.append(" | ".join(cells))
            try:
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                    n = slide.notes_slide.notes_text_frame.text.strip()
                    if n:
                        lines.append("Speaker notes: " + n)
            except Exception:
                pass
            text = _clean("\n".join(lines))
            if text:
                segs.append(Segment(text=text, unit="Slide", start=i, end=i))
        except Exception as e:
            warns.append(f"slide {i}: {type(e).__name__}: {e}")
    return segs, warns, 0


def _fmt_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v).strip().replace("\n", " ")


def _rows_to_segments(rows: Iterable[list[str]], label: str, block: int = 40,
                      max_rows: int = 50_000) -> list[Segment]:
    segs: list[Segment] = []
    header: Optional[list[str]] = None
    buf: list[str] = []
    buf_start = 1
    n = 0
    for n, row in enumerate(rows, start=1):
        if n > max_rows:
            break
        if not any(c for c in row):
            continue
        if header is None:
            header = row
            buf_start = n
            buf.append(" | ".join(c for c in row))
            continue
        if len(header) <= 60 and any(header):
            line = " | ".join(f"{h}: {c}" if h and c else (c or "") for h, c in zip(header, row + [""] * (len(header) - len(row))))
        else:
            line = " | ".join(row)
        buf.append(line)
        if len(buf) >= block:
            segs.append(Segment(text="\n".join(buf), unit="Rows", start=buf_start, end=n, label=label))
            buf, buf_start = [], n + 1
    if buf:
        segs.append(Segment(text="\n".join(buf), unit="Rows", start=buf_start, end=min(n, max_rows), label=label))
    return segs


def parse_xlsx(path: str, settings, ocr: OcrEngine) -> tuple[list[Segment], list[str], int]:
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    segs: list[Segment] = []
    warns: list[str] = []
    try:
        for ws in wb.worksheets:
            try:
                rows = ([_fmt_cell(c) for c in r] for r in ws.iter_rows(values_only=True))
                s = _rows_to_segments(rows, f"Sheet '{ws.title}'")
                if s:
                    # make the sheet name itself searchable
                    s[0].text = f"Sheet: {ws.title}\n" + s[0].text
                segs.extend(s)
            except Exception as e:
                warns.append(f"sheet {ws.title}: {type(e).__name__}: {e}")
    finally:
        wb.close()
    return segs, warns, 0


def _read_text(path: str, limit: int = 8_000_000) -> str:
    raw = open(path, "rb").read(limit)
    for enc in ("utf-8-sig", "utf-16"):
        try:
            if enc == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def parse_csv(path: str, settings, ocr: OcrEngine) -> tuple[list[Segment], list[str], int]:
    text = _read_text(path)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except Exception:
        dialect = csv.excel_tab if path.lower().endswith(".tsv") else csv.excel
    rows = ([c.strip() for c in r] for r in csv.reader(io.StringIO(text), dialect))
    return _rows_to_segments(rows, "CSV"), [], 0


def _line_segments(text: str, block: int = 40, unit: str = "Lines") -> list[Segment]:
    lines = text.splitlines()
    segs = []
    for i in range(0, len(lines), block):
        chunk = "\n".join(lines[i:i + block]).strip()
        if chunk:
            segs.append(Segment(text=_clean(chunk) if unit != "Lines" else chunk, unit=unit,
                                start=i + 1, end=min(i + block, len(lines))))
    return segs


def parse_text(path: str, settings, ocr: OcrEngine) -> tuple[list[Segment], list[str], int]:
    text = _read_text(path)
    ext = Path(path).suffix.lower()
    if ext in (".html", ".htm", ".xml"):
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
    return _line_segments(text), [], 0


def parse_image(path: str, settings, ocr: OcrEngine) -> tuple[list[Segment], list[str], int]:
    from PIL import Image

    warns: list[str] = []
    img = Image.open(path)
    try:
        img.seek(0)
    except Exception:
        pass
    w, h = img.size
    segs: list[Segment] = []
    n_ocr = 0
    if min(w, h) >= 40 and ocr.available():
        try:
            t = _clean(ocr.image_to_text(img))
            n_ocr = 1
            if t:
                segs.append(Segment(text=t, unit="Image", start=1, end=1, label="text in image (OCR)"))
        except Exception as e:
            warns.append(f"OCR failed: {type(e).__name__}: {e}")
    elif not ocr.available():
        warns.append("no OCR engine available; image indexed by file name only")
    if not segs:
        segs.append(Segment(text=f"Image file {Path(path).stem.replace('_', ' ')} ({w}x{h}px)",
                            unit="Image", start=1, end=1, label="image"))
    return segs, warns, n_ocr


PARSERS: dict[str, Callable] = {
    ".pdf": parse_pdf, ".docx": parse_docx, ".pptx": parse_pptx,
    ".xlsx": parse_xlsx, ".xlsm": parse_xlsx, ".csv": parse_csv, ".tsv": parse_csv,
}
for _e in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"):
    PARSERS[_e] = parse_image


def parse_file(path: str, settings) -> ParsedFile:
    """Parse one file. Never raises for content problems; returns warnings instead."""
    meta = file_meta(path)
    ocr = get_ocr(settings)
    parser = PARSERS.get(meta.file_extension, parse_text)
    try:
        segs, warns, n_ocr = parser(path, settings, ocr)
    except Exception as e:
        log.warning("Failed to parse %s: %s: %s", path, type(e).__name__, e)
        segs, warns, n_ocr = [], [f"parse error: {type(e).__name__}: {e}"], 0
    return ParsedFile(meta=meta, segments=[s for s in segs if s.text.strip()], warnings=warns, ocr_pages=n_ocr)
