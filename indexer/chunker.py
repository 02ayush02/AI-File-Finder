"""
Recursive character chunking (paragraph -> line -> sentence -> word) with overlap.

* Target size `chunk_tokens` (default 600, i.e. inside the 500-800 window), overlap 15-20 %.
* Overlap is made of whole "atoms" (sentence/line-sized pieces) so sentences are not cut in half.
* Each chunk remembers *where* it came from (page / slide / sheet rows / lines / section) so the UI can point at it.

Token counts are estimated (~4 characters per token) - no tokenizer download needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from common import Chunk, Segment, est_tokens

# Separators from coarse to fine. Each regex matches the separator; pieces keep their trailing separator so
# that "".join(pieces) reproduces the original text exactly.
_LEVELS = [r"\n\s*\n", r"\n", r"(?<=[.!?;:])\s+", r"\s+"]


@dataclass
class _Atom:
    text: str
    seg: int          # index into the segment list


def _split_keep(text: str, pattern: str) -> list[str]:
    pieces, last = [], 0
    for m in re.finditer(pattern, text):
        pieces.append(text[last:m.end()])
        last = m.end()
    if last < len(text):
        pieces.append(text[last:])
    return pieces


def _atomize(text: str, max_chars: int, level: int = 0) -> list[str]:
    """Recursively split `text` until every piece is <= max_chars (or cannot be split further)."""
    if len(text) <= max_chars:
        return [text]
    if level >= len(_LEVELS):  # no separators left: hard split
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    pieces = _split_keep(text, _LEVELS[level])
    if len(pieces) <= 1:
        return _atomize(text, max_chars, level + 1)
    out: list[str] = []
    for p in pieces:
        out.extend(_atomize(p, max_chars, level + 1) if len(p) > max_chars else [p])
    return out


def format_location(segs: list[Segment]) -> str:
    """Human readable location for a group of segments, e.g. "Pages 3-4" or "Sheet 'Q3' rows 41-80"."""
    if not segs:
        return ""
    groups: list[list[Segment]] = []
    for s in segs:
        if groups and (groups[-1][0].unit, groups[-1][0].label) == (s.unit, s.label):
            groups[-1].append(s)
        else:
            groups.append([s])
    parts = []
    for g in groups[:3]:
        unit, label = g[0].unit, g[0].label
        a = min(s.start for s in g)
        b = max(s.end for s in g)
        rng = f"{a}" if a == b else f"{a}-{b}"
        if unit == "Rows":
            parts.append(f"{label} rows {rng}" if label else f"Rows {rng}")
        elif unit == "Section":
            parts.append(f"Section '{label}'" if label else f"Section {rng}")
        elif unit == "Image":
            parts.append(label or "Image")
        elif unit:
            plural = unit + ("s" if a != b and not unit.endswith("s") else "")
            parts.append(f"{plural} {rng}" + (f" ({label})" if label else ""))
        else:
            parts.append(label or "")
    loc = "; ".join(p for p in parts if p)
    if len(groups) > 3:
        loc += " ..."
    return loc


def chunk_segments(segments: list[Segment], chunk_tokens: int = 600, overlap_ratio: float = 0.17) -> list[Chunk]:
    if not segments:
        return []
    max_tokens = max(50, chunk_tokens)
    overlap_tokens = int(max_tokens * min(max(overlap_ratio, 0.0), 0.4))
    # atoms must be small enough that the overlap can be built out of whole atoms
    max_atom_chars = max(160, overlap_tokens * 4) if overlap_tokens else max_tokens * 4

    atoms: list[_Atom] = []
    for si, seg in enumerate(segments):
        pieces = _atomize(seg.text, max_atom_chars)
        for p in pieces:
            atoms.append(_Atom(p, si))
        if atoms and not atoms[-1].text[-1:].isspace():
            atoms[-1].text += "\n\n"          # keep segments visually separated inside a chunk

    chunks: list[Chunk] = []
    cur: list[_Atom] = []
    cur_tokens = 0
    new_count = 0     # atoms in `cur` that were not carried over as overlap

    def emit():
        nonlocal chunks
        new_atoms = cur[len(cur) - new_count:]
        text = "".join(a.text for a in cur).strip()
        if not text:
            return
        seg_ids = sorted({a.seg for a in new_atoms})
        loc = format_location([segments[i] for i in seg_ids])
        chunks.append(Chunk(index=len(chunks), text=text, location=loc, tokens=est_tokens(text)))

    for atom in atoms:
        t = est_tokens(atom.text)
        if cur and cur_tokens + t > max_tokens and new_count > 0:
            emit()
            carry: list[_Atom] = []
            ct = 0
            for a in reversed(cur):
                at = est_tokens(a.text)
                if ct + at > overlap_tokens:
                    break
                carry.insert(0, a)
                ct += at
            cur, cur_tokens, new_count = carry, ct, 0
        cur.append(atom)
        cur_tokens += t
        new_count += 1
    if new_count > 0:
        emit()
    return chunks
