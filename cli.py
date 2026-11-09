#!/usr/bin/env python3
"""
Command line interface for the AI file finder.

    python cli.py index [--force] [--dirs DIR ...]   scan folders (unchanged files are skipped)
    python cli.py watch                              keep the index live (initial scan + file watcher)
    python cli.py search "the doc with the Q3 pump efficiency calculations" [--top 5] [--trace] [--json]
    python cli.py chat                               interactive search prompt
    python cli.py status                             what is indexed, which models are active
    python cli.py check [--repair]                   verify SQLite <-> vector DB consistency
    python cli.py open "<query>"                     search and open the best match
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time

from config import load_settings


def _print_response(resp, trace: bool) -> None:
    print(f"\n{resp.message}   [{resp.elapsed_s:.2f}s]")
    for i, m in enumerate(resp.matches, start=1):
        conf = f"{m.confidence:.2f}" if m.confidence is not None else "-"
        flag = "OK " if m.verified else "?? "
        print(f"\n{i}. {flag}{m.file_name}   confidence {conf}   (modified {m.modified})")
        print(f"   path:     {m.file_path}")
        if m.evidence_location:
            print(f"   where:    {m.evidence_location}")
        if m.reason:
            print(f"   why:      {m.reason}")
        if m.evidence:
            print(f"   evidence: \"{m.evidence}\"")
        print(f"   link:     {m.open_uri}")
    if trace:
        print("\n--- search trace ---")
        for n, a in enumerate(resp.attempts, start=1):
            p = a.plan
            print(f"attempt {n} [{p.planner}] semantic={p.semantic_query!r} keywords={p.exact_keywords} "
                  f"filters={p.filters.describe()}")
            print(f"   {a.note}; verifier={a.verifier}; confident={a.confident} top={a.top_confidence:.2f}"
                  + (f"; {p.notes}" if p.notes else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="file-finder", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index"); p.add_argument("--force", action="store_true"); p.add_argument("--dirs", nargs="*")
    sub.add_parser("watch")
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("--top", type=int, default=None)
    p.add_argument("--trace", action="store_true"); p.add_argument("--json", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    sub.add_parser("chat")
    sub.add_parser("status")
    p = sub.add_parser("check"); p.add_argument("--repair", action="store_true")
    p = sub.add_parser("open"); p.add_argument("query")
    args = ap.parse_args(argv)

    overrides = {"watch_dirs": args.dirs} if getattr(args, "dirs", None) else {}
    settings = load_settings(**overrides)

    from agent.pipeline import FileFinder
    finder = FileFinder(settings)
    try:
        if args.cmd == "index":
            def prog(i, n, path, status):
                if status in ("indexed", "failed") or i % 50 == 0 or i == n:
                    print(f"[{i}/{n}] {status:16s} {path}", file=sys.stderr)
            stats = finder.index_all(progress=prog, force=args.force)
            print(stats.summary())
            for e in stats.errors[:10]:
                print("  error:", e)
        elif args.cmd == "watch":
            from indexer.file_watcher import FileWatcher
            w = FileWatcher(finder.store, settings)
            w.start()
            print("Watching for changes. Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                w.stop()
        elif args.cmd == "search":
            resp = finder.search(args.query, top_k=args.top, verify=not args.no_verify)
            if args.json:
                print(json.dumps(dataclasses.asdict(resp), indent=2, default=str))
            else:
                _print_response(resp, args.trace)
        elif args.cmd == "chat":
            print("Describe the file you want ('q' to quit).")
            while True:
                try:
                    q = input("\nfind> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q.lower() in ("q", "quit", "exit"):
                    break
                if q:
                    _print_response(finder.search(q), trace=False)
        elif args.cmd == "status":
            print(json.dumps(finder.status(), indent=2, default=str))
        elif args.cmd == "check":
            print(json.dumps(finder.store.consistency_check(repair=args.repair), indent=2))
        elif args.cmd == "open":
            from opener import open_file
            resp = finder.search(args.query)
            _print_response(resp, False)
            if resp.matches:
                open_file(resp.matches[0].file_path)
    finally:
        finder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
