"""Command-line interface: ingest / query / validate / fix / lint / errorbook.

Examples:
    python -m llm_wiki ingest notes.txt --wiki ./wiki
    python -m llm_wiki query "Which film has the older director?" --wiki ./wiki
    python -m llm_wiki validate --wiki ./wiki
    python -m llm_wiki fix --wiki ./wiki            # code autofix + finalize
    python -m llm_wiki errorbook --wiki ./wiki
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import agent, validators
from .compile import Compiler
from .error_book import ErrorBook
from .llm import LLMClient
from .schema import slugify
from .store import WikiStore


def _split_passages(text: str) -> list[str]:
    """Split a source file into passages (blank-line separated paragraphs,
    merged to a reasonable size)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def _components(args):
    store = WikiStore(args.wiki)
    book = ErrorBook(Path(args.wiki).parent / "error_book.yaml")
    llm = LLMClient()
    return store, book, llm


def cmd_ingest(args) -> int:
    store, book, llm = _components(args)
    compiler = Compiler(store, llm, book)
    text = Path(args.file).read_text(encoding="utf-8")
    passages = _split_passages(text)
    stem = slugify(Path(args.file).stem) or "source"
    batch = [(f"{stem}-{i:03d}", p) for i, p in enumerate(passages, 1)]
    print(f"ingesting {len(batch)} passages from {args.file} ...")
    written = compiler.compile_batch(batch)
    compiler.finalize()
    print(f"done: {len(written)} page updates written; wiki has "
          f"{len(store.iter_pages())} pages, {len(book.open_entries())} open error entries")
    if compiler.skipped:
        print(f"WARNING: {len(compiler.skipped)} passage(s) skipped due to failures:")
        for sid, reason in compiler.skipped:
            print(f"  - {sid}: {reason}")
    return 0


def cmd_query(args) -> int:
    store, _book, llm = _components(args)
    result = agent.run_agent(store, llm, args.question)
    for step in result["trace"]:
        print(f"  trace: {step}")
    print(f"\nANSWER: {result['answer']}")
    if result["evidence"]:
        print(f"EVIDENCE: {', '.join(result['evidence'])}")
    return 0


def cmd_validate(args) -> int:
    store, _book, _llm = _components(args)
    errors = validators.structural_validate(store)
    if not errors:
        print("OK: no structural errors")
        return 0
    for e in errors:
        print(e)
    print(f"{len(errors)} structural error(s)")
    return 1


def cmd_fix(args) -> int:
    store, book, llm = _components(args)
    compiler = Compiler(store, llm, book)
    fixes = compiler.code_fix_wiki()
    print(f"code fixes: {fixes or 'none needed'}")
    if args.finalize:
        compiler.finalize()
        print("finalization complete (3 rounds code-fix <-> LLM-fix)")
    return 0


def cmd_errorbook(args) -> int:
    _store, book, _llm = _components(args)
    for e in book.entries:
        print(f"#{e['id']} [{e['status']}] {e['type']} on {e['page']} "
              f"(x{e.get('occurrences', 1)})\n    rule: {e.get('constraint_rule', '-')}")
    if not book.entries:
        print("(error book is empty)")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to GBK; make Chinese output safe everywhere
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="llm_wiki", description=__doc__)
    ap.add_argument("--wiki", default="./wiki", help="wiki root directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="compile a source file into the wiki")
    p.add_argument("file")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("query", help="answer a question via traversal")
    p.add_argument("question")
    p.set_defaults(fn=cmd_query)

    p = sub.add_parser("validate", help="run structural validation")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("fix", help="code autofix; --finalize adds LLM repair rounds")
    p.add_argument("--finalize", action="store_true")
    p.set_defaults(fn=cmd_fix)

    p = sub.add_parser("errorbook", help="show error book entries")
    p.set_defaults(fn=cmd_errorbook)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
