"""Index-time Wiki compilation: Algorithm 1 (paper Appendix D).

    for each source passage x in X:
        S  <- SelectPages(x, I)                     # LLM, k <= 5
        U  <- CompileWikiPages(x, S, C)             # LLM, constraints injected
        Es <- StructuralValidate(U, W)              # deterministic
        Ec <- ContentValidate(U, W, A)              # LLM, source-grounded
        if E != []:
            B <- UpdateErrorBook(B, E)
            C <- ActiveConstraints(B)
            U <- CodeAutoFix(U, Es)
        W  <- ApplyUpdates(W, U')
    if PeriodicFixDue(B):
        W <- LLMPeriodicFix(W, B)
        B <- VerifyAndClose(B, W)

Plus the finalization phase: 3 rounds of code-fix <-> LLM-fix (paper §3.3).
"""

from __future__ import annotations

import json
import re

from . import schema, validators
from .error_book import ErrorBook
from .store import WikiStore
from .validators import WikiError

# ------------------------------------------------------------------- prompts
SELECT_PAGES_PROMPT = """You are the page-selection step of a Wiki compiler.

A new source passage will be integrated into the Wiki. Below are the Wiki
directory indices (page names, aliases, one-line summaries).

DIRECTORY INDICES:
{indices}

NEW SOURCE PASSAGE:
{passage}

Select up to {k} EXISTING pages that are most relevant to this passage
(pages that should be updated, or that the passage's entities should link
to). Reply with a JSON array of page paths only, e.g.:
["people/John-V-Prince-of-Anhalt-Zerbst", "events/German-Reformation"]
Reply [] if none are relevant."""

COMPILE_PAGES_PROMPT = """You are the page-compilation step of a Wiki compiler. You turn a source
passage into structured Wiki updates.

NEW SOURCE PASSAGE (id: {source_id}):
{passage}

SELECTED EXISTING PAGES (you may update these, and ONLY these):
{selected}

{constraints_block}

Output STRICT JSON (no markdown fence) with this shape:
{{
  "digest": {{"id": "{source_id}", "summary": "2-4 sentence digest of the passage"}},
  "pages": [
    {{
      "path": "<category>/<Page-Name>",   // category: concepts/entities/events/systems/benchmarks/topics
      "title": "Human Readable Title",
      "is_new": true,
      "aliases": ["..."],
      "tags": ["..."],
      "summary": "one-line summary",
      "key_facts": ["atomic fact 1", "..."],
      "related_pages": [["<category>/<Other-Page>", "relation note"]],
      "related_sources": [["sources/digests/{source_id}", "what it supports"]]
    }}
  ]
}}

Rules:
- You may create new pages and update the SELECTED existing pages. Do NOT
  invent updates to other existing pages.
- Every page must cite at least one related source digest.
- Link generously to related pages, but ONLY to pages that exist (see
  selected pages / indices) or pages you create in this same output.
- Facts must be grounded in the passage; do not add outside knowledge."""

CONSTRAINTS_HEADER = "CONSTRAINTS from the Error Book (follow all of them):"

REPAIR_PAGE_PROMPT = """The following Wiki page contains facts NOT supported by
its cited source digests.

PAGE:
{page_text}

CITED DIGESTS:
{digests}

UNSUPPORTED FACTS:
{facts}

Rewrite the page: remove or correct the unsupported facts, keep everything
else in the same format. Output the full corrected page text only."""


# ------------------------------------------------------------- JSON parsing
def _extract_json(text: str):
    """Tolerant JSON extraction from an LLM reply."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    for i, ch in enumerate(text):
        if ch in "[{":
            for j in range(len(text), i, -1):
                try:
                    return json.loads(text[i:j])
                except json.JSONDecodeError:
                    continue
            break
    raise ValueError(f"no JSON found in LLM reply: {text[:200]}")


def _normalize_update(update) -> dict:
    """Coerce a raw LLM update into the expected shape; drop unusable entries.

    LLM replies are untrusted: the top level may be a list, pages may lack a
    ``path``, link entries may be malformed. Everything unusable is dropped
    here so downstream code can assume well-formed input.
    """
    if not isinstance(update, dict):
        return {"digest": {}, "pages": []}
    if not isinstance(update.get("digest"), dict):
        update["digest"] = {}
    pages = []
    for p in update.get("pages") or []:
        if not isinstance(p, dict):
            continue
        path = p.get("path")
        if not isinstance(path, str) or "/" not in path:
            continue
        p["related_pages"] = [
            (str(x[0]), str(x[1]) if len(x) > 1 else "")
            for x in (p.get("related_pages") or [])
            if isinstance(x, (list, tuple)) and len(x) >= 1 and isinstance(x[0], str)
        ]
        p["related_sources"] = [
            (str(x[0]), str(x[1]) if len(x) > 1 else "")
            for x in (p.get("related_sources") or [])
            if isinstance(x, (list, tuple)) and len(x) >= 1 and isinstance(x[0], str)
        ]
        for key, default in (("aliases", []), ("tags", []), ("key_facts", [])):
            if not isinstance(p.get(key), list):
                p[key] = default
        pages.append(p)
    update["pages"] = pages
    return update


class Compiler:
    def __init__(self, store: WikiStore, llm, book: ErrorBook,
                 k: int = 5, periodic_every: int = 10):
        self.store = store
        self.llm = llm
        self.book = book
        self.k = k                      # SelectPages budget (paper §4.4)
        self.periodic_every = periodic_every
        self.articles_since_fix = 0
        self.skipped: list[tuple[str, str]] = []  # (source_id, reason) failures

    # ------------------------------------------------------- SelectPages (2)
    def select_pages(self, passage: str) -> list[str]:
        reply = self.llm.chat([{"role": "user", "content": SELECT_PAGES_PROMPT.format(
            indices=self.store.directory_index_listing(), passage=passage, k=self.k)}])
        try:
            pages = _extract_json(reply)
        except ValueError:
            return []
        return [p for p in pages if isinstance(p, str) and self.store.exists(p)][: self.k]

    # --------------------------------------------------- CompileWikiPages (3)
    def compile_pages(self, passage: str, source_id: str, selected: list[str]) -> dict:
        constraints = self.book.active_constraints()
        constraints_block = ""
        if constraints:
            constraints_block = CONSTRAINTS_HEADER + "\n" + "\n".join(f"- {c}" for c in constraints)
        selected_text = "\n\n".join(
            f"--- {p} ---\n{self.store.read(p)}" for p in selected) or "(none selected)"
        reply = self.llm.chat([{"role": "user", "content": COMPILE_PAGES_PROMPT.format(
            passage=passage, source_id=source_id, selected=selected_text,
            constraints_block=constraints_block)}])
        return _extract_json(reply)

    # --------------------------------------------------------- CodeAutoFix (10)
    def code_autofix(self, update: dict, errors: list[WikiError]) -> dict:
        """Deterministic repair of structural errors in a proposed update."""
        pages = update.get("pages", [])
        new_paths = {p["path"] for p in pages if p.get("is_new")}
        known = new_paths | {r for r in self.store.iter_pages()}

        bad_pages = {e.page for e in errors if e.type == validators.UNSEEN_OVERWRITE}
        pages = [p for p in pages if p["path"] not in bad_pages]

        for p in pages:
            # drop dangling links (target neither exists nor is created now)
            p["related_pages"] = [rp for rp in p.get("related_pages", []) if rp[0] in known]
            # drop malformed source refs
            p["related_sources"] = [
                rs for rs in p.get("related_sources", [])
                if schema.SOURCE_REF_RE.match(f"[[{rs[0]}]]")
            ]
        update["pages"] = pages
        return update

    # --------------------------------------------------------- ApplyUpdates (12)
    def apply_updates(self, update: dict, source_id: str, passage: str) -> list[str]:
        today = self.store.today()
        written = []

        digest = update.get("digest") or {}
        digest_text = (
            f"# Digest: {digest.get('id', source_id)}\n\n"
            f"{digest.get('summary', '')}\n\n## Original\n\n{passage}\n"
        )
        self.store.write(f"sources/digests/{source_id}", digest_text)

        for p in update.get("pages", []):
            path = p["path"]
            existing_created = ""
            if self.store.exists(path):
                existing_created = schema.parse_frontmatter(self.store.read(path)).get("created", "")
            page = schema.Page(
                path=path,
                title=p.get("title") or path.split("/")[-1].replace("-", " "),
                page_type=p.get("type") or path.split("/")[0],
                aliases=p.get("aliases", []),
                tags=p.get("tags", []),
                summary=p.get("summary", ""),
                key_facts=p.get("key_facts", []),
                related_pages=[tuple(x) for x in p.get("related_pages", [])],
                related_sources=[tuple(x) for x in p.get("related_sources", [])],
                created=existing_created or today,
                updated=today,
            )
            self.store.write(path, schema.render_page(page, today))
            written.append(path)

        for p in update.get("pages", []):  # bidirectional links
            for target, note in p.get("related_pages", []):
                if self.store.exists(target):
                    self.store.add_backlink(target, p["path"], note="")

        self.store.rebuild_indices_for(written)
        return written

    # ------------------------------------------------------------ main loop
    def compile_passage(self, passage: str, source_id: str) -> list[str]:
        """One iteration of the Algorithm-1 for-loop."""
        selected = self.select_pages(passage)                       # 2: S
        try:
            update = self.compile_pages(passage, source_id, selected)  # 3: U
        except (ValueError, TypeError, KeyError) as e:
            # malformed LLM reply: skip this passage instead of crashing the batch
            self.skipped.append((source_id, f"compile_pages failed: {e}"))
            return []
        update = _normalize_update(update)

        today = self.store.today()
        errors: list[WikiError] = []
        errors += validators.check_unseen_overwrite(                # 4: Es (in-Wiki part)
            {p["path"] for p in update.get("pages", []) if not p.get("is_new")},
            set(selected),
            {p["path"] for p in update.get("pages", []) if p.get("is_new")},
        )
        errors += validators.check_update(update, self.store)       # 4: Es (in-update links/refs)

        if errors:                                                  # 8-9: B, C
            new_entries = self.book.discover(errors, today)
            self.book.attribute_and_constrain(self.llm, new_entries)
        update = self.code_autofix(update, errors)                  # 10: always sanitize U

        written = self.apply_updates(update, source_id, passage)    # 12: W

        # content validation happens after application (needs digests on disk)
        content_errors = validators.llm_content_validate(self.llm, self.store, written)
        if content_errors:
            new_entries = self.book.discover(content_errors, today)
            self.book.attribute_and_constrain(self.llm, new_entries)

        self.articles_since_fix += 1
        if self.articles_since_fix >= self.periodic_every:          # 14-17
            self.llm_periodic_fix()
            self.verify_and_close()
            self.articles_since_fix = 0
        return written

    def compile_batch(self, passages: list[tuple[str, str]]) -> list[str]:
        """X: list of (source_id, passage). Archives each source article.

        A failing passage never aborts the batch: it is recorded in
        ``self.skipped`` and the loop continues with the next passage.
        """
        written: list[str] = []
        for source_id, passage in passages:
            self.store.write(f"sources/articles/{source_id}", passage)
            try:
                written += self.compile_passage(passage, source_id)
            except Exception as e:  # noqa: BLE001 — batch isolation is the point
                self.skipped.append((source_id, f"unexpected: {type(e).__name__}: {e}"))
        return written

    # ------------------------------------------------------------- repairs
    def code_fix_wiki(self) -> list[str]:
        """Layer-1 Code Auto-fix over the whole Wiki (deterministic)."""
        fixes = self.store.sync_bidirectional_links()
        errors = validators.check_index_consistency(self.store)
        if errors:
            self.store.rebuild_all_indices()
            fixes.append("rebuilt indices")
        return fixes

    def llm_periodic_fix(self, pages: list[str] | None = None) -> list[str]:
        """Layer-2 LLM Periodic Fix: repair content-level errors (semantic)."""
        repaired = []
        targets = pages if pages is not None else self.store.iter_pages()
        errors = validators.llm_content_validate(self.llm, self.store, targets)
        for err in errors:
            text = self.store.read(err.page)
            digests = "\n\n".join(
                self.store.read(t) for t, _ in schema.parse_section_links(text, "Related Sources")
                if self.store.exists(t))
            fixed = self.llm.chat([{"role": "user", "content": REPAIR_PAGE_PROMPT.format(
                page_text=text, digests=digests, facts=err.detail)}])
            if fixed.strip().startswith("---"):
                self.store.write(err.page, fixed.strip() + "\n")
                repaired.append(err.page)
        self.store.rebuild_all_indices()
        return repaired

    def verify_and_close(self) -> list[dict]:
        """Targeted re-validation (stage 5): re-check only the pages and error
        types that have open entries, instead of re-validating the whole Wiki."""
        still: list[WikiError] = []
        open_entries = self.book.open_entries()
        types = {e["type"] for e in open_entries}

        structural_pages = sorted(
            {e["page"] for e in open_entries
             if e["type"] in validators.STRUCTURAL_TYPES
             and e["type"] != validators.INDEX_INCONSISTENCY and e["page"]})
        if structural_pages:
            still += validators.structural_validate(self.store, structural_pages)
        if validators.INDEX_INCONSISTENCY in types:
            still += validators.check_index_consistency(self.store)

        content_pages = sorted(
            {e["page"] for e in open_entries if e["type"] == validators.UNSUPPORTED_FACT})
        if content_pages:
            still += validators.llm_content_validate(self.llm, self.store, content_pages)
        if validators.CROSS_PAGE_CONTRADICTION in types:
            still += validators.llm_consistency_check(self.llm, self.store)
        return self.book.verify_and_close(still)

    def finalize(self, rounds: int = 3) -> None:
        """Finalization: 3 rounds of code-fix <-> LLM-fix (paper §3.3),
        then a sampling-based cross-page consistency sweep."""
        for _ in range(rounds):
            self.code_fix_wiki()
            self.llm_periodic_fix()
        contradictions = validators.llm_consistency_check(self.llm, self.store)
        if contradictions:
            new_entries = self.book.discover(contradictions, self.store.today())
            self.book.attribute_and_constrain(self.llm, new_entries)
        self.verify_and_close()
