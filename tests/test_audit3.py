"""Third-audit defects: A (repair invisibility), B (dangling digest refs),
C (digest id), D (path safety parity), E (JSON span typing), F (string-aware
action parsing), plus medium items (targeted periodic fix, repair re-check)."""

import json

from llm_wiki import schema, validators
from llm_wiki.compile import Compiler, _extract_json, _normalize_update
from llm_wiki.error_book import ErrorBook
from llm_wiki.store import WikiStore
from llm_wiki.agent import _parse_action

from conftest import FakeLLM


def _setup(tmp_path, llm):
    store = WikiStore(tmp_path / "wiki")
    book = ErrorBook(tmp_path / "error_book.yaml")
    return store, book, Compiler(store, llm, book)


def _page_with_fact(tmp_path, store):
    page = schema.render_page(schema.Page(
        path="people/A", title="A", summary="s", key_facts=["A was born on Mars"],
        related_sources=[("sources/digests/d1", "n")]), "2026-01-01")
    store.write("people/A", page)
    store.write("sources/digests/d1", "A was born on Earth")
    return page


# --- A ---------------------------------------------------------------------
def test_repair_stripping_citations_is_rejected(tmp_path):
    store, book, c = _setup(tmp_path, FakeLLM([]))
    original = _page_with_fact(tmp_path, store)
    stripped = original.replace("- [[sources/digests/d1]] -- n", "- (none)")
    c.llm.replies = ["UNSUPPORTED: A was born on Mars", stripped]
    repaired = c.llm_periodic_fix(pages=["people/A"])
    assert repaired == []                        # repair rejected
    assert store.read("people/A") == original    # page untouched


def test_facts_without_digests_are_flagged_without_llm(tmp_path):
    store, _book, _c = _setup(tmp_path, FakeLLM([]))
    page = _page_with_fact(tmp_path, store).replace("- [[sources/digests/d1]] -- n", "- (none)")
    store.write("people/A", page)
    llm = FakeLLM([])  # no replies queued: must not be called
    errors = validators.llm_content_validate(llm, store, ["people/A"])
    assert len(errors) == 1 and errors[0].type == validators.UNSUPPORTED_FACT
    assert llm.seen == []


# --- B ---------------------------------------------------------------------
def test_dangling_digest_ref_dropped_and_booked(tmp_path):
    update = {"digest": {"id": "s-001", "summary": "x"}, "pages": [
        {"path": "people/X", "title": "X", "is_new": True, "aliases": [], "tags": [],
         "summary": "s", "key_facts": ["f"], "related_pages": [],
         "related_sources": [["sources/digests/NOPE", "ghost"]]}]}
    llm = FakeLLM(["[]", json.dumps(update),
                   "- id: 1\n  root_cause: cited future digest\n  constraint_rule: ONLY cite digests of this passage",
                   "OK"])
    store, book, c = _setup(tmp_path, llm)
    written = c.compile_passage("p", "s-001")
    assert written == ["people/X"]
    refs = schema.parse_section_links(store.read("people/X"), "Related Sources")
    assert ("sources/digests/NOPE", "ghost") not in refs   # dropped at apply
    assert any(e["type"] == validators.DANGLING_LINK for e in book.entries)


# --- C ---------------------------------------------------------------------
def test_digest_heading_uses_source_id_not_llm_id(tmp_path):
    update = {"digest": {"id": "WRONG-ID", "summary": "x"}, "pages": []}
    llm = FakeLLM(["[]", json.dumps(update)])
    store, _book, c = _setup(tmp_path, llm)
    c.compile_passage("p", "paper-001")
    assert store.read("sources/digests/paper-001").startswith("# Digest: paper-001")
    assert not store.exists("sources/digests/WRONG-ID")


# --- D ---------------------------------------------------------------------
def test_normalize_drops_colon_and_backslash_paths():
    update = _normalize_update({"pages": [
        {"path": "people/A:B"}, {"path": "people/A\\B"}, {"path": "people/OK"},
    ]})
    assert [p["path"] for p in update["pages"]] == ["people/OK"]


# --- E ---------------------------------------------------------------------
def test_extract_json_skips_chatter_object_before_array():
    assert _extract_json('Sure! {"note": "x"} ["people/A"]', expect=list) == ["people/A"]
    assert _extract_json('here: [1,2] and finally {"pages": []}', expect=dict) == {"pages": []}


# --- F ---------------------------------------------------------------------
def test_parse_action_with_brace_inside_string():
    a = _parse_action('{"tool":"answer","answer":"it ends with }","evidence":[]}')
    assert a["answer"] == "it ends with }"


# --- medium ----------------------------------------------------------------
def test_periodic_fix_no_targets_no_llm_calls(tmp_path):
    _store, _book, c = _setup(tmp_path, FakeLLM([]))
    assert c.llm_periodic_fix() == []   # nothing dirty, nothing open
    assert c.llm.seen == []             # zero LLM calls


def test_repair_then_verify_closes_entry(tmp_path):
    store, book, c = _setup(tmp_path, FakeLLM([]))
    original = _page_with_fact(tmp_path, store)
    book.discover([validators.WikiError(
        validators.UNSUPPORTED_FACT, "people/A", "A was born on Mars")], store.today())
    for e in book.entries:
        e["constraint_rule"] = "ground facts"
    book.save()
    good_repair = original.replace("- A was born on Mars", "- A was born on Earth")
    c.llm.replies = [
        "UNSUPPORTED: A was born on Mars",  # content validate finds the fact
        good_repair,                        # repair keeps citation, fixes fact
        "OK",                               # verify_and_close re-check passes
    ]
    assert c.llm_periodic_fix(pages=["people/A"]) == ["people/A"]
    closed = c.verify_and_close()
    assert len(closed) == 1 and book.open_entries() == []
