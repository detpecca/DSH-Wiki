"""Robustness: malformed LLM replies must never crash an ingest batch."""

from llm_wiki.compile import Compiler, _normalize_update
from llm_wiki.error_book import ErrorBook
from llm_wiki.store import WikiStore

from conftest import FakeLLM, compile_reply, make_page


def _setup(tmp_path, llm):
    store = WikiStore(tmp_path / "wiki")
    book = ErrorBook(tmp_path / "error_book.yaml")
    return store, book, Compiler(store, llm, book)


def test_array_instead_of_object_is_skipped(tmp_path):
    _store, _book, c = _setup(tmp_path, FakeLLM(['[]', '[{"pages": []}]']))
    assert c.compile_passage("p", "s-001") == []  # no crash
    # expect=dict: a bare array is not a valid compile reply -> recorded skip
    assert len(c.skipped) == 1 and c.skipped[0][0] == "s-001"


def test_page_missing_path_is_dropped(tmp_path):
    reply = '{"digest": {"id": "s-001", "summary": "x"}, "pages": [{"title": "NoPath"}]}'
    store, _book, c = _setup(tmp_path, FakeLLM(["[]", reply]))
    assert c.compile_passage("p", "s-001") == []
    assert store.iter_pages() == []


def test_no_json_at_all_is_skipped(tmp_path):
    _store, _book, c = _setup(tmp_path, FakeLLM(["[]", "Sorry, I cannot help."]))
    assert c.compile_passage("p", "s-001") == []
    assert len(c.skipped) == 1 and c.skipped[0][0] == "s-001"


def test_one_bad_passage_does_not_kill_batch(tmp_path):
    llm = FakeLLM([
        "[]", "totally not json",                       # passage 1: fails
        "[]", compile_reply([                           # passage 2: succeeds
            make_page("people/A", "A")], digest_id="s-002"),
        "OK",
    ])
    store, _book, c = _setup(tmp_path, llm)
    written = c.compile_batch([("s-001", "bad"), ("s-002", "good")])
    assert written == ["people/A"]     # later passage still compiled
    assert len(c.skipped) == 1
    assert store.exists("people/A")


def test_normalize_coerces_link_pairs():
    update = _normalize_update({"pages": [{
        "path": "people/A",
        "related_pages": [["people/B"], "garbage", 42],
        "related_sources": [],
    }]})
    page = update["pages"][0]
    assert page["related_pages"] == [("people/B", "")]  # garbage dropped, pair padded
