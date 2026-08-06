"""Self-audit fixes: path traversal, read_many tolerance, corrupted YAML,
balanced-brace action parsing."""

import pytest

from llm_wiki.agent import _parse_action
from llm_wiki.compile import _normalize_update
from llm_wiki.error_book import ErrorBook
from llm_wiki.store import WikiStore


def test_page_file_rejects_traversal(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    for bad in ("../../etc/passwd", "..\\win.ini", "/abs/path", "C:/x"):
        with pytest.raises(ValueError):
            store.page_file(bad)
    assert store.page_file("people/A").name == "A.md"  # normal path unaffected


def test_normalize_drops_traversal_paths():
    update = _normalize_update({"pages": [
        {"path": "../../evil", "title": "x"},
        {"path": "people/A", "title": "ok"},
    ]})
    assert [p["path"] for p in update["pages"]] == ["people/A"]


def test_read_many_tolerates_bad_paths(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    out = store.read_many(["../../etc/passwd", "people/Ghost"])
    assert out["../../etc/passwd"] == "(invalid or unreadable path)"
    assert out["people/Ghost"] == "(page not found)"


def test_error_book_loads_corrupted_yaml(tmp_path):
    path = tmp_path / "error_book.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    book = ErrorBook(path)  # must not crash
    assert book.entries == []
    path.write_text("{unclosed: [", encoding="utf-8")
    assert ErrorBook(path).entries == []


def test_parse_action_balanced_braces():
    a = _parse_action('sure! {"tool": "wiki_read", "paths": ["a"]} and {"other": 1}')
    assert a == {"tool": "wiki_read", "paths": ["a"]}
    assert _parse_action('{"tool": "answer", "answer": "x", "evidence": []}')["tool"] == "answer"
    assert _parse_action("no json here") is None
    assert _parse_action("{broken") is None
