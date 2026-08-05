from llm_wiki.error_book import ErrorBook
from llm_wiki.validators import DANGLING_LINK, WikiError

from conftest import FakeLLM

TODAY = "2026-08-04"


def test_discover_and_merge(tmp_path):
    book = ErrorBook(tmp_path / "error_book.yaml")
    e = WikiError(DANGLING_LINK, "people/A", "link to missing [[people/Ghost]]")
    book.discover([e], TODAY)
    book.discover([e], TODAY)  # same pattern recurs -> merge
    assert len(book.entries) == 1
    assert book.entries[0]["occurrences"] == 2
    assert book.entries[0]["status"] == "open"


def test_attribute_inject_and_close_cycle(tmp_path):
    book = ErrorBook(tmp_path / "error_book.yaml")
    llm = FakeLLM([
        "- id: 1\n  root_cause: assumed page exists\n"
        "  constraint_rule: NEVER link to pages not in _index.md",
    ])
    entries = book.discover(
        [WikiError(DANGLING_LINK, "people/A", "link to missing [[people/Ghost]]")], TODAY)
    book.attribute_and_constrain(llm, entries)
    assert book.active_constraints() == ["NEVER link to pages not in _index.md"]

    # error gone on re-validation -> closed, constraint stops being injected
    closed = book.verify_and_close(still_failing=[])
    assert len(closed) == 1
    assert book.active_constraints() == []
    # persistence across reload
    book2 = ErrorBook(tmp_path / "error_book.yaml")
    assert book2.entries[0]["status"] == "closed"


def test_no_close_while_still_failing(tmp_path):
    book = ErrorBook(tmp_path / "error_book.yaml")
    e = WikiError(DANGLING_LINK, "people/A", "link to missing [[people/Ghost]]")
    entries = book.discover([e], TODAY)
    for ent in entries:
        ent["constraint_rule"] = "r"
    book.save()
    assert book.verify_and_close(still_failing=[e]) == []
    assert book.entries[0]["status"] == "open"
