"""Cross-page contradiction detection (sampling-based consistency check)."""

from llm_wiki import schema, validators
from llm_wiki.store import WikiStore

from conftest import FakeLLM


def _page(path, facts, related):
    return schema.render_page(schema.Page(
        path=path, title=path, summary="s", key_facts=facts,
        related_pages=related, related_sources=[("sources/digests/d", "n")],
    ), "2026-08-04")


def test_contradiction_detected_and_dedup_pairs(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    store.write("people/A", _page("people/A", ["born 1926"], [("people/B", "x")]))
    store.write("people/B", _page("people/B", ["born 1927"], [("people/A", "x")]))
    store.write("sources/digests/d", "x")
    # A->B and B->A form ONE pair after dedup
    llm = FakeLLM(["CONTRADICTION: birth year: A says 1926, B says 1927"])
    errors = validators.llm_consistency_check(llm, store)
    assert len(llm.seen) == 1  # deduplicated pair -> single LLM call
    assert len(errors) == 1
    assert errors[0].type == validators.CROSS_PAGE_CONTRADICTION
    assert "1926" in errors[0].detail


def test_consistent_pages_yield_no_errors(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    store.write("people/A", _page("people/A", ["born 1926"], [("people/B", "x")]))
    store.write("people/B", _page("people/B", ["born 1926"], []))
    llm = FakeLLM(["OK"])
    assert validators.llm_consistency_check(llm, store) == []


def test_sampling_caps_pairs(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    for i in range(10):  # 10 pages chained -> 9 pairs, cap at 3
        nxt = [("people/P" + str(i + 1), "next")] if i < 9 else []
        store.write(f"people/P{i}", _page(f"people/P{i}", ["f"], nxt))
    llm = FakeLLM(["OK"] * 10)
    validators.llm_consistency_check(llm, store, max_pairs=3)
    assert len(llm.seen) == 3
