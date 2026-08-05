from llm_wiki import schema, validators
from llm_wiki.store import WikiStore
from llm_wiki.validators import (
    DANGLING_LINK, INDEX_INCONSISTENCY, INCOMPLETE_PAGE, MALFORMED_REF,
    UNSEEN_OVERWRITE,
)


def _good_page(path):
    return schema.render_page(schema.Page(
        path=path, title=path, summary="s", key_facts=["f"],
        related_sources=[("sources/digests/d-1", "n")],
    ), "2026-08-04")


def test_dangling_link_detected(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    store.write("sources/digests/d-1", "digest")
    store.write("people/A", _good_page("people/A") + "\nsee [[people/Ghost]]\n")
    errors = validators.check_dangling_links(store)
    assert len(errors) == 1 and errors[0].type == DANGLING_LINK
    assert "people/Ghost" in errors[0].detail
    assert "people/Ghost" in errors[0].detail


def test_incomplete_page_detected(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    store.write("people/A", "# just a heading\n")
    errors = validators.check_incomplete_pages(store)
    assert errors and all(e.type == INCOMPLETE_PAGE for e in errors)


def test_malformed_ref_detected(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    bad = _good_page("people/A").replace(
        "[[sources/digests/d-1]]", "[[random/place]]")
    store.write("people/A", bad)
    errors = validators.check_malformed_refs(store)
    assert len(errors) == 1 and errors[0].type == MALFORMED_REF


def test_index_inconsistency_both_directions(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    store.write("people/A", _good_page("people/A"))
    store.write("people/_index", "# people\n- [[people/Ghost]]\n")
    errors = validators.check_index_consistency(store)
    kinds = {e.detail for e in errors}
    assert any("not in _index" in d for d in kinds)      # on disk, unlisted
    assert any("missing on disk" in d for d in kinds)    # listed, absent


def test_unseen_overwrite():
    errors = validators.check_unseen_overwrite(
        updated_paths={"people/A", "people/Sneaky"},
        selected={"people/A"},
        new_pages=set(),
    )
    assert [e.page for e in errors] == ["people/Sneaky"]
    assert errors[0].type == UNSEEN_OVERWRITE


def test_structural_validate_clean(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    store.write("people/A", _good_page("people/A"))
    store.write("sources/digests/d-1", "x")
    store.rebuild_all_indices()
    assert validators.structural_validate(store) == []
