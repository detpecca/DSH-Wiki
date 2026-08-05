from llm_wiki import schema


def test_slugify():
    assert schema.slugify("John V, Prince of Anhalt-Zerbst") == "John-V-Prince-of-Anhalt-Zerbst"
    assert schema.slugify("  Monster A Go-Go! ") == "Monster-A-Go-Go"


def test_extract_links_dedup_and_labels():
    text = "see [[people/A]] and [[people/A|alias]] plus [[sources/digests/x]]"
    assert schema.extract_links(text) == ["people/A", "sources/digests/x"]


def test_render_parse_roundtrip(tmp_path):
    page = schema.Page(
        path="people/A", title="A Title", aliases=["AA"], tags=["person"],
        summary="one line", key_facts=["f1", "f2"],
        related_pages=[("people/B", "friend of A")],
        related_sources=[("sources/digests/s-001", "origin")],
    )
    text = schema.render_page(page, "2026-08-04")
    assert text.startswith("---")
    for sec in schema.REQUIRED_SECTIONS:
        assert sec in text
    fm = schema.parse_frontmatter(text)
    assert fm["aliases"] == ["AA"] and fm["tags"] == ["person"]
    assert schema.parse_section_links(text, "Related Pages") == [("people/B", "friend of A")]
    assert schema.parse_section_links(text, "Related Sources") == [("sources/digests/s-001", "origin")]
