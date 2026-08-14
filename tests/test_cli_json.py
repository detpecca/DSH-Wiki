"""Tests for the machine-readable ``--json`` CLI surface.

These cover the adapter contract the DSH plugin relies on: every read-only
subcommand plus ingest prints exactly one JSON document to stdout when
``--json`` is given.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from llm_wiki.cli import main
from llm_wiki.schema import render_page, Page
from llm_wiki.store import WikiStore


def _page(path: str, title: str, aliases=None, tags=None) -> str:
    return render_page(Page(
        path=path,
        title=title,
        page_type=path.split("/")[0],
        aliases=aliases or [],
        tags=tags or [],
        summary=f"summary of {title}",
        key_facts=["fact one"],
        related_pages=[],
        related_sources=[],
    ), today="2026-01-01")


@pytest.fixture()
def wiki(tmp_path: Path):
    store = tmp_path / "wiki"
    store.mkdir()
    (store / "concepts").mkdir()
    (store / "entities").mkdir()
    (store / "concepts" / "retrieval.md").write_text(
        _page("concepts/retrieval", "Retrieval", aliases=["检索"], tags=["ai"]),
        encoding="utf-8")
    (store / "entities" / "paper.md").write_text(
        _page("entities/paper", "LLM-Wiki Paper", tags=["research"]),
        encoding="utf-8")
    (store / "sources" / "digests").mkdir(parents=True)
    (store / "sources" / "digests" / "s-001.md").write_text(
        "digest content\n", encoding="utf-8")
    # a real wiki keeps directory + global indices; validate requires them
    WikiStore(store).rebuild_all_indices()
    return tmp_path


def _run(argv: list[str]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return buf.getvalue(), code


def test_search_json(wiki: Path):
    out, code = _run(["--wiki", str(wiki / "wiki"), "search", "retrieval", "--json"])
    assert code == 0
    data = json.loads(out)
    assert isinstance(data, list) and data
    assert data[0]["path"] == "concepts/retrieval"
    assert data[0]["score"] > 0
    assert data[0]["aliases"] == ["检索"]


def test_read_json(wiki: Path):
    out, code = _run(["--wiki", str(wiki / "wiki"), "read",
                      "concepts/retrieval", "entities/paper", "missing/page", "--json"])
    assert code == 0
    data = json.loads(out)
    assert isinstance(data, dict)
    assert "## Key Facts" in data["concepts/retrieval"]
    assert data["entities/paper"].startswith("---")
    assert data["missing/page"] == "(page not found)"


def test_stats_json(wiki: Path):
    out, code = _run(["--wiki", str(wiki / "wiki"), "stats", "--json"])
    assert code == 0
    data = json.loads(out)
    assert data["pages"] == 2
    assert data["categories"] == {"concepts": 1, "entities": 1}
    assert data["digests"] == 1
    assert data["errorBookEntries"] == 0


def test_validate_json_ok(wiki: Path):
    out, code = _run(["--wiki", str(wiki / "wiki"), "validate", "--json"])
    assert code == 0
    data = json.loads(out)
    assert data["ok"] is True
    assert data["errors"] == []


def test_validate_json_errors(wiki: Path):
    (wiki / "wiki" / "concepts" / "broken.md").write_text(
        "# Broken\n\nNo required sections here.\n", encoding="utf-8")
    out, code = _run(["--wiki", str(wiki / "wiki"), "validate", "--json"])
    assert code == 1
    data = json.loads(out)
    assert data["ok"] is False
    assert any(e["page"] == "concepts/broken" for e in data["errors"])
    assert all(set(e) == {"type", "page", "detail"} for e in data["errors"])


def test_errorbook_json(wiki: Path):
    out, code = _run(["--wiki", str(wiki / "wiki"), "errorbook", "--json"])
    assert code == 0
    data = json.loads(out)
    assert data == {"entries": []}


def test_fix_json(wiki: Path):
    out, code = _run(["--wiki", str(wiki / "wiki"), "fix", "--json"])
    assert code == 0
    data = json.loads(out)
    assert set(data) == {"codeFixes", "finalized", "repaired",
                         "closedErrorEntries", "openErrorEntries"}
    assert data["finalized"] is False
    assert data["codeFixes"] == []
    assert data["repaired"] == []
    assert data["closedErrorEntries"] == 0
    assert data["openErrorEntries"] == 0
