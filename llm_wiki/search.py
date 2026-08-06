"""wiki_search implementation (paper §3.2 Tool Interface).

Prioritizes structured signals — page names, aliases, tags, summaries —
before falling back to page content. Pure Python scoring; no embeddings.
"""

from __future__ import annotations

import re

from . import schema
from .store import WikiStore

WEIGHTS = {"name": 8, "alias": 6, "tag": 4, "summary": 2, "content": 1}


def _tokens(query: str) -> list[str]:
    """Tokenize a query: alphanumeric words plus CJK bigrams.

    Chinese text has no whitespace separators, so CJK runs are broken into
    overlapping bigrams (e.g. "导演年龄" -> ["导演", "演年", "年龄"]);
    1-2 char runs are kept whole.
    """
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 1]
    for run in re.findall("[一-鿿]+", query):
        if len(run) <= 2:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def search(store: WikiStore, query: str, limit: int = 10) -> list[dict]:
    """Return candidate pages with metadata, best first."""
    tokens = _tokens(query)
    if not tokens:
        return []
    scored = []
    for rel in store.iter_pages():
        text = store.read(rel)
        fm = schema.parse_frontmatter(text)
        name = rel.split("/")[-1].replace("-", " ").lower()
        aliases = [a.lower() for a in fm.get("aliases", [])]
        tags = [t.lower() for t in fm.get("tags", [])]
        m = re.search(r"^>\s*(.+)$", text, re.M)
        summary = (m.group(1).lower() if m else "")
        content = text.lower()

        score, matched = 0, set()
        for tok in tokens:
            if tok in name:
                score += WEIGHTS["name"]; matched.add(tok)
            if any(tok in a for a in aliases):
                score += WEIGHTS["alias"]; matched.add(tok)
            if any(tok in t for t in tags):
                score += WEIGHTS["tag"]; matched.add(tok)
            if tok in summary:
                score += WEIGHTS["summary"]; matched.add(tok)
            elif tok in content:  # content fallback only if no structured hit
                score += WEIGHTS["content"]; matched.add(tok)
        if score:
            scored.append((score, len(matched), rel, fm))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [
        {
            "path": rel,
            "score": score,
            "aliases": fm.get("aliases", []),
            "tags": fm.get("tags", []),
            "summary": _summary_of(store.read(rel)),
        }
        for score, _n, rel, fm in scored[:limit]
    ]


def _summary_of(text: str) -> str:
    m = re.search(r"^>\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else ""
