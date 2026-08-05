"""Shared test fixtures: a FakeLLM driven by queued / rule-based replies."""

from __future__ import annotations

import re


class FakeLLM:
    """Deterministic LLM stand-in.

    replies: list of strings popped in order; fallback(rule) used when the
    queue is empty. Records every prompt it receives in `seen`.
    """

    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies or [])
        self.seen: list[str] = []

    def chat(self, messages, **kwargs) -> str:
        prompt = messages[-1]["content"]
        self.seen.append(prompt)
        if self.replies:
            return self.replies.pop(0)
        return self.fallback(prompt)

    def fallback(self, prompt: str) -> str:  # noqa: ARG002
        return "OK"


def compile_reply(pages: list[dict], digest_id: str = "s-001") -> str:
    """Build a CompileWikiPages JSON reply for given page dicts."""
    import json

    return json.dumps({"digest": {"id": digest_id, "summary": "test digest"},
                       "pages": pages})


def make_page(path: str, title: str = "T", is_new: bool = True,
              related_pages=None, related_sources=None, key_facts=None) -> dict:
    return {
        "path": path,
        "title": title,
        "is_new": is_new,
        "aliases": [],
        "tags": [],
        "summary": f"summary of {path}",
        "key_facts": key_facts or ["fact one"],
        "related_pages": related_pages or [],
        "related_sources": related_sources or [["sources/digests/s-001", "supports"]],
    }


def agent_script(*actions) -> list[str]:
    """Build agent replies: JSON actions in order."""
    import json

    return [json.dumps(a) for a in actions]


def read_action(*paths) -> dict:
    return {"tool": "wiki_read", "paths": list(paths)}


def search_action(query: str) -> dict:
    return {"tool": "wiki_search", "query": query}


def answer_action(answer: str, evidence=None) -> dict:
    return {"tool": "answer", "answer": answer, "evidence": evidence or []}
