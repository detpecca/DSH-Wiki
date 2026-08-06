"""Query-time agent: compositional Wiki traversal (paper §3.2, Figure 2).

The agent composes wiki_search / wiki_read calls through a ReAct-style
loop, following links and checking evidence sufficiency, until it answers
or hits a termination condition (paper §3.2 Termination):

  - evidence sufficient (all reasoning chains traced), or
  - tool-call budget T_max = 15 reached, or
  - consecutive empty searches exceed patience P = 3;
  - at least one wiki_read is required before answering.

Strategies (paper Appendix H), chosen adaptively by the agent:
  direct retrieval / link-following traversal / browse & aggregation.
"""

from __future__ import annotations

import json

from . import search
from .store import WikiStore

T_MAX = 15
PATIENCE = 3

SYSTEM_PROMPT = """You are a retrieval agent answering questions by traversing a
structured Wiki. You do NOT answer from memory — you gather evidence with
tools first.

TOOLS (reply with exactly one JSON action per turn):
{"tool": "wiki_search", "query": "<search terms>"}
  -> returns candidate pages with aliases/tags/summaries. Structured
     signals are matched first, so entity names and aliases work best.
{"tool": "wiki_read", "paths": ["<dir/Page>", "..."]}
  -> batch-reads pages or directory indices (e.g. "people/_index").
     Page content contains [[links]] to related pages — follow them.
{"tool": "answer", "answer": "<final answer>", "evidence": ["<dir/Page>", ...]}

STRATEGIES:
- Direct retrieval: single-entity question -> one search, read the page.
- Link-following: multi-hop question -> read page A, follow its [[links]]
  to page B, and so on. Each hop uses explicit links, not guesswork.
- Browse & aggregate: open-ended/enumeration question -> read a directory
  index ("<category>/_index") first for an overview, then batch-read the
  promising pages.

RULES:
- After every wiki_read, check sufficiency: do you have ALL the evidence
  the question needs? If not, follow links or issue a REVISED search.
- You must call wiki_read at least once before answering.
- Cite the pages your answer is grounded in.
- Reply with ONE JSON action and nothing else."""


def _parse_action(text: str) -> dict | None:
    """Extract the first balanced {...} block and parse it as an action."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    action = json.loads(text[start:i + 1])
                    return action if isinstance(action, dict) and "tool" in action else None
                except json.JSONDecodeError:
                    return None
    return None


def run_agent(store: WikiStore, llm, question: str,
              t_max: int = T_MAX, patience: int = PATIENCE) -> dict:
    """Run the traversal loop; returns {answer, evidence, trace}."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION: {question}"},
    ]
    trace: list[dict] = []
    reads = 0
    empty_searches = 0

    for _step in range(t_max):
        reply = llm.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        action = _parse_action(reply)
        if action is None:
            messages.append({"role": "user", "content": "Reply with ONE JSON action only."})
            continue
        tool = action["tool"]

        if tool == "wiki_search":
            results = search.search(store, action.get("query", ""))
            empty_searches = empty_searches + 1 if not results else 0
            trace.append({"tool": tool, "query": action.get("query"), "hits": len(results)})
            obs = json.dumps(results, ensure_ascii=False) if results else "(no results)"
            if empty_searches >= patience:
                trace.append({"tool": "terminate", "reason": "patience exceeded"})
                break
        elif tool == "wiki_read":
            paths = [p.removesuffix(".md") for p in action.get("paths", [])]
            contents = store.read_many(paths)
            reads += 1
            empty_searches = 0
            trace.append({"tool": tool, "paths": paths})
            obs = "\n\n".join(f"===== {p} =====\n{c}" for p, c in contents.items())
        elif tool == "answer":
            if reads == 0:  # paper: at least one wiki_read before answering
                messages.append({"role": "user", "content":
                    "You must call wiki_read at least once before answering."})
                continue
            trace.append({"tool": "answer", "answer": action.get("answer")})
            return {"answer": action.get("answer", ""),
                    "evidence": action.get("evidence", []), "trace": trace}
        else:
            obs = f"(unknown tool '{tool}')"
        messages.append({"role": "user", "content": f"OBSERVATION:\n{obs[:12000]}"})

    # budget or patience exhausted: force a best-effort answer from the trace
    return {"answer": "(terminated without sufficient evidence)",
            "evidence": [], "trace": trace}
