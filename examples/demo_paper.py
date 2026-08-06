"""End-to-end demo: compile the LLM-Wiki paper itself, then traverse it.

Runs the REAL system (compile pipeline, validators, error book, store,
search, agent loop). The LLM steps are scripted here because this demo runs
without an API key; the scripted outputs are genuine compilations of the
paper content. With LLM_WIKI_API_KEY set, `python -m llm_wiki ingest` does
the same thing fully automatically.

Usage:  .venv/Scripts/python examples/demo_paper.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_wiki import agent, validators  # noqa: E402
from llm_wiki.compile import Compiler  # noqa: E402
from llm_wiki.error_book import ErrorBook  # noqa: E402
from llm_wiki.store import WikiStore  # noqa: E402

# --- three real passages from the paper (abstract / §3.1 / §3.3) ----------
PASSAGES = [
    ("paper-001", "We propose LLM-Wiki, an agent-native retrieval system that "
     "operationalizes the Retrieval-as-Reasoning paradigm. LLM-Wiki compiles "
     "documents into structured Wiki pages with bidirectional links, exposes "
     "search, read, and link-following operations through standard tool-calling "
     "interfaces, and introduces an Error Book for persistent structural and "
     "semantic self-correction. It outperforms HippoRAG 2, LightRAG, and "
     "GraphRAG by 2.0-8.1 F1 points on HotpotQA, MuSiQue, and 2WikiMultiHopQA."),
    ("paper-002", "Retrieval-as-Reasoning rests on three principles: "
     "Compilability (raw documents become structured, explicitly linked units "
     "in a persistent knowledge base), Composability (retrieval decomposes "
     "into atomic operations such as search, read, and link following that the "
     "agent composes through its reasoning loop), and Evolvability (the "
     "knowledge structure self-corrects over time rather than silently "
     "degrading)."),
    ("paper-003", "The Error Book is a persistent self-correction mechanism "
     "with a five-stage lifecycle: Discover (deterministic validators catch "
     "structural errors; LLM verification catches content errors), Attribute "
     "(trace each error to its root cause), Constrain (formalize the cause as "
     "a natural-language constraint rule), Inject (append open rules to the "
     "compilation prompt), and Verify & Close (re-validate and close fixed "
     "entries). Repair is two-layer: Code Auto-fix for structural errors and "
     "LLM Periodic Fix for semantic errors."),
]

# --- genuine compilation of the passages above ------------------------------
def page(path, title, summary, facts, related, sources, tags):
    return {"path": path, "title": title, "is_new": True, "aliases": [],
            "tags": tags, "summary": summary, "key_facts": facts,
            "related_pages": related, "related_sources": sources}


COMPILE_OUTPUTS = {
    "paper-001": {
        "digest": {"id": "paper-001", "summary": "LLM-Wiki 提出：编译文档为带双向链接的 Wiki，检索即推理，Error Book 自纠错，多跳基准上超 HippoRAG 2/LightRAG/GraphRAG 2.0-8.1 F1。"},
        "pages": [
            page("systems/LLM-Wiki", "LLM-Wiki",
                 "Agent-native retrieval system compiling documents into a self-evolving Wiki",
                 ["compiles documents into Wiki pages with bidirectional links",
                  "exposes search/read/link-following via tool-calling interfaces",
                  "introduces the Error Book for persistent self-correction",
                  "outperforms HippoRAG 2, LightRAG, GraphRAG by 2.0-8.1 F1"],
                 [["concepts/Retrieval-as-Reasoning", "paradigm it operationalizes"],
                  ["concepts/Error-Book", "its self-correction mechanism"],
                  ["systems/HippoRAG-2", "baseline it outperforms"],
                  ["systems/LightRAG", "strongest baseline"]],
                 [["sources/digests/paper-001", "abstract"]],
                 ["agent", "RAG", "wiki"]),
        ],
    },
    "paper-002": {
        "digest": {"id": "paper-002", "summary": "检索即推理三原则：可编译性、可组合性、可进化性。"},
        "pages": [
            page("concepts/Retrieval-as-Reasoning", "Retrieval as Reasoning",
                 "Paradigm where retrieval is a compositional reasoning activity",
                 ["principle 1: Compilability - documents become linked units in a persistent KB",
                  "principle 2: Composability - search/read/link-following composed by the agent loop",
                  "principle 3: Evolvability - the knowledge structure self-corrects over time"],
                 [["systems/LLM-Wiki", "system that operationalizes this paradigm"],
                  ["concepts/Error-Book", "realizes the Evolvability principle"]],
                 [["sources/digests/paper-002", "principles"]],
                 ["paradigm", "retrieval"]),
        ],
    },
    "paper-003": {
        "digest": {"id": "paper-003", "summary": "Error Book 五阶段生命周期与双层修复机制。"},
        "pages": [
            page("concepts/Error-Book", "Error Book",
                 "Persistent self-correction mechanism with a five-stage lifecycle",
                 ["five stages: Discover, Attribute, Constrain, Inject, Verify & Close",
                  "constraints are injected into future compilation prompts",
                  "Layer-1 Code Auto-fix repairs structural errors",
                  "Layer-2 LLM Periodic Fix repairs semantic errors"],
                 [["systems/LLM-Wiki", "component of LLM-Wiki"],
                  ["concepts/Retrieval-as-Reasoning", "realizes Evolvability"]],
                 [["sources/digests/paper-003", "error book design"]],
                 ["self-correction", "constraints"]),
            page("systems/HippoRAG-2", "HippoRAG 2",
                 "KG-triple based retrieval baseline",
                 ["exposes KG triples for retrieval",
                  "outperformed by LLM-Wiki by 2.0-8.1 F1 on multi-hop benchmarks"],
                 [["systems/LightRAG", "fellow graph-enhanced baseline"]],
                 [["sources/digests/paper-001", "mentioned as baseline"]],
                 ["baseline", "knowledge-graph"]),
            page("systems/LightRAG", "LightRAG",
                 "Entity/relation index based retrieval; strongest baseline overall",
                 ["relies on entity and relation indices",
                  "strongest baseline in the LLM-Wiki evaluation"],
                 [["systems/HippoRAG-2", "fellow graph-enhanced baseline"]],
                 [["sources/digests/paper-001", "mentioned as baseline"]],
                 ["baseline", "graph"]),
        ],
    },
}


class ScriptedLLM:
    """Plays the LLM role with pre-authored genuine outputs."""

    def __init__(self, compile_outputs, agent_actions=None):
        self.outputs = compile_outputs
        self.agent_actions = list(agent_actions or [])
        self.current_source = None

    def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "page-selection step" in prompt:
            return "[]"  # wiki starts empty; links are made via later passages
        if "page-compilation step" in prompt:
            for sid, _ in PASSAGES:
                if f"id: {sid}" in prompt:
                    return json.dumps(self.outputs[sid])
            raise AssertionError("unknown passage in compile prompt")
        if "verifying a Wiki page" in prompt:
            return "OK"  # facts are grounded (they are, we wrote them from the paper)
        if "systematic errors" in prompt:  # Error Book attribution
            return ("- id: 1\n  root_cause: linked to pages not yet created\n"
                    "  constraint_rule: ONLY link to pages in the indices or created in the same output")
        if self.agent_actions:
            return self.agent_actions.pop(0)
        return "OK"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    work = Path(tempfile.mkdtemp(prefix="llmwiki_demo_"))
    store = WikiStore(work / "wiki")
    book = ErrorBook(work / "error_book.yaml")

    # multi-hop question: LLM-Wiki -> Error Book -> its five stages
    question = "LLM-Wiki 依靠什么机制实现自我纠错？它包含哪些阶段？"
    llm = ScriptedLLM(COMPILE_OUTPUTS, agent_actions=[
        json.dumps({"tool": "wiki_search", "query": "LLM-Wiki self-correction"}),
        json.dumps({"tool": "wiki_read", "paths": ["systems/LLM-Wiki"]}),
        json.dumps({"tool": "wiki_read", "paths": ["concepts/Error-Book"]}),
        json.dumps({"tool": "answer",
                    "answer": "依靠 Error Book（错误记录本）：五个阶段为 "
                              "Discover（发现）、Attribute（归因）、Constrain（约束化）、"
                              "Inject（注入）、Verify & Close（验证关闭）；"
                              "修复分两层：Code Auto-fix 修结构错误，LLM Periodic Fix 修语义错误。",
                    "evidence": ["systems/LLM-Wiki", "concepts/Error-Book"]}),
    ])
    compiler = Compiler(store, llm, book)

    print("== 1. ingest (Algorithm 1) ==")
    written = compiler.compile_batch(PASSAGES)
    print(f"   {len(written)} page updates: {sorted(set(written))}")

    print("\n== 2. structural validation ==")
    errors = validators.structural_validate(store)
    print(f"   {len(errors)} error(s)" if errors else "   clean: 0 structural errors")
    for e in errors:
        print(f"   {e}")

    print("\n== 3. error book ==")
    print(f"   open entries: {len(book.open_entries())}")

    print(f"\n== 4. query: {question} ==")
    result = agent.run_agent(store, llm, question)
    for step in result["trace"]:
        print(f"   trace: {step}")
    print(f"\nANSWER: {result['answer']}")
    print(f"EVIDENCE: {result['evidence']}")

    keep = work
    print(f"\ndemo wiki kept at: {keep}")
    return 0 if not errors and "Error Book" in result["answer"] else 1


if __name__ == "__main__":
    sys.exit(main())
