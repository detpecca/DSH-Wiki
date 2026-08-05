"""Query-time agent: traversal, termination, and the wiki_read requirement."""

from llm_wiki import agent, schema
from llm_wiki.store import WikiStore

from conftest import FakeLLM, agent_script, answer_action, read_action, search_action


def _film_wiki(tmp_path):
    """Mini 2Wiki-style corpus: two films + two directors (paper Case 1)."""
    store = WikiStore(tmp_path / "wiki")

    def page(path, summary, facts, related):
        return schema.render_page(schema.Page(
            path=path, title=path, summary=summary, key_facts=facts,
            related_pages=related, related_sources=[("sources/digests/d", "n")],
        ), "2026-08-04")

    store.write("media/The-Gamecock", page(
        "media/The-Gamecock", "1974 film", ["directed by Pasquale Festa Campanile"],
        [("people/Pasquale-Festa-Campanile", "director")]))
    store.write("media/Monster-A-Go-Go", page(
        "media/Monster-A-Go-Go", "1965 film", ["directed by Herschell Gordon Lewis"],
        [("people/Herschell-Gordon-Lewis", "director")]))
    store.write("people/Pasquale-Festa-Campanile", page(
        "people/Pasquale-Festa-Campanile", "Italian director", ["born 28 July 1927"], []))
    store.write("people/Herschell-Gordon-Lewis", page(
        "people/Herschell-Gordon-Lewis", "American director", ["born 15 June 1926"], []))
    store.rebuild_all_indices()
    return store


def test_bridge_comparison_via_link_following(tmp_path):
    store = _film_wiki(tmp_path)
    llm = FakeLLM(agent_script(
        search_action("The Gamecock Monster A Go-Go"),
        read_action("media/The-Gamecock", "media/Monster-A-Go-Go"),
        read_action("people/Pasquale-Festa-Campanile", "people/Herschell-Gordon-Lewis"),
        answer_action("Monster A Go-Go (Lewis 1926 is older than Campanile 1927)",
                      ["people/Herschell-Gordon-Lewis", "people/Pasquale-Festa-Campanile"]),
    ))
    result = agent.run_agent(store, llm, "Which film has the older director?")
    assert "Monster A Go-Go" in result["answer"]
    tools = [t["tool"] for t in result["trace"]]
    assert tools == ["wiki_search", "wiki_read", "wiki_read", "answer"]


def test_answer_requires_wiki_read(tmp_path):
    store = _film_wiki(tmp_path)
    llm = FakeLLM(agent_script(
        answer_action("premature answer"),      # rejected: no wiki_read yet
        read_action("media/The-Gamecock"),
        answer_action("grounded answer", ["media/The-Gamecock"]),
    ))
    result = agent.run_agent(store, llm, "q")
    assert result["answer"] == "grounded answer"


def test_patience_terminates_on_empty_searches(tmp_path):
    store = _film_wiki(tmp_path)
    llm = FakeLLM(agent_script(*[search_action("zzzz nothing")] * 5))
    result = agent.run_agent(store, llm, "unanswerable", patience=3)
    assert "terminated" in result["answer"]
    searches = [t for t in result["trace"] if t["tool"] == "wiki_search"]
    assert len(searches) == 3  # stopped at patience threshold


def test_budget_terminates(tmp_path):
    store = _film_wiki(tmp_path)
    llm = FakeLLM(agent_script(*[read_action("media/The-Gamecock")] * 20))
    result = agent.run_agent(store, llm, "q", t_max=5)
    assert "terminated" in result["answer"]
    assert len([t for t in result["trace"] if t["tool"] == "wiki_read"]) == 5
