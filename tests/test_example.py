import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import run_example  # noqa: E402
from sova.reasoning import ask_queue, conclusions  # noqa: E402


def memory():
    mem = run_example.EvidenceMemory(run_example.FIRE)
    for f in run_example.to_facts(run_example.load(), upto=run_example.T):
        mem.add(f)
    return mem


def test_example_matches_expected_statuses():
    st = memory().state(run_example.T)
    assert {a: st[a].status for a in run_example.EXPECTED} == run_example.EXPECTED


def test_example_rules_and_questions_match_expected():
    mem = memory()
    rules = {c.rule.attr: c.value for c in conclusions(mem, run_example.T)}
    assert rules == run_example.EXPECTED_RULES
    urgent = [(a.attr, a.whom) for a in ask_queue(mem, run_example.T) if a.priority == 1]
    assert urgent == run_example.EXPECTED_ASKS


def test_relay_fields_match_review():
    """ЕДДС «по данным главы»: передал ЕДДС, ссылается на главу, наблюдатель и время не указаны."""
    f = next(f for f in run_example.to_facts(run_example.load()) if f.rid == "Д-6")
    assert (f.source, f.cited, f.observer, f.t_unknown, f.subject) == ("ЕДДС", "Глава поселения", "", True, "")
    assert f.quote in next(r["text"] for r in run_example.load() if r["id"] == "Д-6")


def test_every_observation_time_is_grounded_in_the_report_text():
    """Время наблюдения у каждого утверждения подкреплено фрагментом оригинала,
    а если времени в тексте нет, оно не придумано."""
    texts = {r["id"]: r["text"] for r in run_example.load()}
    for f in memory().known(run_example.T):
        if f.t_unknown:
            assert f.t_quote == ""
        else:
            assert f.t_quote and f.t_quote in texts[f.rid]
