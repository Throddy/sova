import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment" / "v2"))

import run as v2  # noqa: E402


def test_rules_on_gold_extraction_known_score_after_v3():
    """На правилах v2 правила на эталоне проходили 27 из 27. В v3 для дороги нужен участок
    и условия проезда, поэтому в Н1 две службы без участка дают «требует уточнения», а не противоречие."""
    rows = [r for c in v2.CASES for r in v2.score(c, v2.from_memory(v2.facts_from(c, v2.gold_claims(c)), v2.tm(c["T"])))]
    fails = [(r["case"], r["check"]["t"]) for r in rows if not r["ok"]]
    assert len(rows) == 27 and fails == [("Н1", "status")]


def test_baselines_scores_are_stable():
    got = {name: sum(r["ok"] for c in v2.CASES for r in v2.score(c, v2.baselines(c)[name]))
           for name in ("Последнее полученное", "Голосование источников")}
    assert got == {"Последнее полученное": 10, "Голосование источников": 15}


def test_gold_quotes_are_in_text():
    for c in v2.CASES:
        for r in c["reports"]:
            assert all(g["quote"] in r["text"] for g in r["gold"])
