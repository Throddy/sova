"""Мини-эксперимент: семь вымышленных кейсов про пожар у посёлка.

python experiment/run.py                       # облачные ответы из answers/cloud_*.json
python experiment/run.py --answers ollama_qwen3_8b   # ответы локальной модели (см. run_llm.py)

Ожидаемые результаты лежат в cases.json и записаны до запуска любой системы.
При воспроизведении каждая система видит только донесения, полученные к моменту
сводки. Сравниваются:
  1. последнее полученное,
  2. голосование источников,
  3. простая LLM: все донесения и правила штаба одним запросом,
  4. СОВА: та же модель только извлекает факты, статусы считают правила,
  5. СОВА с ручной разметкой фактов: проверка правил отдельно от извлечения.
Результат: experiment/RESULTS_AUTO.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from sova.baselines import last_received, majority_vote  # noqa: E402
from sova.extract import find_relay, parse_value  # noqa: E402
from sova.memory import EvidenceMemory, Fact  # noqa: E402
from sova.profiles import FIRE  # noqa: E402
from sova.reasoning import ask_queue, conclusions  # noqa: E402
from sova.timeparse import MSK_SHIFT, parse_obs_time  # noqa: E402

ATTRS = ["distance_km", "area_ha", "wind", "road", "people_in_zone", "evacuated", "personnel", "equipment",
         "casualties"]
EVAC, ROAD = FIRE.rules[0], FIRE.rules[1]


def tm(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def rid(x) -> str | None:
    if x is None:
        return None
    x = str(x).strip()
    return x if x.startswith("Д-") else f"Д-{x}"


def received(case: dict) -> list[dict]:
    """Только то, что дошло до штаба к моменту сводки."""
    return [r for r in case["reports"] if tm(r["received"]) <= tm(case["T"])]


# ---------- извлечение фактов

def facts_gold(case: dict) -> list[Fact]:
    out = []
    for r in received(case):
        t_rec = tm(r["received"])
        t_obs, est = parse_obs_time(r["text"], t_rec)
        origin = find_relay(r["text"]) or r["source"]
        for g in r["facts"]:
            v = tuple(g["value"]) if isinstance(g["value"], list) else g["value"]
            out.append(Fact(len(out), r["id"], r["source"], g["attr"], v, t_obs, t_rec, est, origin))
    return out


def facts_llm(case: dict, extracted: dict) -> list[Fact]:
    out = []
    for r in received(case):
        e = extracted.get(f"{case['id']}/{r['id']}", {})
        t_rec = tm(r["received"])
        if e.get("time"):
            t_obs, est = (tm(e["time"]) + (MSK_SHIFT if e.get("msk") else 0)) % (24 * 60), False
        else:
            t_obs, est = t_rec - 10, True
        origin = e.get("relay_from") or r["source"]
        for f in e.get("facts", []):
            if f.get("attr") not in FIRE.indicators:
                continue
            v = parse_value(FIRE, f["attr"], str(f["value"]))
            if v is not None:
                out.append(Fact(len(out), r["id"], r["source"], f["attr"], v, t_obs, t_rec, est, origin))
    return out


# ---------- ответы систем в общем виде

def decision_word(rule, value) -> str:
    if value is None:
        return "уточнить"
    danger = rule.is_danger(value)
    if rule is EVAC:
        return "нужна" if danger else "не нужна"
    return "нельзя" if danger else "можно"


def from_memory(facts: list[Fact], T: int) -> dict:
    mem = EvidenceMemory(FIRE)
    for f in facts:
        mem.add(f)
    st = mem.state(T)
    ind = {}
    for a in ATTRS:
        s = st[a]
        val = None
        if s.current and s.status != "оспорен":
            val = s.current[-1].value
        ind[a] = {"value": val, "status": s.status, "sources": [f.rid for f in s.current]}
    words = {EVAC.name: {EVAC.danger: "нужна", EVAC.safe: "не нужна", EVAC.watch: "уточнить"},
             ROAD.name: {ROAD.danger: "нельзя", ROAD.safe: "можно", ROAD.watch: "уточнить"}}
    dec = {}
    for c in conclusions(mem, T):
        dec["evacuation" if c.rule is EVAC else "road"] = words[c.rule.name][c.value]
    asks = [{"attr": a.attr, "kind": a.kind, "rid": a.fact.rid if a.fact else None} for a in ask_queue(mem, T)]
    return {"indicators": ind, "decisions": dec, "asks": asks}


def from_view(view: dict) -> dict:
    """Простые способы: показывают одно значение, статусов и запросов у них нет."""
    ind = {a: {"value": view[a][0].value if a in view else None, "status": None,
               "sources": [view[a][0].rid] if a in view else []} for a in ATTRS}
    dec = {"evacuation": decision_word(EVAC, view["distance_km"][0].value if "distance_km" in view else None),
           "road": decision_word(ROAD, view["road"][0].value if "road" in view else None)}
    return {"indicators": ind, "decisions": dec, "asks": None}


def from_llm(ans: dict) -> dict:
    ind = {}
    for a in ATTRS:
        x = ans.get("indicators", {}).get(a, {}) or {}
        raw = x.get("value")
        val = parse_value(FIRE, a, str(raw)) if raw not in (None, "", "null") else None
        ind[a] = {"value": val, "status": x.get("status") or None,
                  "sources": [rid(s) for s in x.get("sources", []) or []]}
    asks = [{"attr": q.get("attr"), "kind": q.get("kind"), "rid": rid(q.get("rid"))} for q in ans.get("asks", [])]
    return {"indicators": ind, "decisions": ans.get("decisions", {}), "asks": asks}


# ---------- оценка

def norm(s):
    return None if s is None else str(s).strip().lower().replace("ё", "е")


def same_value(a, b) -> bool:
    if a is None or b is None:
        return False
    if isinstance(b, list) or isinstance(b, tuple):
        return tuple(a) == tuple(b) if isinstance(a, (list, tuple)) else False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 0.05
    return norm(a) == norm(b)


def score(case: dict, out: dict) -> list[dict]:
    rows = []
    for ch in case["checks"]:
        t, got, ok = ch["t"], None, None
        if t == "status":
            got = out["indicators"][ch["attr"]]["status"]
            ok = None if got is None else norm(got) == norm(ch["want"])
        elif t == "value":
            got = out["indicators"][ch["attr"]]["value"]
            ok = same_value(got, ch["want"])
        elif t == "decision":
            got = out["decisions"].get(ch["rule"])
            ok = norm(got) == norm(ch["want"])
        elif t == "ask_time":
            if out["asks"] is not None:
                hits = [q for q in out["asks"] if q["attr"] == ch["attr"] and q["kind"] == "time" and q["rid"] == ch["rid"]]
                got, ok = ("есть" if hits else "нет"), bool(hits)
        elif t == "no_conflict":
            flagged = [a for a, x in out["indicators"].items() if norm(x["status"]) == "оспорен"]
            if all(x["status"] is not None for x in out["indicators"].values()):
                got, ok = (", ".join(flagged) or "нет"), not flagged
        rows.append({"case": case["id"], "check": ch, "got": got, "ok": ok})
    return rows


def citations(case: dict, out: dict) -> tuple[int, int]:
    """Сколько ссылок указывают на донесение, где действительно есть этот показатель."""
    gold = {r["id"]: {g["attr"] for g in r["facts"]} for r in received(case)}
    good = total = 0
    for a, x in out["indicators"].items():
        if x["value"] is None and norm(x["status"]) not in ("оспорен", "устарел"):
            continue
        for s in x["sources"]:
            total += 1
            good += a in gold.get(s, set())
    return good, total


def label(ch: dict) -> str:
    names = {"distance_km": "расстояние", "area_ha": "площадь", "wind": "ветер", "road": "дорога",
             "people_in_zone": "население", "evacuated": "эвакуировано", "personnel": "личный состав",
             "evacuation": "эвакуация", "casualties": "пострадавшие", "equipment": "техника"}
    t = ch["t"]
    if t == "status":
        return f"статус: {names[ch['attr']]}"
    if t == "value":
        return f"значение: {names[ch['attr']]}"
    if t == "decision":
        return f"решение: {'эвакуация' if ch['rule'] == 'evacuation' else 'вывоз по дороге'}"
    if t == "ask_time":
        return f"запрос времени {ch['rid']}"
    return "нет ложных противоречий"


def want_text(ch: dict) -> str:
    if ch["t"] == "value" and isinstance(ch["want"], list):
        return f"{ch['want'][0]} {ch['want'][1]}"
    return {"ask_time": "есть", "no_conflict": "нет"}.get(ch["t"], str(ch.get("want", "")))


def got_text(row: dict) -> str:
    g = row["got"]
    if row["ok"] is None:
        return "—"
    if isinstance(g, tuple):
        g = f"{g[0]} {g[1]}"
    if isinstance(g, float):
        g = f"{g:g}".replace(".", ",")
    mark = "" if row["ok"] else " ✗"
    return f"{g}{mark}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="cloud", help="префикс файлов в answers/: <префикс>_baseline.json и _extract.json")
    args = ap.parse_args()
    cases = json.load(open(HERE / "cases.json", encoding="utf-8"))["cases"]
    base_path, ext_path = HERE / "answers" / f"{args.answers}_baseline.json", HERE / "answers" / f"{args.answers}_extract.json"
    baseline = json.load(open(base_path, encoding="utf-8")) if base_path.exists() else {}
    extracted = json.load(open(ext_path, encoding="utf-8")) if ext_path.exists() else {}
    model = "облачная модель" if args.answers == "cloud" else args.answers

    systems = ["Последнее полученное", "Голосование источников", f"Простая LLM ({model})",
               f"СОВА, факты извлекает {model}", "СОВА, факты из ручной разметки"]
    results = {s: [] for s in systems}
    cites = {s: [0, 0] for s in systems}
    for case in cases:
        T = tm(case["T"])
        gold = facts_gold(case)
        mem = EvidenceMemory(FIRE)
        for f in gold:
            mem.add(f)
        outs = {
            systems[0]: from_view(last_received(gold, T)),
            systems[1]: from_view(majority_vote(mem, gold, T)),
            systems[2]: from_llm(baseline[case["id"]]) if case["id"] in baseline else None,
            systems[3]: from_memory(facts_llm(case, extracted), T) if extracted else None,
            systems[4]: from_memory(gold, T),
        }
        for s, out in outs.items():
            if out is None:
                continue
            results[s] += score(case, out)
            g, n = citations(case, out)
            cites[s][0] += g
            cites[s][1] += n

    def frac(rows, kind=None, dangerous=False):
        rs = [r for r in rows if (kind is None or r["check"]["t"] in kind) and r["ok"] is not None]
        if dangerous:
            rs = [r for r in rows if r["check"]["t"] == "decision" and r["check"]["rule"] == "evacuation"
                  and r["check"]["want"] != "не нужна" and r["ok"] is not None]
            bad = sum(1 for r in rs if norm(r["got"]) == "не нужна")
            return f"{bad} из {len(rs)}"
        return f"{sum(r['ok'] for r in rs)} из {len(rs)}" if rs else "—"

    def false_alarms(rows):
        rs = [r for r in rows if r["ok"] is not None and (
            (r["check"]["t"] == "status" and norm(r["got"]) == "оспорен" and r["check"]["want"] != "оспорен")
            or (r["check"]["t"] == "no_conflict" and not r["ok"]))]
        return str(len(rs)) if any(r["ok"] is not None and r["check"]["t"] == "status" for r in rows) else "—"

    lines = ["# Мини-эксперимент: результаты", "",
             f"7 кейсов, 26 донесений, 33 проверки, ожидаемые ответы записаны заранее (cases.json). "
             f"Модель: {model}. Каждая система видит только донесения, полученные к моменту сводки.", "",
             "| Система | Всё вместе | Статусы | Значения | Решения | Опасный пропуск: «эвакуация не нужна», а ждали иное | "
             "Запрос времени | Ложные противоречия | Верные ссылки |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in systems:
        rows = results[s]
        if not rows:
            lines.append(f"| {s} | нет ответов | | | | | | | |")
            continue
        g, n = cites[s]
        lines.append(f"| {s} | {frac(rows)} | {frac(rows, {'status', 'no_conflict'})} | {frac(rows, {'value'})} | "
                     f"{frac(rows, {'decision'})} | {frac(rows, dangerous=True)} | {frac(rows, {'ask_time'})} | "
                     f"{false_alarms(rows)} | {g} из {n} |")
    lines += ["", "«—» значит, что у системы нет такой функции: простые способы не ставят статусов и не задают вопросов.", "",
              "## Все проверки", "", "| Кейс | Проверка | Ждали | " + " | ".join(systems) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in systems) + " |"]
    for i, row in enumerate(results[systems[4]]):
        cells = [got_text(results[s][i]) if results[s] else "" for s in systems]
        lines.append(f"| {row['case']} | {label(row['check'])} | {want_text(row['check'])} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    suffix = "" if args.answers == "cloud" else f"_{args.answers}"
    (HERE / f"RESULTS_AUTO{suffix}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
