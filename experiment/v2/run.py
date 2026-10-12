"""Эксперимент v2: проверка поведения на отложенных авторских сценариях Н1–Н8.

python experiment/v2/run.py --prompts      пишет подсказки для модели в prompts/
python experiment/v2/run.py                считает результаты по ответам в answers/, пишет RESULTS_AUTO.md

Варианты:
- последнее полученное и голосование источников на эталонном извлечении;
- простая LLM: все донесения и правила словами в одном запросе (answers/summary_runN.json);
- СОВА на эталонном извлечении: проверяет только правила;
- СОВА с извлечением моделью (answers/extract_runN.json): модель заполняет поля утверждений, статус считают правила.

Ошибки считаются по типам: извлечение (значение, время, источник, участок, цитата), статусы,
вопросы, правила сценария, ссылки утверждений сводки на донесения.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from prompts import ATTRS, SUBJECTS, extract_prompt, summary_prompt  # noqa: E402
from sova import forms  # noqa: E402
from sova.extract import parse_value  # noqa: E402
from sova.memory import EvidenceMemory, Fact  # noqa: E402
from sova.profiles import FIRE  # noqa: E402
from sova.reasoning import ask_queue, conclusions  # noqa: E402

CASES = json.load(open(HERE / "cases.json", encoding="utf-8"))["cases"]
RULE_ATTR = {"evacuation": "distance_km", "road": "road"}
# облачные модели через субагентов среды; Haiku добавлен после того, как был виден результат Sonnet
MODELS = {"Sonnet": "", "Haiku": "haiku_"}


def tm(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def received(case: dict) -> list[dict]:
    return [r for r in case["reports"] if tm(r["received"]) <= tm(case["T"])]


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower().replace("ё", "е")).strip()


def to_fact(i: int, r: dict, c: dict) -> Fact | None:
    value = c["value"]
    if isinstance(value, str):
        value = parse_value(FIRE, c["attr"], value)
    if value is None or c["attr"] not in ATTRS:
        return None
    t = c.get("time")
    unknown = not t or not re.fullmatch(r"\d{1,2}:\d{2}", str(t))
    subject = c.get("subject") or ""
    return Fact(i, r["id"], r["source"], c["attr"], value, tm(r["received"]) if unknown else tm(t), tm(r["received"]),
                t_unknown=unknown, cited=c.get("cited") or "", observer=c.get("observer") or "",
                subject=subject if subject in SUBJECTS else "", quote=c.get("quote") or "")


def facts_from(case: dict, claims: dict[str, list[dict]]) -> list[Fact]:
    out = []
    for r in received(case):
        for c in claims.get(r["id"], []):
            f = to_fact(len(out), r, c)
            if f:
                out.append(f)
    return out


def gold_claims(case: dict) -> dict[str, list[dict]]:
    return {r["id"]: r["gold"] for r in case["reports"]}


# Словарь статусов v2. После v2 статусы переименованы и актуальность вынесена отдельно
# (v3: «согласуется» только почти на одно время, «не подтверждено»). Числа v2 получены
# на правилах v2 (RESULTS_AUTO.md); здесь новые статусы переводятся в словарь v2 для повторного прогона.
V2_NAMES = {"согласуется": "подтверждён", "не подтверждено": "не подтверждён"}


def v2_status(s) -> str:
    if not s.fresh and s.status in ("согласуется", "не подтверждено"):
        return "устарел"
    return V2_NAMES.get(s.status, s.status)


# выход в общем виде: показатели, правила, вопросы
def from_memory(facts: list[Fact], T: int) -> dict:
    mem = EvidenceMemory(FIRE)
    for f in facts:
        mem.add(f)
    st = mem.state(T)
    out = {"indicators": {}, "rules": {}, "questions": []}
    for attr in ATTRS:
        s = st[attr]
        out["indicators"][attr] = {
            "value": forms._value(mem, s) if s.shown else None, "status": v2_status(s),
            "basis": [{"rid": f.rid, "value": FIRE.ind(attr).fmt(f.value), "quote": f.quote} for f in s.shown]}
    for c in conclusions(mem, T):
        key = next(k for k, a in RULE_ATTR.items() if a == c.rule.attr)
        out["rules"][key] = {"result": c.value, "inputs_reliable": c.recheck is None}
    out["questions"] = [{"attr": a.attr, "to": a.whom, "text": a.problem} for a in ask_queue(mem, T)
                        if a.priority == 1 and a.attr in ATTRS]
    return out


def from_single(view: dict, T: int, votes: bool) -> dict:
    """Простые способы показывают одно значение без статусов и вопросов."""
    out = {"indicators": {}, "rules": {}, "questions": []}
    for attr in ATTRS:
        if attr not in view:
            out["indicators"][attr] = {"value": None, "status": "нет сведений", "basis": []}
            continue
        f, n = view[attr]
        status = "подтверждён" if votes and n >= 2 else "не подтверждён"
        out["indicators"][attr] = {"value": FIRE.ind(attr).fmt(f.value), "status": status,
                                   "basis": [{"rid": f.rid, "value": FIRE.ind(attr).fmt(f.value), "quote": f.quote}]}
    for rule in FIRE.rules:
        key = next(k for k, a in RULE_ATTR.items() if a == rule.attr)
        if rule.attr in view:
            res = rule.danger if rule.is_danger(view[rule.attr][0].value) else rule.safe
        else:
            res = rule.watch
        out["rules"][key] = {"result": res, "inputs_reliable": True}
    return out


def baselines(case: dict) -> dict[str, dict]:
    from sova.baselines import last_received, majority_vote
    facts = facts_from(case, gold_claims(case))
    T = tm(case["T"])
    mem = EvidenceMemory(FIRE)
    for f in facts:
        mem.add(f)
    return {"Последнее полученное": from_single(last_received(facts, T), T, False),
            "Голосование источников": from_single(majority_vote(mem, facts, T), T, True)}


# проверки
def numbers(s) -> list[float]:
    return [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", str(s or ""))]


def score(case: dict, out: dict) -> list[dict]:
    rows = []
    for ch in case["checks"]:
        ind = out.get("indicators", {}).get(ch.get("attr"), {}) or {}
        t = ch["t"]
        if t == "status":
            got = norm(ind.get("status"))
            ok = got in [norm(w) for w in ch["want"]]
        elif t == "value_has":
            got = ind.get("value")
            ok = all((any(abs(n - w) < 1e-6 for n in numbers(got)) if isinstance(w, (int, float))
                      else norm(w) in norm(got)) for w in ch["want"])
        elif t == "basis_excludes":
            got = [b.get("rid") for b in ind.get("basis", []) or []]
            ok = ch["rid"] not in got
        elif t == "ask":
            qs = [q for q in out.get("questions", []) or [] if q.get("attr") == ch["attr"]]
            got = "; ".join(str(q.get("to")) for q in qs)
            ok = all(norm(w) in norm(got) for w in ch["whom"])
        elif t == "rule":
            got = (out.get("rules", {}).get(ch["rule"]) or {}).get("result")
            ok = norm(got) == norm(ch["want"])
        elif t == "rule_flag":
            r = out.get("rules", {}).get(ch["rule"]) or {}
            got = r.get("inputs_reliable")
            ok = got is False or norm(r.get("result")) == norm("нельзя определить до уточнения")
        else:
            raise ValueError(t)
        rows.append({"case": case["id"], "check": ch, "got": got, "ok": bool(ok)})
    return rows


def false_contradictions(case: dict, out: dict) -> int:
    """Показатель назван противоречием, хотя по замыслу сценария это не так."""
    wanted = {ch["attr"] for ch in case["checks"] if ch["t"] == "status" and "противоречие" in ch["want"]}
    return sum(1 for a, x in (out.get("indicators") or {}).items()
               if norm((x or {}).get("status")) == "противоречие" and a not in wanted)


def claim_refs(case: dict, out: dict) -> tuple[int, int]:
    """Сколько утверждений сводки (элементов basis) ссылаются на донесение, где это
    действительно сказано, с цитатой, которая есть в тексте. Нечисловые утверждения тоже считаются."""
    texts = {r["id"]: r["text"] for r in case["reports"]}
    gold = [((r["id"], g["attr"]), g) for r in case["reports"] for g in r["gold"]]
    ok = total = 0
    for attr, x in (out.get("indicators") or {}).items():
        for b in (x or {}).get("basis", []) or []:
            total += 1
            rid, quote = b.get("rid"), norm(b.get("quote"))
            golds = [g for (r, a), g in gold if r == rid and a == attr]
            if not golds or not quote or quote not in norm(texts.get(rid)):
                continue
            if any(value_match(attr, b.get("value"), g["value"]) for g in golds):
                ok += 1
    return ok, total


def value_match(attr: str, got, want) -> bool:
    if isinstance(want, str):
        return norm(want)[:5] in norm(got)
    return any(abs(n - want) < 1e-6 for n in numbers(got))


def extraction(case: dict, claims: dict[str, list[dict]]) -> dict[str, list[bool]]:
    """Сравнение извлечения с эталоном по полям. Наблюдатель нормализуется:
    без ссылки на другого наблюдатель это сам отправитель."""
    res: dict[str, list[bool]] = {k: [] for k in ("found", "value", "time", "cited", "observer", "subject", "quote")}
    extra = 0
    for r in case["reports"]:
        got = list(claims.get(r["id"], []))
        def obs(c, src=r["source"]):
            return norm(c.get("observer")) or ("" if c.get("cited") else norm(src))
        for g in r["gold"]:
            cand = [c for c in got if c.get("attr") == g["attr"]]
            cand.sort(key=lambda c: not value_match(g["attr"], c.get("value"), g["value"]))
            res["found"].append(bool(cand))
            if not cand:
                continue
            c = cand[0]
            got.remove(c)
            res["value"].append(value_match(g["attr"], c.get("value"), g["value"]))
            t = c.get("time")
            res["time"].append((None if t in (None, "", "null") else str(t).zfill(5)) == g["time"])
            res["cited"].append(norm(c.get("cited")) == norm(g["cited"]))
            res["observer"].append(obs(c) == obs(g))
            if g["attr"] == "distance_km":
                res["subject"].append(norm(c.get("subject")) == norm(g["subject"]))
            q = norm(c.get("quote"))
            res["quote"].append(bool(q) and q in norm(r["text"]) and value_match(g["attr"], c.get("quote"), g["value"]))
        extra += len(got)
    res["extra"] = [extra]
    return res


def load_runs(prefix: str) -> list[dict]:
    return [json.load(open(p, encoding="utf-8")) for p in sorted((HERE / "answers").glob(f"{prefix}_run*.json"))]


def write_prompts() -> None:
    d = HERE / "prompts"
    d.mkdir(exist_ok=True)
    summ = "\n\n".join(f"=== {c['id']} ===\n{summary_prompt(c, received(c))}" for c in CASES)
    (d / "summary.txt").write_text(summ, encoding="utf-8")
    ext = "\n\n".join(f"=== {c['id']} {r['id']} ===\n{extract_prompt(r)}" for c in CASES for r in received(c))
    (d / "extract.txt").write_text(ext, encoding="utf-8")
    print(f"{len(CASES)} сводок, {sum(len(received(c)) for c in CASES)} донесений")


def systems() -> dict[str, list[dict[str, dict]]]:
    """Имя варианта -> список прогонов, прогон: {сценарий: выход}."""
    out: dict[str, list[dict[str, dict]]] = {}
    for name in ("Последнее полученное", "Голосование источников"):
        out[name] = [{c["id"]: baselines(c)[name] for c in CASES}]
    out["СОВА, эталонное извлечение"] = [{c["id"]: from_memory(facts_from(c, gold_claims(c)), tm(c["T"])) for c in CASES}]
    for model, prefix in MODELS.items():
        out[f"{model} без отдельной памяти"] = [{c["id"]: run.get(c["id"], {}) for c in CASES} for run in load_runs(prefix + "summary")]
        out[f"СОВА с извлечением через {model}"] = [
            {c["id"]: from_memory(facts_from(c, {rid: (x or {}).get("claims", []) for rid, x in run.get(c["id"], {}).items()}),
                                  tm(c["T"])) for c in CASES} for run in load_runs(prefix + "extract")]
    return {k: v for k, v in out.items() if v}


KINDS = {"status": "статусы", "value_has": "значения", "basis_excludes": "значения", "ask": "вопросы",
         "rule": "правила сценария", "rule_flag": "правила сценария"}


def spread(vals: list[float]) -> str:
    if len(vals) == 1:
        return f"{vals[0]:g}"
    return f"{statistics.fmean(vals):.1f} ({min(vals):g}–{max(vals):g})".replace(".", ",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", action="store_true")
    ap.add_argument("--v2", action="store_true", help="перезаписать RESULTS_AUTO.md (только на правилах v2)")
    args = ap.parse_args()
    if args.prompts:
        write_prompts()
        return
    sysmap = systems()
    total = sum(len(c["checks"]) for c in CASES)
    kinds = list(dict.fromkeys(KINDS.values()))
    lines = ["# Эксперимент v2: результаты (пишется скриптом)", "",
             f"Сценариев {len(CASES)}, проверок {total}. Для вариантов с моделью через «;» указаны прогоны, "
             "в скобках разброс. Подробный разбор в RESULTS.md.", "",
             "| Вариант | Прогонов | Верных проверок из " + str(total) + " | " + " | ".join(kinds)
             + " | Ложные противоречия | Утверждения сводки с верной ссылкой |",
             "| --- | --- | --- | " + " --- |" * len(kinds) + " --- | --- |"]
    per_case: dict[str, dict[str, list[int]]] = {}
    fails: dict[str, list[str]] = {}
    for name, runs in sysmap.items():
        okc, by_kind, fc, refs = [], {k: [] for k in kinds}, [], []
        for run in runs:
            rows = [r for c in CASES for r in score(c, run.get(c["id"], {}))]
            okc.append(sum(r["ok"] for r in rows))
            for k in kinds:
                rk = [r for r in rows if KINDS[r["check"]["t"]] == k]
                by_kind[k].append(sum(r["ok"] for r in rk))
            fc.append(sum(false_contradictions(c, run.get(c["id"], {})) for c in CASES))
            o = t = 0
            for c in CASES:
                a, b = claim_refs(c, run.get(c["id"], {}))
                o, t = o + a, t + b
            refs.append(100 * o / t if t else 0.0)
            for c in CASES:
                rc = [r for r in rows if r["case"] == c["id"]]
                per_case.setdefault(c["id"], {}).setdefault(name, []).append(sum(r["ok"] for r in rc))
            for r in rows:
                if not r["ok"]:
                    fails.setdefault(name, []).append(f"{r['case']} {r['check']['t']} {r['check'].get('attr') or r['check'].get('rule')}: "
                                                      f"получено {r['got']!r}")
        nk = {k: sum(1 for c in CASES for ch in c["checks"] if KINDS[ch["t"]] == k) for k in kinds}
        lines.append(f"| {name} | {len(runs)} | {spread(okc)} | "
                     + " | ".join(f"{spread(by_kind[k])} из {nk[k]}" for k in kinds)
                     + f" | {spread(fc)} | {spread([round(x) for x in refs])}% |")
    lines += ["", "## По сценариям, верных проверок", "",
              "| Сценарий | Проверок | " + " | ".join(sysmap) + " |", "| --- | --- |" + " --- |" * len(sysmap)]
    for c in CASES:
        lines.append(f"| {c['id']} {c['title']} | {len(c['checks'])} | "
                     + " | ".join("; ".join(str(x) for x in per_case[c["id"]][n]) for n in sysmap) + " |")
    names = {"found": "утверждение найдено", "value": "значение", "time": "время наблюдения (или «не указано»)",
             "cited": "на кого ссылается", "observer": "кто наблюдал", "subject": "участок (для расстояния)",
             "quote": "цитата есть в тексте и содержит значение", "extra": "лишних утверждений, штук"}
    for model, prefix in MODELS.items():
        ext_runs = load_runs(prefix + "extract")
        if not ext_runs:
            continue
        lines += ["", f"## Извлечение по полям, {model}", "",
                  "Верных полей по эталонным утверждениям, по прогонам.", "",
                  "| Поле | " + " | ".join(f"прогон {i + 1}" for i in range(len(ext_runs))) + " |",
                  "| --- |" + " --- |" * len(ext_runs)]
        per_run = []
        for run in ext_runs:
            agg: dict[str, list[bool]] = {}
            for c in CASES:
                claims = {rid: (x or {}).get("claims", []) for rid, x in run.get(c["id"], {}).items()}
                for k, v in extraction(c, claims).items():
                    agg.setdefault(k, []).extend(v)
            per_run.append(agg)
        for k, label in names.items():
            cells = [str(sum(a.get(k, []))) if k == "extra" else f"{sum(a.get(k, []))} из {len(a.get(k, []))}"
                     for a in per_run]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Ошибки", ""]
    for name, fl in fails.items():
        lines += [f"### {name}", ""] + [f"- {x}" for x in fl] + [""]
    # числа v2 лежат в RESULTS_AUTO.md и получены на правилах v2;
    # на текущем коде пишем отдельный файл, чтобы их не затереть
    out = "RESULTS_AUTO.md" if args.v2 else "RESULTS_RERUN.md"
    if not args.v2:
        lines[0] = "# Повторный прогон на текущем коде (правила v3), не числа v2"
    (HERE / out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
