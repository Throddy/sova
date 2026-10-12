"""Демонстратор СОВЫ.

python run_demo.py [--seeds 30]

Одно и то же ядро без изменений кода прогоняется на трёх видах ЧС.
data/           потоки донесений с разметкой
out/<профиль>/  сводки, проекты форм, запросы, разбор, сравнение с базовыми линиями
out/results.md  метрики на всех сценариях
Сторонних библиотек и сетевых запросов нет.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sova.baselines import last_received, majority_vote
from sova.evaluate import METRICS, aggregate, fmt, run_seed
from sova.extract import oracle_facts
from sova.memory import EvidenceMemory
from sova.profiles import PROFILES
from sova.profiles.base import END, generate, hhmm
from sova.reasoning import ask_queue
from sova.summary import render, render_forms, render_requests, render_review

ROOT = Path(__file__).parent
DEMO_SEEDS = {"fire": 1, "flood": 14, "heat": 47}
TIMES = (13 * 60, 15 * 60, 17 * 60 + 30)
BASELINE_ONLY = {"Последнее полученное": ("pairs", "false_alarm", "asks", "asks_clean", "recheck", "fake_confirm"),
                 "Голосование источников": ("pairs", "false_alarm", "asks", "asks_clean", "recheck"),
                 "Идеальная память": ("crit", "pairs", "false_alarm", "asks", "asks_clean", "recheck", "fake_confirm",
                                      "kept")}


def dump_corpus(p, seed: int) -> None:
    _, reports, pairs = generate(p, seed)
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    with open(data / f"{p.key}_seed{seed}.jsonl", "w", encoding="utf-8") as fh:
        for r in reports:
            fh.write(json.dumps({
                "id": r.rid, "source": r.source, "received": hhmm(r.t_rec), "text": r.text, "relay_of": r.relay_of,
                "gold": [{"attr": g.attr, "value": g.value, "observed": hhmm(g.t_obs), "planted": g.planted}
                         for g in r.facts]}, ensure_ascii=False) + "\n")
    md = [f"# Донесения. {p.title}, {p.place}, сценарий {seed}", "",
          "Время местное (МСК+6). В последней колонке разметка генератора, системе она не видна.", "",
          "| № | Получено | Источник | Текст | Разметка |", "| --- | --- | --- | --- | --- |"]
    for r in reports:
        marks = []
        for g in r.facts:
            m = f"{g.attr} = {p.ind(g.attr).fmt(g.value)} на {hhmm(g.t_obs)}"
            m += {"error": " (ошибка источника)", "echo": " (пересказ ошибки)"}.get(g.planted, "")
            marks.append(m)
        if r.relay_of:
            marks.append(f"пересказ {r.relay_of}")
        md.append(f"| {r.rid} | {hhmm(r.t_rec)} | {r.source} | {r.text} | {'; '.join(marks) or 'фактов нет'} |")
    md += ["", "Заложенные противоречия (ошибка и верное донесение): " + ", ".join(f"{a} и {b}" for a, b, _ in pairs)]
    (data / f"{p.key}_seed{seed}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def baseline_table(p, view: dict, T: int, title: str) -> str:
    out = [f"# {title}: сводка на {hhmm(T)}", "", "| Показатель | Значение | Донесение |", "| --- | --- | --- |"]
    for attr, ind in p.indicators.items():
        f = view[attr][0] if attr in view else None
        out.append(f"| {ind.name} | {ind.fmt(f.value) if f else 'нет сведений'} | {f.rid if f else ''} |")
    return "\n".join(out) + "\n"


def dump_outputs(p, seed: int) -> None:
    truth, reports, _ = generate(p, seed)
    facts = oracle_facts(reports, profile=p)
    mem = EvidenceMemory(p)
    for f in facts:
        mem.add(f)
    out = ROOT / "out" / p.key
    out.mkdir(parents=True, exist_ok=True)

    # решение штаба: первое расхождение по показателю из плана действий штаб снимает запросом
    staff_note = ""
    for T in range(11 * 60, END, 10):
        ask = next((a for a in ask_queue(mem, T) if a.state.status == "противоречие" and a.state.conflicts
                    and a.attr in {r.attr for r in p.rules}), None)
        if ask:
            bad = {(r.rid, g.attr) for r in reports for g in r.facts if g.planted}
            right = next(f for pair in ask.state.conflicts for f in pair if (f.rid, f.attr) not in bad)
            mem.decide(ask.attr, right.rid, "начальник штаба", T + 15, "уточнено запросом")
            staff_note = (f"В {hhmm(T + 15)} начальник штаба по итогам запроса принял {right.rid} "
                          f"({p.ind(ask.attr).name[0].lower() + p.ind(ask.attr).name[1:]}).")
            break

    for T in TIMES:
        tag = hhmm(T).replace(":", "")
        (out / f"svodka_{tag}.md").write_text(render(mem, T, T - 60), encoding="utf-8")
        (out / f"zaprosy_{tag}.md").write_text(render_requests(mem, T), encoding="utf-8")
        (out / f"b1_poslednee_{tag}.md").write_text(
            baseline_table(p, last_received(facts, T), T, "Последнее полученное"), encoding="utf-8")
        (out / f"b2_golosovanie_{tag}.md").write_text(
            baseline_table(p, majority_vote(mem, facts, T), T, "Голосование источников"), encoding="utf-8")
    for T in (13 * 60, END):
        (out / f"formy_{hhmm(T).replace(':', '')}.md").write_text(render_forms(mem, T), encoding="utf-8")
    review = render_review(mem, list(range(11 * 60, END + 1, 60)), END + 120, truth)
    if staff_note:
        review += "\n" + staff_note + "\n"
    (out / "razbor.md").write_text(review, encoding="utf-8")


def results_table(rows: list[dict], title: str) -> list[str]:
    agg = aggregate(rows)
    systems = list(agg)
    lines = [f"## {title}", "", "| Метрика | " + " | ".join(systems) + " |", "|" + " --- |" * (len(systems) + 1)]
    for key, name in METRICS.items():
        cells = ["—" if key in BASELINE_ONLY.get(s, ()) else fmt(key, agg[s][key]) for s in systems]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()

    for key, p in PROFILES.items():
        dump_corpus(p, DEMO_SEEDS[key])
        dump_outputs(p, DEMO_SEEDS[key])

    head = [f"# Сравнение: {len(PROFILES)} вида ЧС по {args.seeds} сценариев, по 8 сводок в каждом (11:00–18:00)", "",
            "В скобках 95% бутстрэп-интервал по сценариям. Факты взяты из разметки, кроме отдельно оговорённых "
            "прогонов с ошибками чтения, поэтому проверяется логика памяти, а не качество модели.", "",
            "«Идеальная память» заранее знает, какие донесения ошибочны, и берёт самое свежее верное. "
            "Её пропуски неустранимы: об изменении обстановки к моменту сводки ещё никто не сообщил.", ""]
    all_rows, text = [], list(head)
    for key, p in PROFILES.items():
        rows = [r for s in range(args.seeds) for r in run_seed(p, s)]
        all_rows += rows
        text += results_table(rows, p.title)
    text += results_table(all_rows, "Все три вида ЧС вместе")
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out" / "results.json").write_text(json.dumps(aggregate(all_rows), ensure_ascii=False, indent=1),
                                               encoding="utf-8")

    stress = ["## Устойчивость: все три вида ЧС вместе", "",
              "Шум источников увеличен втрое либо модель ошибается при чтении донесений: с заданной вероятностью "
              "пропускает факт, читает неверное значение и теряет часовой пояс.", "",
              "| Условия | Система | Критические ошибки на сводку | Опасность пропущена | Найдено противоречий | "
              "Противоречия без ошибки в источнике | «Требует уточнения» на сводку |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, noise, err in (("шум ×3", 3.0, 0.0), ("ошибки чтения 5%", 1.0, 0.05), ("ошибки чтения 10%", 1.0, 0.10)):
        rows = [r for p in PROFILES.values() for s in range(args.seeds) for r in run_seed(p, s, noise, err)]
        agg = aggregate(rows)
        for sname in ("Последнее полученное", "Голосование источников", "СОВА"):
            a = agg[sname]
            pr = "—" if sname != "СОВА" else fmt("pairs", a["pairs"])
            fa = "—" if sname != "СОВА" else fmt("false_alarm", a["false_alarm"])
            ak = "—" if sname != "СОВА" else fmt("asks", a["asks"])
            stress.append(f"| {label} | {sname} | {fmt('crit', a['crit'])} | {fmt('miss', a['miss'])} | {pr} | {fa} | {ak} |")
    text += stress + [""]
    (ROOT / "out" / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
