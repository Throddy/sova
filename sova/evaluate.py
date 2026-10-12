"""Метрики: СОВА против базовых линий на многих сценариях каждого профиля.

Сводка собирается каждый час с 11:00 до 18:00. Для каждой сводки считаем,
что в ней показано без пометки, хотя должно было быть помечено, и верно ли
решение по каждому правилу плана действий.
"""
from __future__ import annotations

import random
import statistics

from .baselines import last_received, majority_vote
from .extract import noisy, oracle_facts
from .memory import WINDOW, EvidenceMemory, Fact
from .profiles.base import END, Profile, generate
from .reasoning import conclusions

CHECKPOINTS = list(range(11 * 60, END + 1, 60))
FLAGGED = {"противоречие", "требует уточнения"}


def _critical_errors(p: Profile, shown: dict[str, list[Fact]], flags: dict[str, set], known: list[Fact],
                     bad: set, T: int) -> int:
    """Критический показатель показан без пометки, хотя значение устарело,
    вытеснено более свежим наблюдением или взято из ошибочного донесения."""
    n = 0
    for attr, ind in p.indicators.items():
        items = shown.get(attr)
        if not ind.critical or not items:
            continue
        fl = flags.get(attr, set())
        best = max(f.t_obs for f in known if f.attr == attr)
        newest = max(f.t_obs for f in items)
        stale = newest < best - WINDOW
        expired = T - newest > ind.ttl and "устарел" not in fl
        wrong = any((f.rid, f.attr) in bad for f in items) and not (fl & FLAGGED)
        n += bool(stale or expired or wrong)
    return n


def run_seed(p: Profile, seed: int, noise: float = 1.0, extract_err: float = 0.0) -> list[dict]:
    truth, reports, pairs = generate(p, seed, noise)
    bad = {(r.rid, g.attr) for r in reports for g in r.facts if g.planted}
    planted_attr = {r.rid: g.attr for r in reports for g in r.facts if g.planted == "error"}
    rec = {r.rid: r.t_rec for r in reports}
    by_rid = {r.rid: r for r in reports}

    def true_who(f: Fact) -> str:
        r = by_rid[f.rid]
        return by_rid[r.relay_of].source if r.relay_of else r.source

    def fake(items: list[Fact]) -> bool:
        """Подтверждение есть, но все согласные донесения восходят к одному наблюдателю."""
        return len({true_who(f) for f in items}) < 2

    facts, broken = noisy(oracle_facts(reports, profile=p), p, extract_err, seed)
    raw, _ = noisy(oracle_facts(reports, detect_relays=False, profile=p), p, extract_err, seed)
    bad = bad | broken  # ошибка источника, её пересказ или ошибка чтения
    systems = {"СОВА": EvidenceMemory(p), "СОВА без учёта пересказов": EvidenceMemory(p, independent_only=False)}
    for f in facts:
        systems["СОВА"].add(f)
    for f in raw:
        systems["СОВА без учёта пересказов"].add(f)
    ref = systems["СОВА"]

    def oracle(T: int) -> dict[str, tuple[Fact, int]]:
        """Идеальная память: знает, какие донесения ошибочны, и берёт самое
        свежее верное наблюдение. Её ошибки неустранимы: об изменении ещё
        никто не сообщил."""
        view = {}
        for f in sorted((f for f in facts if f.t_rec <= T and (f.rid, f.attr) not in bad), key=lambda f: f.t_obs):
            view[f.attr] = (f, 1)
        return view

    def decisions(view: dict) -> tuple[float, float]:
        miss = false = 0
        for r in p.rules:
            if r.attr not in view:
                continue
            shown = r.is_danger(view[r.attr][0].value)
            miss += real[r.name] and not shown
            false += shown and not real[r.name]
        return miss / len(p.rules), false / len(p.rules)

    rows = []
    ever: dict[str, list] = {name: [] for name in systems}
    for T in CHECKPOINTS:
        known = [f for f in facts if f.t_rec <= T]
        active = [(a, b) for a, b, _ in pairs if rec[a] <= T and rec[b] <= T]
        real = {r.name: r.is_danger(truth.value(r.attr, T)) for r in p.rules}

        for name, view in (("Последнее полученное", last_received(facts, T)),
                           ("Голосование источников", majority_vote(ref, facts, T)),
                           ("Идеальная память", oracle(T))):
            shown = {a: [f] for a, (f, _) in view.items()}
            row = {"system": name, "T": T, "crit": _critical_errors(p, shown, {}, known, bad, T)}
            row["fake_confirm"] = 0
            if name.startswith("Голосование"):
                for a, (f, n) in view.items():
                    if n >= 2:
                        sup = [g for g in known if g.attr == a and abs(g.t_obs - f.t_obs) <= WINDOW
                               and not ref.values_differ(f, g)]
                        row["fake_confirm"] += fake(sup)
            row["miss"], row["false"] = decisions(view)
            row["miss_drift"] = row["miss"]
            row["asks"] = row["asks_clean"] = 0
            row["recheck"] = 0.0
            row["pairs_hit"], row["pairs_active"], row["false_alarm"] = 0, len(active), 0
            row["kept"] = len(view) / max(len(known), 1)
            rows.append(row)

        for name, mem in systems.items():
            st = mem.state(T)
            shown = {a: s.shown for a, s in st.items() if s.shown}
            flags = {a: {s.status} | ({"требует уточнения"} if s.unclear else set())
                     | (set() if s.fresh else {"устарел"}) for a, s in st.items()}
            row = {"system": name, "T": T, "crit": _critical_errors(p, shown, flags, known, bad, T)}
            row["fake_confirm"] = sum(1 for s in st.values() if s.status == "согласуется" and fake(s.current))
            for label, drift in (("", False), ("_drift", True)):
                cs = conclusions(mem, T, use_drift=drift)
                row["miss" + label] = sum(c.recheck is None and not c.danger and real[c.rule.name] for c in cs) / len(cs)
                if not drift:
                    row["false"] = sum(c.recheck is None and c.danger and not real[c.rule.name] for c in cs) / len(cs)
                    row["recheck"] = sum(c.recheck is not None for c in cs) / len(cs)

            def key(a, b):
                return {(a.rid, a.attr), (b.rid, b.attr)} if b is not None else {(a.rid, a.attr)}
            contra = [key(a, b) for s in st.values() for a, b in s.conflicts]
            unclear = [key(u.a, u.b) for s in st.values() for u in s.unclear]
            ever[name] += [key(a, b) for a, b in mem.all_conflicts(T)] + unclear  # помечено хотя бы в одной сводке
            found = ever[name]
            row["pairs_hit"] = sum(1 for a, _ in active if any((a, planted_attr[a]) in k for k in found))
            row["pairs_active"] = len(active)
            row["false_alarm"] = sum(1 for k in contra if not (k & bad))
            row["asks"] = len(unclear)
            row["asks_clean"] = sum(1 for k in unclear if not (k & bad))
            row["kept"] = 1.0
            rows.append(row)
    for r in rows:
        r["seed"], r["profile"] = seed, p.key
    return rows


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


METRICS = {
    "crit": "Критические ошибки на сводку",
    "miss": "Правило сценария дало «не нужно» без пометки, а по замыслу нужно",
    "miss_drift": "То же с прогнозом сдвига показателя",
    "false": "Правило дало «нужно» без пометки, а по замыслу не нужно",
    "recheck": "Входные данные правила помечены как ненадёжные",
    "fake_confirm": "Ложные подтверждения за счёт пересказа, на сводку",
    "pairs": "Заложенные ошибки источников, которые помечены",
    "false_alarm": "Противоречия без ошибки в источнике, на сводку",
    "asks": "Пометок «требует уточнения», на сводку",
    "asks_clean": "Из них без ошибки в источнике, на сводку",
    "kept": "Сохранено фактов",
}
PERCENT = ("miss", "miss_drift", "false", "recheck", "pairs", "kept")


def aggregate(rows: list[dict]) -> dict:
    """Среднее по сводкам внутри сценария, потом по сценариям с бутстрэпом."""
    systems = list(dict.fromkeys(r["system"] for r in rows))
    scen = sorted({(r["profile"], r["seed"]) for r in rows})
    out = {}
    for s in systems:
        out[s] = {}
        for key in METRICS:
            vals = []
            for pk, sd in scen:
                rs = [r for r in rows if r["system"] == s and r["seed"] == sd and r["profile"] == pk]
                if key == "pairs":
                    act = sum(r["pairs_active"] for r in rs)
                    vals.append(sum(r["pairs_hit"] for r in rs) / act if act else 0.0)
                else:
                    vals.append(statistics.fmean(r[key] for r in rs))
            out[s][key] = (statistics.fmean(vals), *bootstrap_ci(vals))
    return out


def fmt(key: str, v) -> str:
    m, lo, hi = v
    if key in PERCENT:
        return f"{100 * m:.0f}% [{100 * lo:.0f}; {100 * hi:.0f}]"
    return f"{m:.2f} [{lo:.2f}; {hi:.2f}]".replace(".", ",")
