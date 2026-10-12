"""Базовые линии, с которыми сравнивается память."""
from __future__ import annotations

from .memory import WINDOW, EvidenceMemory, Fact


def last_received(facts: list[Fact], T: int) -> dict[str, tuple[Fact, int]]:
    """Б1. Последнее полученное затирает предыдущее. Так устроена память
    агентов, которая при конфликте оставляет более новую запись, и так же
    часто ведут обстановку на доске или в таблице."""
    view: dict[str, tuple[Fact, int]] = {}
    for f in sorted((f for f in facts if f.t_rec <= T), key=lambda f: (f.t_rec, f.fid)):
        view[f.attr] = (f, 1)
    return view


def _same(mem: EvidenceMemory, a: Fact, b: Fact) -> bool:
    """Совпадают ли значения с учётом допуска. Голосование не знает о неизвестном
    времени и берёт время наблюдения как есть."""
    ind = mem.p.ind(a.attr)
    if ind.kind != "num":
        return a.value == b.value if ind.kind == "cat" else not mem.values_differ(a, b)
    mean = (abs(a.value) + abs(b.value)) / 2
    return abs(a.value - b.value) <= max(ind.tol_abs, ind.tol_rel * mean) + ind.rate * abs(a.t_obs - b.t_obs)


def majority_vote(mem: EvidenceMemory, facts: list[Fact], T: int) -> dict[str, tuple[Fact, int]]:
    """Б2. Голосование источников среди почти одновременных наблюдений.
    Побеждает значение, с которым согласно больше донесений, при равенстве
    более свежее. Пересказ считается отдельным голосом."""
    view = {}
    for attr in mem.p.indicators:
        known = [f for f in facts if f.attr == attr and f.t_rec <= T]
        if not known:
            continue
        latest = max(f.t_obs for f in known)
        cur = [f for f in known if f.t_obs >= latest - WINDOW]
        support = {f.fid: sum(1 for g in cur if _same(mem, f, g)) for f in cur}
        best = max(cur, key=lambda f: (support[f.fid], f.t_obs))
        view[attr] = (best, support[best.fid])
    return view
