"""Извлечение фактов из донесений.

В демонстраторе значения берутся из разметки генератора, а время наблюдения
и признак пересказа разбираются из текста, как это делала бы система. Так
логика памяти проверяется отдельно от качества модели. Извлечение локальной
моделью делает llm_extract.py, а noisy() имитирует её ошибки, чтобы понять,
какое качество извлечения нужно.
"""
from __future__ import annotations

import random
import re
from dataclasses import replace

from .memory import Fact
from .profiles.base import COMPASS, RELAY_NAMES, Profile, Report, _distort
from .timeparse import obs_time_quote, parse_obs_time

_ORIGIN = {gen: name for name, gen in RELAY_NAMES.items()}
RELAY_RE = re.compile(r"[Пп]о данным (" + "|".join(re.escape(g) for g in sorted(_ORIGIN, key=len, reverse=True)) + r")")


def find_relay(text: str) -> str | None:
    """Чьё наблюдение пересказано, если в тексте есть «по данным ...»."""
    m = RELAY_RE.search(text)
    return _ORIGIN[m.group(1)] if m else None


def oracle_facts(reports: list[Report], tz_aware: bool = True, detect_relays: bool = True,
                 profile: Profile | None = None) -> list[Fact]:
    """В учебном генераторе все замеры одного показателя относятся к одному участку,
    поэтому участок берётся из профиля."""
    facts: list[Fact] = []

    def p_subject(attr: str) -> str:
        return profile.ind(attr).subject if profile else ""

    for r in reports:
        t_obs, est = parse_obs_time(r.text, r.t_rec, tz_aware)
        origin = (find_relay(r.text) if detect_relays else None) or r.source
        for g in r.facts:
            relay_of = None
            if origin != r.source:
                orig = next((f for f in facts if f.source == origin and f.attr == g.attr
                             and f.value == g.value and abs(f.t_obs - t_obs) <= 2), None)
                relay_of = orig.rid if orig else None
            relay = origin != r.source
            facts.append(Fact(len(facts), r.rid, r.source, g.attr, g.value, t_obs, r.t_rec, t_unknown=est,
                              cited=origin if relay else "", relay_of=relay_of,
                              subject=p_subject(g.attr), quote=r.text,
                              t_quote="" if est else obs_time_quote(r.text)))
    return facts


def noisy(facts: list[Fact], p: Profile, rate: float, seed: int = 0) -> tuple[list[Fact], set]:
    """Имитация ошибок модели при извлечении. С вероятностью rate факт
    пропущен, с той же вероятностью значение прочитано неверно и с той же
    вероятностью время сдвинуто на 6 часов (потерян часовой пояс).
    Возвращает факты и множество (донесение, показатель), испорченных при чтении."""
    if rate <= 0:
        return list(facts), set()
    rng = random.Random(f"noisy-{seed}-{rate}")
    out, broken = [], set()
    for f in facts:
        if rng.random() < rate:
            continue
        value, t_obs = f.value, f.t_obs
        if rng.random() < rate:
            value = _distort(p.ind(f.attr), f.value, rng)
        if rng.random() < rate and not f.t_unknown:
            t_obs = f.t_obs - 6 * 60
        if (value, t_obs) != (f.value, f.t_obs):
            broken.add((f.rid, f.attr))
        out.append(replace(f, fid=len(out), value=value, t_obs=t_obs))
    return out, broken


def parse_value(p: Profile, attr: str, raw: str):
    """Приводит ответ модели к типу показателя; None, если не удалось."""
    ind = p.ind(attr)
    raw = raw.strip()
    if ind.kind == "cat":
        low = raw.lower()
        return next((v for v in ind.values if v[:4] in low), None)
    if ind.kind == "wind":
        m = re.search(r"(СВ|СЗ|ЮВ|ЮЗ|С|В|Ю|З)\D*(\d+)", raw.upper())
        return (m.group(1), int(m.group(2))) if m and m.group(1) in COMPASS else None
    m = re.search(r"[-−]?\d+(?:[.,]\d+)?", raw)
    if not m:
        return None
    v = float(m.group().replace(",", ".").replace("−", "-"))
    return round(v, ind.decimals) if ind.decimals else int(v)
