"""Проекты форм донесений 2/ЧС–4/ЧС (приказ МЧС России от 11.01.2021 № 2).

Какие пункты форм каким показателям соответствуют, задаёт профиль ЧС. Здесь
для каждого пункта собирается значение, его статус и ссылки на донесения,
а пустые пункты превращаются в запросы. Подписывает форму человек.

Формы 2/ЧС–4/ЧС после первого донесения обновляются к 7:00 и 19:00 по
Москве. В Якутске это 13:00 и 01:00 по местному времени.
"""
from __future__ import annotations

from dataclasses import dataclass

from .memory import AttrState, EvidenceMemory, Fact
from .profiles.base import MSK_SHIFT, hhmm

DEADLINES_MSK = (7 * 60, 19 * 60)
READY = ("согласуется", "принято штабом")


@dataclass
class FormField:
    form: str
    item: str
    attr: str
    name: str
    value: str
    status: str
    basis: list[Fact]
    todo: str  # что сделать до подачи, пусто если пункт готов


def next_deadline(T: int) -> int:
    """Ближайший срок обновления форм, местное время в минутах от полуночи."""
    local = sorted((d + MSK_SHIFT) % (24 * 60) for d in DEADLINES_MSK)
    for d in local:
        if d >= T:
            return d
    return local[0] + 24 * 60


def last_deadline(T: int) -> int | None:
    local = sorted((d + MSK_SHIFT) % (24 * 60) for d in DEADLINES_MSK)
    past = [d for d in local if d <= T]
    return past[-1] if past else None


def _latest_by_who(facts: list[Fact]) -> list[Fact]:
    """Последнее утверждение каждого наблюдателя; пересказ не повторяем, если есть первоисточник."""
    latest: dict[str, Fact] = {}
    for f in sorted(facts, key=lambda f: (f.t_obs, f.fid)):
        if not f.cited or f.who not in latest:
            latest[f.who] = f
    return sorted(latest.values(), key=lambda f: (f.t_unknown, -f.t_obs))


def _value(mem: EvidenceMemory, s: AttrState) -> str:
    ind = mem.p.ind(s.attr)
    if s.decision:
        f = next(f for f in s.shown if f.rid == s.decision.rid)
        return ind.fmt(f.value)
    if s.conflicts:
        facts = [f for pair in s.conflicts for f in pair]
        text = " или ".join(f"{ind.fmt(f.value)} ({f.who}, {f.rid})" for f in _latest_by_who(facts))
    elif s.current:
        by_subject: dict[str, Fact] = {}
        for f in sorted(s.current, key=lambda f: (f.t_obs, f.fid)):
            by_subject[f.subject if ind.needs_subject else ""] = f
        if len(by_subject) == 1:
            text = ind.fmt(next(iter(by_subject.values())).value)
        else:
            text = "; ".join(f"{ind.fmt(f.value)} ({k or 'участок не указан'})" for k, f in by_subject.items())
    else:
        text = ""
    if s.loose:
        extra = ", ".join(f"{ind.fmt(f.value)} ({f.who}, {f.rid})" for f in _latest_by_who(s.loose))
        text = f"{text}; без указания времени: {extra}" if text else f"время не указано: {extra}"
    return text


def build(mem: EvidenceMemory, T: int, known_at: int | None = None) -> list[FormField]:
    st = mem.state(T, known_at)
    out = []
    for form, item, attr in mem.p.forms:
        ind, s = mem.p.ind(attr), st[attr]
        if not s.shown:
            out.append(FormField(form, item, attr, ind.name, "", "нет сведений", [], f"запросить: {ind.ask}"))
            continue
        basis = [f for f in s.shown if not s.decision or f.rid == s.decision.rid]
        if not s.fresh:
            todo = f"обновить сведения: {ind.ask}"
        elif s.status in READY:
            todo = ""
        elif s.status == "противоречие":
            todo = f"выбрать значение после уточнения: {ind.ask}"
        elif s.status == "требует уточнения":
            todo = f"уточнить: {s.why}"
        else:
            todo = f"один источник, по возможности подтвердить: {ind.ask}"
        out.append(FormField(form, item, attr, ind.name, _value(mem, s), s.status, basis, todo))
    return out


def readiness(fields: list[FormField]) -> tuple[int, int, int]:
    """(готово, требует проверки, пусто)."""
    ready = sum(f.status in READY for f in fields)
    empty = sum(f.status == "нет сведений" for f in fields)
    return ready, len(fields) - ready - empty, empty


def changed_since(mem: EvidenceMemory, T_prev: int, T: int) -> list[tuple[FormField, FormField]]:
    """Пункты, у которых с прошлой подачи поменялось значение или статус."""
    before = {(f.form, f.item, f.attr): f for f in build(mem, T_prev)}
    return [(before[(f.form, f.item, f.attr)], f) for f in build(mem, T)
            if (before[(f.form, f.item, f.attr)].value, before[(f.form, f.item, f.attr)].status) != (f.value, f.status)]
