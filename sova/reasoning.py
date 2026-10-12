"""Вопросы к источникам и проверка входных данных правил учебного сценария.

Главный выход СОВЫ это сведения, их источники, ограничения и вопросы.
Решений СОВА не предлагает. Блок «к плану действий» только показывает, что
даёт правило из замысла учения на текущих сведениях и надёжны ли входные
данные этого правила. Пороги правил берутся из профиля и должны прийти
от Академии; в примере они условные.

Вывод помнит своё основание, то есть утверждения, из которых он получен.
Если утверждение из основания спорное, требует уточнения или устарело,
вывод помечается «пересмотреть» (идея систем поддержки истинности, Doyle, 1979).

Проверка «мог ли показатель уже перейти порог» по наблюдавшейся скорости
это прогноз, а не сведения, поэтому по умолчанию выключена (use_drift=False).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .memory import AttrState, EvidenceMemory, Fact
from .profiles.base import Rule, hhmm


@dataclass
class Conclusion:
    rule: Rule
    value: str
    basis: list[Fact] = field(default_factory=list)
    recheck: str | None = None

    @property
    def danger(self) -> bool:
        return self.value == self.rule.danger


@dataclass
class Ask:
    priority: int  # 1 срочно, 2 в плановом порядке
    attr: str
    problem: str
    whom: str
    state: AttrState
    kind: str = "value"  # value: уточнить значение; detail: уточнить время, участок, наблюдателя
    facts: list[Fact] = field(default_factory=list)
    need: tuple[str, ...] = ()


def _q(ind, v) -> str:
    return f"«{v}»" if ind.kind == "cat" else ind.fmt(v)


def low(s: str) -> str:
    return s[0].lower() + s[1:]


def when(f: Fact) -> str:
    return "время не указано" if f.t_unknown else f"на {hhmm(f.t_obs)}"


def who(f: Fact) -> str:
    """Кто передал и на кого ссылается."""
    if f.cited:
        return f"{f.source}, со слов: {f.cited}" + (f", наблюдал: {f.observer}" if f.observer else "")
    return f.source


def say(mem: EvidenceMemory, f: Fact) -> str:
    """Утверждение одной строкой: номер, источник, время, участок, значение."""
    ind = mem.p.ind(f.attr)
    place = ""
    if ind.needs_subject:
        place = f", {f.subject}" if f.subject else ", участок не указан"
    return f"{f.rid} ({who(f)}, {when(f)}{place}): {ind.fmt(f.value).rstrip('.')}"


def _jump_text(mem: EvidenceMemory, s: AttrState) -> str:
    a, b = s.jump
    ind = mem.p.ind(s.attr)
    how = ("при допустимой скорости изменения так быстро не меняется (скорость условная)"
           if ind.kind == "num" else "обратного перехода не бывает")
    return f"резкое расхождение с прежним замером: {say(mem, a)}; {say(mem, b)}; {how}"


def sides(mem: EvidenceMemory, s: AttrState) -> str:
    """Противоречие словами: обе стороны, свежие первыми."""
    seen, parts = set(), []
    for a, b in s.conflicts:
        for f in (a, b):
            if f.fid not in seen:
                seen.add(f.fid)
                parts.append(f)
    return "; ".join(say(mem, f) for f in sorted(parts, key=lambda f: -f.t_obs))


def _shown(s: AttrState) -> list[Fact]:
    if s.decision:
        return [f for f in s.shown if f.rid == s.decision.rid]
    return s.shown


def conclusions(mem: EvidenceMemory, T: int, known_at: int | None = None, use_drift: bool = False) -> list[Conclusion]:
    """Что даёт правило учебного сценария на текущих сведениях и можно ли
    опираться на его входные данные. Это не рекомендация."""
    st = mem.state(T, known_at)
    out = []
    for rule in mem.p.rules:
        s = st[rule.attr]
        ind = mem.p.ind(rule.attr)
        facts = _shown(s)
        if not facts:
            out.append(Conclusion(rule, rule.watch, [], f"нет сведений: {low(ind.name)}"))
            continue
        flags = [rule.is_danger(f.value) for f in facts]
        if all(flags):
            c = Conclusion(rule, rule.danger, facts)
        elif not any(flags):
            c = Conclusion(rule, rule.safe, facts)
        else:
            c = Conclusion(rule, rule.watch, facts, "сведения лежат по разные стороны порога")
        if s.decision:
            pass
        elif not s.fresh:
            c.value, c.recheck = rule.watch, f"сведения устарели, последние {when(facts[-1])}"
        elif s.status in ("противоречие", "требует уточнения") and c.recheck:
            c.recheck += f", статус «{s.status}»: {s.why}"
        elif s.status in ("противоречие", "требует уточнения"):
            c.recheck = f"входные данные: «{s.status}» ({s.why})"
        elif (c.value == rule.safe and s.status == "не подтверждено" and s.superseded
              and rule.is_danger(s.superseded[-1].value)):
            prev = s.superseded[-1]
            c.value = rule.watch
            c.recheck = (f"опасность снята одним неподтверждённым донесением {facts[-1].rid}, "
                         f"до этого {_q(ind, prev.value)} ({prev.rid}, {prev.who})")
        if use_drift and ind.kind == "num" and not s.decision and s.fresh:
            safe = [f for f in facts if not rule.is_danger(f.value) and not f.t_unknown]
            if safe:
                last = min(safe, key=lambda f: f.value) if rule.op == "<=" else max(safe, key=lambda f: f.value)
                rate = mem.trend(rule.attr, T, towards_lower=(rule.op == "<="), known_at=known_at)
                moved = rate * (T - last.t_obs)
                projected = last.value - moved if rule.op == "<=" else last.value + moved
                if moved > 0 and rule.is_danger(projected):
                    drift = (f"прогноз: с {hhmm(last.t_obs)} показатель мог уйти на {ind.fmt(moved)} и перейти порог "
                             f"{ind.fmt(rule.threshold)}")
                    c.value = rule.watch
                    c.recheck = drift if c.recheck is None else f"{c.recheck}; {drift}"
        out.append(c)
    return out


def _needs(mem: EvidenceMemory, f: Fact) -> tuple[str, ...]:
    need = []
    if f.t_unknown:
        need.append("время наблюдения")
    ind = mem.p.ind(f.attr)
    if ind.needs_subject and not f.subject:
        need.append("до какого места измеряли" if ind.kind == "num" else "какой участок и для какого транспорта")
    if not f.observed:
        need.append("кто наблюдал")
    return tuple(need)


def needed(mem: EvidenceMemory, T: int, st: dict[str, AttrState] | None = None) -> dict[str, str]:
    """Какие показатели нужны по профилю сводки сейчас: обязательные и те, чьё
    условие выполнено. Отсутствие остальных пробелом не считается."""
    st = mem.state(T) if st is None else st
    sp = mem.p.summary
    if sp is None:
        return {a: "обязательно" for a, i in mem.p.indicators.items() if i.critical or a in {r.attr for r in mem.p.rules}}
    out = {a: "обязательно" for a in sp.required}
    rules = {c.rule.attr: c for c in conclusions(mem, T)}
    for attr, text, kind in sp.conditional:
        if kind == "reported" and st[attr].status != "нет сведений":
            out[attr] = text
        elif kind == "rule_not_safe" and any(c.value != c.rule.safe for c in rules.values()
                                             if c.rule.attr in sp.required):
            out[attr] = text
    return out


def ask_queue(mem: EvidenceMemory, T: int, drift: bool = False) -> list[Ask]:
    """Что уточнить и у кого. Сначала то, от чего зависят правила сценария
    и безопасность людей, потом остальное. Пробелы и устаревшие сведения
    спрашиваем, только если они нужны по профилю сводки. Вопрос готовит СОВА,
    отправляет офицер."""
    st = mem.state(T)
    need = needed(mem, T, st)
    rule_attrs = [r.attr for r in mem.p.rules]
    asks: list[Ask] = []
    for attr, s in st.items():
        ind = mem.p.ind(attr)
        prio = 1 if attr in rule_attrs or ind.critical else 2
        if s.status == "нет сведений":
            if attr in need:
                asks.append(Ask(prio, attr, "сведений нет", ind.ask, s))
        elif not s.fresh and s.status in ("не подтверждено", "согласуется"):
            if attr in need:
                last = (s.current or s.loose or s.superseded)[-1]
                asks.append(Ask(prio, attr, f"сведения устарели, последние {when(last)}", ind.ask, s, facts=[last]))
        elif s.status == "противоречие":
            fs = sorted({f for p in s.conflicts for f in p}, key=lambda f: f.fid)
            whom = ", ".join(dict.fromkeys(f.who for f in fs))
            asks.append(Ask(prio, attr, "противоречие: " + sides(mem, s), whom, s, facts=fs))
        elif s.status == "требует уточнения":
            if s.jump:
                asks.append(Ask(prio, attr, _jump_text(mem, s), ind.ask, s, facts=[s.jump[1]]))
            weak: dict[str, list[Fact]] = {}
            other: dict[str, list[Fact]] = {}
            for u in s.unclear:
                for f, g in ((u.a, u.b), (u.b, u.a)):
                    if f is not None and _needs(mem, f):
                        k = f.cited or f.source
                        if f not in weak.setdefault(k, []):
                            weak[k].append(f)
                        if g is not None and g not in other.setdefault(k, []):
                            other[k].append(g)
            for whom, fs in weak.items():
                missing = tuple(dict.fromkeys(n for f in fs for n in _needs(mem, f)))
                vs = "; ".join(say(mem, f) for f in fs)
                rest = other.get(whom, [])
                problem = (f"{vs}. Не указано: {', '.join(missing)}"
                           + (f". Поэтому нельзя сопоставить с {', '.join(g.rid for g in rest)}" if rest else ""))
                asks.append(Ask(prio, attr, problem, whom, s, "detail", fs, missing))
    if drift:
        for c in conclusions(mem, T, use_drift=True):
            if c.recheck and "прогноз" in c.recheck and not any(a.attr == c.rule.attr for a in asks):
                asks.append(Ask(1, c.rule.attr, c.recheck, mem.p.ind(c.rule.attr).ask, st[c.rule.attr]))
    order = {a: i for i, a in enumerate(rule_attrs)}
    return sorted(asks, key=lambda a: (a.priority, order.get(a.attr, 99), a.attr))


def draft_request(mem: EvidenceMemory, ask: Ask, T: int) -> str:
    """Проект вопроса. Проверяет и отправляет офицер штаба."""
    ind = mem.p.ind(ask.attr)
    s = ask.state
    name = low(ind.name)
    if ask.kind == "detail":
        got = "; ".join(f"{f.rid} от {hhmm(f.t_rec)}"
                        + (f" ({f.source} передал с ваших слов)" if f.cited else "")
                        + f": «{f.quote or ind.fmt(f.value)}»" for f in ask.facts)
        return (f"Кому: {ask.whom}. В штаб поступило: {got}. "
                f"Прошу уточнить {', '.join(ask.need)}.")
    if s.status == "противоречие":
        t = max(f.t_obs for f in ask.facts)
        return (f"Кому: {ask.whom}. Сведения расходятся: {sides(mem, s)}. Прошу уточнить, кто и где проверял, "
                f"как было на {hhmm(t)} и как сейчас.")
    elif s.status == "требует уточнения" and s.jump:
        a, b = s.jump
        context = f"Сведения резко расходятся с прежними. {say(mem, a)}; {say(mem, b)}."
    elif not s.fresh and ask.facts:
        context = f"Последние сведения: {say(mem, ask.facts[0])}."
    elif s.status == "нет сведений":
        context = "Сведений в штаб не поступало."
    else:
        context = ask.problem[0].upper() + ask.problem[1:] + "."
    return (f"Кому: {ask.whom}. Прошу сообщить по состоянию на {hhmm(T)}: {name}, время наблюдения, "
            f"кто наблюдал. {context}")
