"""Профиль ЧС и общий генератор учебных сценариев.

Ядро СОВЫ ничего не знает о конкретном виде ЧС. Всё, что зависит от
вида ЧС, лежит в профиле: список показателей с допусками и сроками годности,
правила плана действий, источники донесений и привязка показателей к полям
форм 2/ЧС–4/ЧС. Чтобы добавить новый вид ЧС, пишется новый профиль, код
ядра не меняется.

Генератор берёт профиль и «истинный» ход событий и пишет поток донесений:
с задержками, шумом, временем по Москве, донесениями без времени, ошибками
источников и пересказами. Разметка получается сама.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

MSK_SHIFT = 6 * 60
START, END = 10 * 60, 18 * 60
COMPASS = ["С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ"]


def hhmm(t: int) -> str:
    t %= 24 * 60
    return f"{t // 60:02d}:{t % 60:02d}"


@dataclass(frozen=True)
class Indicator:
    key: str
    name: str
    kind: str = "num"  # num | cat | wind
    unit: str = ""
    decimals: int = 0
    tol_abs: float = 0.0  # допустимое расхождение источников
    tol_rel: float = 0.0
    rate: float = 0.0  # наибольшая правдоподобная скорость изменения, единиц в минуту
    ttl: int = 120  # срок годности сведений, минут
    critical: bool = False  # ошибка в этом показателе опасна для людей
    ask: str = ""  # у кого уточнять
    phrases: tuple[str, ...] = ("{name} {v}",)
    values: tuple[str, ...] = ()  # варианты для показателя-категории
    monotone: bool = False  # категория меняется только в порядке values (затор держится, потом прорван)
    zero: str = ""  # как говорят о нуле: «пострадавших нет»
    matters: str = ""  # зачем эти сведения в сводке, для списка «требует внимания»
    needs_subject: bool = False  # сравнивать можно только утверждения об одном и том же участке
    subject: str = ""  # участок по умолчанию в учебном генераторе, где все замеры об одном объекте

    def fmt(self, v) -> str:
        if self.kind == "wind":
            return f"{v[0]} {v[1]} м/с"
        if self.kind == "cat":
            return str(v)
        if self.zero and v == 0:
            return self.zero
        s = f"{v:.{self.decimals}f}".replace(".", ",")
        return f"{s} {self.unit}".strip()


@dataclass(frozen=True)
class Rule:
    """Правило плана действий: при каком значении показателя что делать."""
    name: str
    attr: str
    op: str  # "<=" опасно при значении не больше порога, ">=" не меньше, "==" при равенстве
    threshold: object
    danger: str
    safe: str
    watch: str  # что писать, пока не ясно, перешли порог или нет

    def is_danger(self, v) -> bool:
        if self.op == "==":
            return v == self.threshold
        return v <= self.threshold if self.op == "<=" else v >= self.threshold


@dataclass(frozen=True)
class Source:
    name: str
    head: str  # как начинается донесение
    attrs: tuple[str, ...]
    times: tuple[str, ...]  # примерное время наблюдений
    delay: tuple[int, int]  # задержка доставки, мин
    noise: dict = field(default_factory=dict)  # attr -> абсолютный шум
    msk: bool = False  # пишет время по Москве
    notime: float = 0.0  # доля донесений без указания времени
    relays: bool = False  # может пересказывать чужие донесения (ЕДДС)
    events: bool = False  # сообщает об изменении сразу, как оно случилось (закрытие дороги)


@dataclass(frozen=True)
class SummaryProfile:
    """Профиль сводки: кому она, зачем и какие сведения в ней обязательны.
    Отсутствие сведений считается пробелом, только если они нужны по профилю.
    Для первого этапа один профиль; получателя и состав согласовать с Академией."""
    name: str
    reader: str
    purpose: str
    required: tuple[str, ...]
    # (показатель, условие словами, вид условия): "reported" нужны, если о них уже сообщали;
    # "rule_not_safe" нужны, если правило сценария по своему показателю не даёт «безопасно»
    conditional: tuple[tuple[str, str, str], ...] = ()
    note: str = ""  # что ещё обсудить с Академией


@dataclass
class Profile:
    key: str
    title: str
    place: str
    indicators: dict[str, Indicator]
    sources: list[Source]
    rules: list[Rule]
    forms: list[tuple[str, str, str]]  # (форма, пункт, показатель)
    make_truth: Callable[[random.Random], "object"]
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)  # группа штаба -> её показатели
    calls: tuple[str, ...] = ("просят прислать автобус для пожилых жителей",
                              "соседи говорят, что скоро всех вывезут, это правда?")  # звонки без фактов
    summary: SummaryProfile | None = None

    def ind(self, key: str) -> Indicator:
        return self.indicators[key]


@dataclass
class GoldFact:
    attr: str
    value: object
    t_obs: int
    planted: str | None = None  # "error" у ошибки источника, "echo" у её пересказа


@dataclass
class Report:
    rid: str
    source: str
    t_obs: int
    t_rec: int
    text: str
    facts: list[GoldFact] = field(default_factory=list)
    relay_of: str | None = None


def _to_min(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _observe(p: Profile, src: Source, attr: str, truth, t: int, rng: random.Random, noise: float):
    ind = p.ind(attr)
    v = truth.value(attr, t)
    if ind.kind == "num":
        n = src.noise.get(attr, 0.0) * noise
        v = v + rng.uniform(-n, n)
        v = round(v, ind.decimals) if ind.decimals else int(round(v))
        return max(v, 0) if attr != "air_temp" else v
    return v


def _distort(ind: Indicator, v, rng: random.Random):
    """Ошибка источника: значение, несовместимое с правдой при любом допуске."""
    if ind.kind == "cat":
        return next(o for o in ind.values if o != v)
    if ind.kind == "wind":
        d = COMPASS[(COMPASS.index(v[0]) + 3) % 8]
        return (d, max(2, v[1] - 6))
    gap = 2.5 * (max(ind.tol_abs, ind.tol_rel * abs(v)) + ind.rate * 30)
    sign = rng.choice([-1, 1]) if v - gap > 0 else 1
    w = v + sign * gap
    return round(w, ind.decimals) if ind.decimals else int(round(w))


def _phrase(ind: Indicator, v, rng: random.Random) -> str:
    if ind.zero and v == 0:
        return ind.zero
    return rng.choice(ind.phrases).format(v=ind.fmt(v), name=ind.name.lower())


def _when(t_obs: int, msk: bool, notime: bool, rng: random.Random) -> str:
    if notime:
        return rng.choice(["сейчас ", "только что ", ""])
    if msk:
        return f"по состоянию на {hhmm(t_obs - MSK_SHIFT)} МСК "
    return rng.choice([f"на {hhmm(t_obs)} ", f"в {hhmm(t_obs)} ", f"по состоянию на {hhmm(t_obs)} "])


def _text(head: str, when: str, phrases: list[str], relay_from: str | None = None) -> str:
    body = when + ", ".join(phrases)
    if relay_from:
        body = f"по данным {relay_from}, " + body
    if head.endswith(". "):
        body = body[0].upper() + body[1:]
    return head + body + ("" if body.endswith(".") else ".")


RELAY_NAMES = {"РТП": "РТП", "Глава поселения": "главы администрации", "Оперативная группа": "оперативной группы",
               "Гидропост": "гидропоста", "Котельная": "котельной", "Фельдшер": "фельдшера",
               "Авиалесоохрана": "Авиалесоохраны", "Метеостанция": "метеостанции", "Дорожная служба": "дорожной службы",
               "ИСДМ": "ИСДМ-Рослесхоз", "БПЛА": "оператора БПЛА", "Авиаразведка": "авиаразведки", "ГИБДД": "ГИБДД"}


def generate(p: Profile, seed: int, noise: float = 1.0, n_errors: int = 8, n_relays: int = 3):
    """Возвращает (истина, донесения, пары «ошибка, верное донесение»)."""
    rng = random.Random(f"{p.key}-{seed}")
    truth = p.make_truth(rng)
    plan = []  # (t_obs, source)
    for src in p.sources:
        for t in src.times:
            plan.append((_to_min(t) + rng.randint(-4, 4), src))
        if src.events:
            for attr in src.attrs:
                prev = truth.value(attr, START)
                for t in range(START + 1, END):
                    v = truth.value(attr, t)
                    if v != prev:
                        plan.append((t + rng.randint(5, 20), src))
                    prev = v
    plan.sort(key=lambda x: x[0])

    reports: list[Report] = []
    for t_obs, src in plan:
        notime = rng.random() < src.notime
        facts = [GoldFact(a, _observe(p, src, a, truth, t_obs, rng, noise), t_obs) for a in src.attrs]
        t_rec = t_obs + rng.randint(*src.delay)
        reports.append(Report("", src.name, t_obs, t_rec, "", facts))

    # ошибки источников: только там, где рядом есть верное донесение другого источника о том же
    def neighbours(i: int, attr: str, window: int = 30) -> list[int]:
        r = reports[i]
        return [j for j, o in enumerate(reports) if o.source != r.source
                and any(g.attr == attr and abs(g.t_obs - r.t_obs) <= window for g in o.facts)]

    events = {s.name for s in p.sources if s.events}
    candidates = [(i, f.attr) for i, r in enumerate(reports) for f in r.facts
                  if r.source not in events and neighbours(i, f.attr)]
    rng.shuffle(candidates)
    planted_reports, planted, protected = set(), set(), set()
    for i, attr in candidates:
        if len(planted_reports) >= n_errors or i in planted_reports or (i, attr) in protected:
            continue
        near = neighbours(i, attr)
        if any((j, attr) in planted for j in range(len(reports))
               if abs(reports[j].t_obs - reports[i].t_obs) <= 60):
            continue  # две независимые ошибки об одном и том же почти в одно время маловероятны
        free = [j for j in near if j not in planted_reports]
        if not free:
            continue
        f = next(g for g in reports[i].facts if g.attr == attr)
        f.value, f.planted = _distort(p.ind(attr), f.value, rng), "error"
        planted_reports.add(i)
        planted.add((i, attr))
        protected.add((free[0], attr))
        protected.update((free[0], g.attr) for g in reports[free[0]].facts)

    # пересказы: ЕДДС пересылает чужие донесения, одно из них с ошибкой
    relay_src = next(s for s in p.sources if s.relays)
    pool = [r for r in reports if r.source not in (relay_src.name, "112") and r.facts]
    with_error = [r for r in pool if any(f.planted for f in r.facts)]
    chosen = (with_error[:1] if with_error else []) + rng.sample([r for r in pool if r not in with_error[:1]],
                                                                  n_relays - min(1, len(with_error)))
    relays = []
    for orig in chosen:
        facts = [GoldFact(f.attr, f.value, f.t_obs, "echo" if f.planted else None) for f in orig.facts]
        relays.append((orig, Report("", relay_src.name, orig.t_obs, orig.t_rec + rng.randint(8, 30), "", facts)))

    allr = reports + [r for _, r in relays]
    allr.sort(key=lambda r: (r.t_rec, r.t_obs))
    for k, r in enumerate(allr, 1):
        r.rid = f"Д-{k:02d}"
    for orig, rel in relays:
        rel.relay_of = orig.rid
    src_by = {s.name: s for s in p.sources}
    for r in allr:
        src = src_by[r.source]
        if r.relay_of:
            orig = next(o for o in allr if o.rid == r.relay_of)
            phrases = [_phrase(p.ind(f.attr), f.value, rng) for f in r.facts]
            r.text = _text(src.head, _when(r.t_obs, True, False, rng), phrases,
                           RELAY_NAMES.get(orig.source, orig.source))
        elif not r.facts:
            r.text = src.head + rng.choice(p.calls)
        else:
            notime = rng.random() < src.notime
            phrases = [_phrase(p.ind(f.attr), f.value, rng) for f in r.facts]
            r.text = _text(src.head, _when(r.t_obs, src.msk, notime, rng), phrases)
    return truth, allr, _planted_pairs(allr)


def _planted_pairs(reports: list[Report]) -> list[tuple[str, str, str]]:
    """(донесение с ошибкой, ближайшее верное донесение другого источника, показатель)."""
    pairs = []
    for r in reports:
        for f in r.facts:
            if f.planted != "error":
                continue
            best = None
            for o in reports:
                if o.source == r.source or o.relay_of:
                    continue
                for g in o.facts:
                    if g.attr == f.attr and not g.planted and abs(g.t_obs - f.t_obs) <= 30:
                        if best is None or abs(g.t_obs - f.t_obs) < best[0]:
                            best = (abs(g.t_obs - f.t_obs), o.rid)
            if best:
                pairs.append((r.rid, best[1], f.attr))
    return pairs
