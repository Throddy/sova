"""Доказательная память штаба. Не зависит от вида ЧС: всё специфичное берётся из профиля.

Что хранится. Утверждение из донесения: что сообщили и о каком показателе,
кто передал, на кого ссылается, кто наблюдал (только если это сказано),
о каком участке речь, когда наблюдали (если сказано), когда штаб получил,
и точная цитата. Решение штаба: какое из спорных донесений принято, кем и когда.
Ничего не удаляется, статус пересчитывается на нужный момент по правилам.
Модель (LLM) в этом не участвует, она только заполняет поля утверждения.

Три вида расхождений, которые различает память:

- изменение обстановки: значения разные, но относятся к разным моментам
  и совместимы с допустимой скоростью изменения;
- противоречие: два сопоставимых утверждения несовместимы. Сопоставимы они,
  если у обоих известно время наблюдения и оно почти одинаковое, известен
  наблюдатель, участок один и тот же, а источники независимы;
- согласуется: другой отправитель почти на то же время (10 минут) сообщает
  совместимое значение. Это не доказательство: два отправителя могут опираться
  на одно наблюдение. Более раннее совместимое значение позднее не подтверждает;
- требует уточнения: значения расходятся, но сопоставить их нельзя, потому что
  не указано время, участок или наблюдатель, или потому что значение резко
  не сходится с прежними замерами одного источника. Это не обвинение источника,
  а вопрос.

Актуальность считается отдельно от проверки: у каждого типа сведений свой
срок, после него сведения помечаются как устаревшие, но статус проверки
остаётся. Сроки условные, их согласует эксперт.

Времени, которого нет в тексте, память не придумывает. Про такое утверждение
известно только одно: наблюдали его не позже, чем оно пришло в штаб.

Две оси времени. state(T) показывает обстановку на момент T по тому, что
штаб знал к моменту known_at. По умолчанию known_at = T, то есть «как видел
штаб тогда». Если known_at позже T, получаем «что теперь известно о моменте T».
Это нужно для разбора после ЧС и после учений.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .profiles.base import COMPASS, Profile

WINDOW = 30  # минут: ближе этого несовместимые значения считаем противоречием с учётом скорости (условно)
AGREE_WINDOW = 10  # минут: согласие засчитываем только почти на одно время; раннее значение позднее не подтверждает
RULES_VERSION = "2026-10-11"  # версия правил, пишется в историю статусов


@dataclass(frozen=True)
class Fact:
    fid: int
    rid: str  # номер донесения
    source: str  # кто передал донесение в штаб
    attr: str
    value: object
    t_obs: int  # время наблюдения; если не указано, здесь время получения, только для порядка
    t_rec: int  # время получения штабом
    t_unknown: bool = False  # время наблюдения в тексте не указано
    cited: str = ""  # на кого ссылается донесение: «по данным главы» даёт «глава»
    relay_of: str | None = None  # номер донесения, которое пересказано, если оно есть в штабе
    observer: str = ""  # кто наблюдал, только если сказано явно или донесение от первого лица
    subject: str = ""  # участок или объект; пусто, если не указан
    quote: str = ""  # точный фрагмент текста, из которого взято значение
    t_quote: str = ""  # фрагмент текста, из которого код взял время наблюдения; пусто, если времени нет

    @property
    def who(self) -> str:
        """Чьё это утверждение по существу. Пересказ не становится вторым источником."""
        return self.observer or self.cited or self.source

    @property
    def observed(self) -> bool:
        """Известно, кто наблюдал: либо сказано явно, либо источник сообщает от себя."""
        return bool(self.observer) or not self.cited


@dataclass(frozen=True)
class Decision:
    attr: str
    rid: str  # какое донесение штаб принял за верное
    by: str
    at: int
    note: str = ""


@dataclass(frozen=True)
class Unclear:
    """Расхождение, которое нельзя признать ни изменением, ни противоречием."""
    a: Fact
    b: Fact | None
    reasons: tuple[str, ...]


@dataclass
class AttrState:
    attr: str
    status: str  # состояние проверки: нет сведений | не подтверждено | согласуется | требует уточнения | противоречие | принято штабом
    current: list[Fact] = field(default_factory=list)  # утверждения с известным временем на последний момент
    superseded: list[Fact] = field(default_factory=list)  # более ранние, остаются в истории
    conflicts: list[tuple[Fact, Fact]] = field(default_factory=list)  # установленные противоречия
    age: int = 0
    decision: Decision | None = None
    jump: tuple[Fact, Fact] | None = None  # (прежнее надёжное наблюдение, резко отличающееся новое)
    loose: list[Fact] = field(default_factory=list)  # без времени, могли быть позже текущих
    unclear: list[Unclear] = field(default_factory=list)
    confirmed: bool = False  # есть совместимое сообщение другого отправителя почти на то же время
    why: str = ""  # почему такой статус, словами
    fresh: bool = True  # актуальность отдельно от проверки: сведения не старше срока для этого типа

    @property
    def shown(self) -> list[Fact]:
        return self.current + self.loose


class EvidenceMemory:
    def __init__(self, profile: Profile, independent_only: bool = True) -> None:
        self.p = profile
        self.independent_only = independent_only
        self._facts: list[Fact] = []
        self._decisions: list[Decision] = []
        self._notes: list[tuple[str, str, int, str]] = []
        self._originals: dict[str, tuple[str, int, str]] = {}

    # запись
    def add(self, fact: Fact) -> None:
        self._facts.append(fact)

    def add_note(self, rid: str, source: str, t_rec: int, text: str) -> None:
        """Сообщение, из которого не удалось извлечь утверждений по схеме.
        Не теряется: показывается оператору отдельным списком."""
        self._notes.append((rid, source, t_rec, text))

    def notes(self, T: int) -> list[tuple[str, str, int, str]]:
        return [n for n in self._notes if n[2] <= T]

    def add_original(self, rid: str, source: str, t_rec: int, text: str) -> None:
        """Оригинал донесения как пришёл. Не меняется."""
        self._originals[rid] = (source, t_rec, text)

    def quote_ok(self, f: Fact) -> bool | None:
        """Первая проверка основания, её делает код: цитата дословно есть в оригинале,
        то есть модель её не придумала и не дополнила. Соответствует ли утверждение
        цитате (значение, отрицание, время, объект, источник), код не проверяет:
        это разметка и проверка оператором. None, если оригинала нет."""
        orig = self._originals.get(f.rid)
        if orig is None or not f.quote:
            return None
        norm = lambda t: " ".join(t.split()).lower()
        return norm(f.quote) in norm(orig[2])

    def decide(self, attr: str, rid: str, by: str, at: int, note: str = "") -> None:
        self._decisions.append(Decision(attr, rid, by, at, note))

    # сравнение утверждений
    def _who(self, f: Fact) -> str:
        return f.who if self.independent_only else f.source

    def independent(self, a: Fact, b: Fact) -> bool:
        return self._who(a) != self._who(b)

    def other_object(self, a: Fact, b: Fact) -> bool:
        """Оба указали участок, и участки разные: это разные объекты, а не спор."""
        return self.p.ind(a.attr).needs_subject and bool(a.subject) and bool(b.subject) and a.subject != b.subject

    def obstacles(self, a: Fact, b: Fact) -> tuple[str, ...]:
        """Что мешает сопоставить два утверждения. Пусто, если сопоставимы."""
        why = []
        if a.t_unknown or b.t_unknown:
            why.append("не указано время наблюдения")
        if self.p.ind(a.attr).needs_subject and not (a.subject and b.subject):
            why.append("не указан участок")
        if not (a.observed and b.observed):
            why.append("не указано, кто наблюдал")
        return tuple(why)

    def values_differ(self, a: Fact, b: Fact, dt: int | None = None) -> bool:
        """Расходятся ли значения больше допуска с учётом допустимой скорости
        изменения за dt минут. Допуски и скорости условные, их задают эксперты."""
        ind = self.p.ind(a.attr)
        if dt is None:
            dt = 0 if a.t_unknown or b.t_unknown else abs(a.t_obs - b.t_obs)
        if ind.kind == "cat":
            return a.value != b.value
        if ind.kind == "wind":
            steps = abs(COMPASS.index(a.value[0]) - COMPASS.index(b.value[0]))
            return min(steps, 8 - steps) >= 2 or abs(a.value[1] - b.value[1]) > 3 + 0.05 * dt
        mean = (abs(a.value) + abs(b.value)) / 2
        return abs(a.value - b.value) > max(ind.tol_abs, ind.tol_rel * mean) + ind.rate * dt

    def _close(self, a: Fact, b: Fact) -> bool:
        return not a.t_unknown and not b.t_unknown and abs(a.t_obs - b.t_obs) <= WINDOW

    def conflict(self, a: Fact, b: Fact) -> bool:
        """Установленное противоречие: сопоставимые утверждения несовместимы."""
        return (a.attr == b.attr and self._close(a, b) and self.independent(a, b)
                and not self.other_object(a, b) and not self.obstacles(a, b) and self.values_differ(a, b))

    def agree(self, a: Fact, b: Fact) -> bool:
        """Согласие: другой отправитель, тот же участок, время почти одно (AGREE_WINDOW),
        значение совместимо. Утверждение без времени ничего не подтверждает. Независимость
        самих наблюдений этим не доказана: два отправителя могут опираться на одно наблюдение."""
        return (a.attr == b.attr and self._close(a, b) and abs(a.t_obs - b.t_obs) <= AGREE_WINDOW
                and self.independent(a, b)
                and not self.other_object(a, b) and not self.obstacles(a, b) and not self.values_differ(a, b))

    def possible(self, a: Fact, b: Fact) -> tuple[str, ...]:
        """Возможное расхождение: значения не сходятся, если считать их одновременными,
        но сопоставить утверждения нельзя. Возвращает причины, пусто, если не оно."""
        if a.attr != b.attr or not self.independent(a, b) or self.other_object(a, b):
            return ()
        why = self.obstacles(a, b)
        if not why:
            return ()
        if not (a.t_unknown or b.t_unknown) and not self._close(a, b):
            return ()  # время известно и разное: это изменение, сравнивает проверка скачков
        return why if self.values_differ(a, b) else ()

    # чтение
    def known(self, known_at: int, valid_at: int | None = None) -> list[Fact]:
        return [f for f in self._facts if f.t_rec <= known_at and (valid_at is None or f.t_obs <= valid_at)]

    def history(self, attr: str, T: int, known_at: int | None = None) -> list[Fact]:
        known_at = T if known_at is None else known_at
        return sorted((f for f in self.known(known_at, T) if f.attr == attr), key=lambda f: (f.t_obs, f.fid))

    def all_conflicts(self, T: int) -> list[tuple[Fact, Fact]]:
        facts = self.known(T)
        return [(a, b) for i, a in enumerate(facts) for b in facts[i + 1:] if self.conflict(a, b)]

    def state(self, T: int, known_at: int | None = None) -> dict[str, AttrState]:
        known_at = T if known_at is None else known_at
        out = {}
        for attr, ind in self.p.indicators.items():
            hist = self.history(attr, T, known_at)
            if not hist:
                out[attr] = AttrState(attr, "нет сведений", why="донесений не было")
                continue
            timed = [f for f in hist if not f.t_unknown]
            untimed = [f for f in hist if f.t_unknown]
            if timed:
                latest = timed[-1].t_obs
                current = [f for f in timed if f.t_obs >= latest - WINDOW]
                older = [f for f in timed if f.t_obs < latest - WINDOW]
                # без времени: известно только, что наблюдали не позже получения. Сопоставляем
                # с текущими, если по времени получения оно могло относиться к тому же моменту
                loose = [f for f in untimed if f.t_rec >= latest - WINDOW]
                older += [f for f in untimed if f.t_rec < latest - WINDOW]
                age = T - latest
            else:
                current, older = [], []
                loose = [f for f in untimed if T - f.t_rec <= ind.ttl] or untimed[-1:]
                older = [f for f in untimed if f not in loose]
                age = T - max(f.t_rec for f in untimed)  # не меньше этого
            pairs = [(a, b) for i, a in enumerate(current) for b in current[i + 1:] if self.conflict(a, b)]
            unclear = [Unclear(a, b, w) for i, a in enumerate(current) for b in current[i + 1:]
                       if (w := self.possible(a, b))]
            unclear += [Unclear(u, f, w) for u in loose for f in current if (w := self.possible(u, f))]
            if not current:
                unclear += [Unclear(u, None, ("не указано время наблюдения",)) for u in loose]
            decision = next((d for d in reversed(self._decisions)
                             if d.attr == attr and d.at <= known_at and any(f.rid == d.rid for f in current + loose)), None)
            agreed = any(self.agree(a, b) for i, a in enumerate(current) for b in current[i + 1:])
            jump = None if agreed or pairs else self._jump(timed, older, current)
            if jump:
                unclear.append(Unclear(jump[1], jump[0], ("резко расходится с прежним замером при допустимой "
                                                          "скорости изменения",)))
            fresh = (T - latest <= ind.ttl) if current else (T - max(f.t_rec for f in loose) <= ind.ttl)
            if decision:
                status, why = "принято штабом", f"решение {decision.by} в {decision.at // 60:02d}:{decision.at % 60:02d}"
            elif pairs:
                a, b = pairs[0]
                status, why = "противоречие", f"{a.rid} и {b.rid}: одно время, один участок, разные значения"
            elif unclear:
                status, why = "требует уточнения", "; ".join(sorted({r for u in unclear for r in u.reasons}))
            elif agreed:
                status, why = "согласуется", ("другой отправитель на то же время сообщает совместимое значение; "
                                              "независимость наблюдений не проверена")
            else:
                status, why = "не подтверждено", "нет второго сообщения на то же время"
            out[attr] = AttrState(attr, status, current, older, pairs, age, decision, jump, loose, unclear,
                                  agreed, why, fresh)
        return out

    def _trusted(self, hist: list[Fact]) -> set[int]:
        """Опорные наблюдения: каждое совместимо с предыдущим опорным с учётом
        скорости изменения либо подтверждено независимым источником. Одиночная
        ошибка в опорные не попадает и не тянет за собой следующие значения."""
        trusted: list[Fact] = []
        for f in hist:
            if (not trusted or not self._incompatible(trusted[-1], f)
                    or any(self.agree(f, g) for g in hist if g is not f)):
                trusted.append(f)
        return {f.fid for f in trusted}

    def _incompatible(self, a: Fact, b: Fact) -> bool:
        """b не может следовать за a: слишком быстрое изменение числа
        или невозможный переход категории (прорванный затор снова держится)."""
        if self.other_object(a, b):
            return False
        ind = self.p.ind(a.attr)
        if ind.kind == "num":
            return self.values_differ(a, b)
        if ind.kind == "cat" and ind.monotone:
            return ind.values.index(b.value) < ind.values.index(a.value)
        return False

    def _jump(self, timed: list[Fact], older: list[Fact], current: list[Fact]) -> tuple[Fact, Fact] | None:
        if not current:
            return None
        ind = self.p.ind(current[0].attr)
        if not older or not (ind.kind == "num" or ind.monotone):
            return None
        ok = self._trusted(timed)
        anchor = next((f for f in reversed(older) if f.fid in ok and not f.t_unknown), None)
        if anchor is None:
            return None
        return next(((anchor, f) for f in current if f.fid not in ok and self._incompatible(anchor, f)), None)

    def gaps(self, T: int) -> list[str]:
        return [a for a, s in self.state(T).items() if s.status == "нет сведений" or not s.fresh]

    def trend(self, attr: str, T: int, towards_lower: bool, known_at: int | None = None) -> float:
        """Наибольшая наблюдавшаяся скорость изменения показателя в опасную сторону,
        единиц в минуту, по опорным наблюдениям за последние 3 часа. Используется
        только в проверке «мог ли показатель уже перейти порог», которая выключена
        по умолчанию: это прогноз, а не сведения."""
        known_at = T if known_at is None else known_at
        hist = [f for f in self.history(attr, T, known_at) if not f.t_unknown]
        ok = self._trusted(hist)
        hist = [f for f in hist if f.t_obs >= T - 180 and f.fid in ok]
        rates = []
        for a, b in zip(hist, hist[1:]):
            if b.t_obs - a.t_obs >= 20 and not self.other_object(a, b):
                d = (a.value - b.value) if towards_lower else (b.value - a.value)
                rates.append(d / (b.t_obs - a.t_obs))
        return max([r for r in rates if r > 0], default=0.0)
