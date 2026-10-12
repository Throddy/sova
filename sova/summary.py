"""Черновики для штаба: сводка, проекты форм, запросы и разбор.

У каждой строки указано, из каких донесений она взята.
"""
from __future__ import annotations

from . import forms
from .memory import AttrState, EvidenceMemory, Fact
from .profiles.base import MSK_SHIFT, hhmm
from .reasoning import ask_queue, conclusions, draft_request, needed


def low(name: str) -> str:
    return name[0].lower() + name[1:]


def hours(minutes: int) -> str:
    """Срок актуальности в часах: 120 даёт «2 ч», 90 даёт «1,5 ч»."""
    return f"{minutes // 60} ч" if minutes % 60 == 0 else f"{minutes / 60:.1f} ч".replace(".", ",")


def obs_time_text(f: Fact) -> str:
    """Время наблюдения вместе с фрагментом текста, из которого его взял код."""
    if f.t_unknown:
        return "в тексте не указано; известно только, что не позже получения"
    return f"{hhmm(f.t_obs)}, в тексте: «{f.t_quote}»" if f.t_quote else hhmm(f.t_obs)


def cite(f: Fact) -> str:
    who = f.source if not f.cited else f"{f.source} со слов: {f.cited}"
    t = "время не указано" if f.t_unknown else f"набл. {hhmm(f.t_obs)}"
    return f"{f.rid} ({who}, {t}, получ. {hhmm(f.t_rec)})"


def observer_text(f: Fact) -> str:
    """Кто наблюдал: только то, что следует из текста или метаданных."""
    if f.observer:
        return f"{f.observer} (сказано в тексте)"
    if f.cited:
        return "в тексте не указан"
    return "в тексте не указан; ссылок на других нет, считается наблюдением отправителя"


def chain(mem: EvidenceMemory, f: Fact) -> str:
    """Цепочка основания одной строкой: донесение, кто передал, на кого ссылается,
    кто наблюдал, участок, цитата и её проверка, время наблюдения и получения."""
    ind = mem.p.ind(f.attr)
    parts = [f.rid, f"передал: {f.source}"]
    if f.cited:
        parts.append(f"ссылается на: {f.cited}")
    parts.append(f"наблюдал: {observer_text(f)}")
    if ind.needs_subject:
        parts.append(f"участок: {f.subject or 'не указан'}")
    parts.append(f"цитата: «{f.quote}»" if f.quote else "цитата: нет")
    ok = mem.quote_ok(f)
    if ok is not None:
        parts.append("цитата есть в оригинале" if ok else "цитаты НЕТ в оригинале")
    parts.append(f"наблюдали: {obs_time_text(f)}")
    parts.append(f"получено: {hhmm(f.t_rec)}")
    return " · ".join(parts)


def fact_check(mem: EvidenceMemory, f: Fact, s: AttrState) -> str:
    """Состояние проверки одного утверждения, а не показателя в целом."""
    from .reasoning import _needs
    if any(f in pair for pair in s.conflicts):
        other = next(g for pair in s.conflicts if f in pair for g in pair if g is not f)
        return f"противоречие с [{other.rid}]: одно время, участок и условия, значения несовместимы"
    if s.decision and s.decision.rid == f.rid:
        return f"принято штабом ({s.decision.by})"
    missing = _needs(mem, f)
    if missing and any(f in (u.a, u.b) for u in s.unclear):
        return "требует уточнения: не указано " + ", ".join(missing)
    if s.jump and s.jump[1] is f:
        return "требует уточнения: резко расходится с прежним замером"
    agree = [g for g in s.current if g is not f and mem.agree(f, g)]
    if agree:
        return f"согласуется с [{agree[0].rid}]; независимость наблюдений не проверена"
    earlier = [g for g in s.current if g is not f and g.t_obs < f.t_obs and not mem.values_differ(f, g)]
    note = (f"; [{earlier[-1].rid}] на {hhmm(earlier[-1].t_obs)} не противоречит, но и не подтверждает"
            if earlier else "")
    return "не подтверждено: на это время нет второго сообщения" + note


def card(mem: EvidenceMemory, f: Fact, s: AttrState) -> list[str]:
    """Раскрытая карточка основания одного утверждения."""
    ind = mem.p.ind(f.attr)
    ok = mem.quote_ok(f)
    lines = [f"**{ind.name}: {ind.fmt(f.value)}** [{f.rid}]", "",
             f"- Донесение: {f.rid}, получено {hhmm(f.t_rec)}",
             f"- Цитата: «{f.quote}»" + ("" if ok is None else (" (есть в оригинале, проверено кодом)" if ok
                                                               else " (НЕТ в оригинале)")),
             f"- Передал: {f.source}" + (f"; ссылается на: {f.cited}" if f.cited else ""),
             f"- Наблюдал: {observer_text(f)}"]
    if ind.needs_subject:
        lines.append(f"- Участок: {f.subject or 'не указан'}")
    lines += [f"- Время наблюдения: {obs_time_text(f)}",
              f"- Проверка: {fact_check(mem, f, s)}",
              f"- Актуальность: {'актуально' if s.fresh else 'устарело'}, условный срок для этого типа {hours(ind.ttl)}",
              "- Соответствует ли утверждение цитате, проверяет оператор", ""]
    return lines


def value_of(mem: EvidenceMemory, s: AttrState) -> str:
    if not s.shown:
        return "нет сведений"
    return forms._value(mem, s)


def status_of(s: AttrState) -> str:
    if s.decision:
        return f"принято штабом в {hhmm(s.decision.at)} ({s.decision.by})"
    return s.status


def freshness(mem: EvidenceMemory, s: AttrState) -> str:
    """Актуальность отдельно от проверки: когда наблюдали и не истёк ли срок для этого типа сведений."""
    if not s.shown:
        return ""
    ttl = mem.p.ind(s.attr).ttl
    if s.current:
        last = max(f.t_obs for f in s.current)
        when = f"набл. {hhmm(last)}"
    else:
        when = f"время не указано, получ. {hhmm(max(f.t_rec for f in s.loose))}"
    return f"актуально: {when}, срок {hours(ttl)}" if s.fresh else f"устарело: {when}, срок {hours(ttl)}"


def _msk(T: int) -> str:
    return f"{hhmm(T)} ({hhmm(T - MSK_SHIFT)} МСК)"


def _since(f: Fact) -> str:
    return "время наблюдения не указано" if f.t_unknown else f"на {hhmm(f.t_obs)}"


def brief(mem: EvidenceMemory, T: int, st: dict[str, AttrState], need: dict[str, str]) -> tuple[list[str], list[Fact]]:
    """Краткая обстановка: последние известные наблюдения по профилю, по одному
    предложению, ссылка рядом с утверждением. Собирается шаблоном, без модели."""
    out, used = [], []
    for attr in need:
        s = st[attr]
        if not s.current or not s.fresh or s.status == "противоречие":
            continue
        ind = mem.p.ind(attr)
        f = max(s.current, key=lambda x: (x.t_obs, x.fid))
        text = f"{ind.name}: {ind.fmt(f.value)} {_since(f)} [{f.rid}]"
        if ind.zero and f.value == 0:
            text = f"{ind.zero[0].upper() + ind.zero[1:]} по донесению {_since(f)} [{f.rid}]"
        if s.status == "согласуется":
            other = next(g for g in s.current if g is not f and mem.agree(f, g))
            text += f", совпадает с [{other.rid}]"
        elif s.status in ("не подтверждено", "требует уточнения"):
            text += ", других сообщений на это время нет"
        if s.status == "требует уточнения":
            text += "; есть сведения, требующие уточнения"
        text += f"; актуально, срок {hours(ind.ttl)}"
        out.append(text + ".")
        used.append(f)
    return out, used


def attention(mem: EvidenceMemory, T: int, need: dict[str, str]) -> tuple[list[str], list[Fact]]:
    """Требует внимания: что неясно, почему это важно для сводки, что уточнить
    и у кого, в одном пункте. Плюс сообщения вне текущей схемы."""
    p = mem.p
    out, used = [], []
    for a in ask_queue(mem, T):
        ind = p.ind(a.attr)
        refs = "[" + ", ".join(dict.fromkeys(f.rid for f in a.facts)) + "]" if a.facts else ""
        why = f" Важно: {ind.matters}." if ind.matters else ""
        if a.state.status == "противоречие":
            subj = a.facts[0].subject if a.facts and a.facts[0].subject else ""
            what = (f"{ind.name}{', ' + subj if subj else ''}: отправители сообщают противоположное на одно время "
                    f"{refs}.{why} Уточнить проходимость сейчас, кто и где проверял. Кому вопрос: {a.whom}.")
        elif a.kind == "detail":
            rest = [g.rid for u in a.state.unclear for g in (u.a, u.b) if g is not None and g not in a.facts]
            vs = "; ".join(f"«{f.quote or ind.fmt(f.value)}» [{f.rid}]" for f in a.facts)
            what = (f"{ind.name}: {vs}. Возможно, о том же"
                    + (f", что [{', '.join(dict.fromkeys(rest))}]" if rest else "")
                    + f", но сопоставить нельзя: не указано {', '.join(a.need)}.{why} Кому вопрос: {a.whom}.")
        elif a.state.status == "нет сведений":
            what = f"{ind.name}: сведений нет, а по профилю они нужны ({need.get(a.attr, 'обязательно')}).{why} Кому вопрос: {a.whom}."
        elif not a.state.fresh:
            what = (f"{ind.name}: {a.problem} {refs}, условный срок для этого типа {hours(ind.ttl)}.{why} "
                    f"Запросить на сейчас, кому вопрос: {a.whom}.")
        else:
            what = f"{ind.name}: {a.problem} {refs}.{why} Кому вопрос: {a.whom}."
        out.append(what)
        used += [f for f in a.facts if f not in used]
    for rid, src, t, text in mem.notes(T):
        out.append(f"Вне текущей схемы [{rid}], {src}, получено {hhmm(t)}: «{text.rstrip('.')}». Рассмотреть офицеру.")
    return out, used


def render(mem: EvidenceMemory, T: int, prev_T: int | None = None, drift: bool = False) -> str:
    """Черновик сводки собирает шаблон, модель текст не пишет: краткая обстановка,
    «требует внимания» и карточки оснований строятся из записей памяти, ссылка
    стоит рядом с каждым утверждением. Полная таблица в конце, это режим «Все сведения»."""
    p = mem.p
    st = mem.state(T)
    need = needed(mem, T, st)
    out = [f"# Сводка на {_msk(T)}. {p.title}, {p.place}", "",
           "Черновик. Готовит СОВА, проверяет и подписывает штаб. Время местное, МСК+6.", ""]
    if p.summary:
        sp = p.summary
        out += [f"Профиль: {sp.name}. Получатель: {sp.reader}. Назначение: {sp.purpose}. "
                "Состав и условия предварительные, согласуются с Академией.", ""]
        terms = list(sp.required) + [c[0] for c in sp.conditional]
        out += ["Сроки актуальности условные, согласуются с Академией: "
                + ", ".join(f"{low(p.ind(a).name)} {hours(p.ind(a).ttl)}" for a in terms) + ".", ""]

    lines, used_b = brief(mem, T, st, need)
    out += ["## Краткая обстановка", ""] + ([" ".join(lines)] if lines else ["Подтверждённых свежих сведений нет."]) + [""]

    items, used_a = attention(mem, T, need)
    out += ["## Требует внимания", "", "Вопросы готовит СОВА, офицер проверяет и отправляет.", ""]
    out += [f"{i}. {x}" for i, x in enumerate(items, 1)] or ["Ничего."]
    out.append("")

    out += ["## Основания", ""]
    for f in sorted(dict.fromkeys(used_b + used_a), key=lambda f: f.fid):
        out += card(mem, f, st[f.attr])

    out += ["## К плану действий: правила учебного сценария", "",
            "Не рекомендации СОВЫ: что даёт правило из замысла учения на текущих сведениях и можно ли опираться "
            "на его входные данные. Пороги условные, их задаёт Академия.", "",
            "| Правило сценария | Что даёт на текущих сведениях | Основание | Входные данные |",
            "| --- | --- | --- | --- |"]
    for c in conclusions(mem, T, use_drift=drift):
        basis = ", ".join(dict.fromkeys(f.rid for f in sorted(c.basis, key=lambda f: f.fid))) or "нет"
        out.append(f"| {c.rule.name} | {c.value} | {basis} | {c.recheck or 'надёжны'} |")

    out += ["", "## Все сведения", "", "Дополнительный режим: полная таблица по показателям.", "",
            "| Показатель | Сведения | Основание | Актуальность | Проверка |", "| --- | --- | --- | --- | --- |"]
    order = list(need) + [a for a in p.indicators if a not in need and st[a].shown]
    for attr in order:
        s = st[attr]
        name = p.ind(attr).name + ("" if attr in need else " (дополнительно)")
        out.append(f"| {name} | {value_of(mem, s)} | {', '.join(f.rid for f in s.shown) or 'нет'} | "
                   f"{freshness(mem, s) or 'нет сведений'} | {status_of(s)} |")
    skipped = [low(p.ind(a).name) for a in p.indicators if a not in order]
    if skipped:
        out += ["", "Не требуются профилем и не поступали: " + ", ".join(skipped) + "."]
    if p.summary and p.summary.note:
        out += ["", f"Обсудить с Академией: {p.summary.note}"]
    out += ["", f"Утверждений в памяти: {len(mem.known(T))}, ничего не удалено. Проекты форм 2/ЧС–4/ЧС в первый этап не входят."]
    return "\n".join(out) + "\n"


def render_forms(mem: EvidenceMemory, T: int) -> str:
    p = mem.p
    fields = forms.build(mem, T)
    ready, check, empty = forms.readiness(fields)
    out = [f"# Проект форм 2/ЧС–4/ЧС на {_msk(T)}. {p.title}, {p.place}", "",
           "Черновик. Заполнены только пункты, для которых в профиле ЧС задано соответствие показателю. "
           "Подписывает и отправляет штаб.", "",
           f"Готово {ready} из {len(fields)}, проверить {check}, пустых {empty}.", "",
           "| Форма | Пункт | Показатель | Значение | Статус | Донесения | Что сделать до подачи |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for f in fields:
        out.append(f"| {f.form} | {f.item} | {f.name} | {f.value or 'нет сведений'} | {f.status} | "
                   f"{', '.join(x.rid for x in f.basis)} | {f.todo} |")
    prev = forms.last_deadline(T - 1)
    if prev is not None and mem.known(prev):
        ch = forms.changed_since(mem, prev, T)
        out += ["", f"## Изменилось с подачи в {hhmm(prev)}", ""]
        out += [f"- {b.form} п. {b.item}, {low(b.name)}: было {a.value or 'пусто'} ({a.status}), "
                f"стало {b.value or 'пусто'} ({b.status})" for a, b in ch] or ["Ничего."]
    return "\n".join(out) + "\n"


def render_requests(mem: EvidenceMemory, T: int) -> str:
    out = [f"# Проекты запросов на {_msk(T)}", "", "Отправляет человек. Порядок по важности для решений.", ""]
    for i, a in enumerate(ask_queue(mem, T), 1):
        out += [f"{i}. {'Срочно. ' if a.priority == 1 else ''}{draft_request(mem, a, T)}", ""]
    return "\n".join(out)


def render_review(mem: EvidenceMemory, checkpoints: list[int], T_now: int, truth=None) -> str:
    """Разбор: что штаб знал в момент решения и что известно о том же моменте теперь.
    Если есть замысел учения, рядом ставится, как было на самом деле."""
    p = mem.p
    out = [f"# Разбор. {p.title}, {p.place}", "",
           f"Для каждого момента два среза одной памяти: что было у штаба тогда и что стало известно "
           f"о том же моменте к {hhmm(T_now)}, когда пришли запоздавшие донесения."
           + (" Последняя колонка из замысла учения." if truth else ""), "",
           "| Время | Решение | Штаб видел тогда | Известно теперь | " + ("По замыслу | " if truth else "") + "Оценка |",
           "| --- | --- | --- | --- | " + ("--- | " if truth else "") + "--- |"]
    for T in checkpoints:
        then = {c.rule.name: c for c in conclusions(mem, T)}
        now = {c.rule.name: c for c in conclusions(mem, T, known_at=T_now)}
        for rule in p.rules:
            a, b = then[rule.name], now[rule.name]
            fa = f"{a.value} ({', '.join(sorted({f.rid for f in a.basis})) or 'нет сведений'})"
            fb = f"{b.value} ({', '.join(sorted({f.rid for f in b.basis})) or 'нет сведений'})"
            row = f"| {hhmm(T)} | {rule.name} | {fa} | {fb} | "
            verdict = "без изменений" if a.value == b.value else "картина поменялась после поступления запоздавших донесений"
            if truth:
                real = rule.danger if rule.is_danger(truth.value(rule.attr, T)) else rule.safe
                row += f"{real} | "
                if a.value == real:
                    verdict = "верно"
                elif a.recheck:
                    verdict = "не ясно, но помечено «пересмотреть»"
                else:
                    verdict = "ошибка без пометки"
            out.append(row + verdict + " |")
    late = [f for f in mem.known(T_now) if any(f.t_obs <= T < f.t_rec for T in checkpoints)
            and f.attr in {r.attr for r in p.rules}]
    if late:
        out += ["", "## Донесения, которые опоздали к решению", ""]
        seen = set()
        for f in sorted(late, key=lambda f: f.t_obs):
            if f.rid in seen:
                continue
            seen.add(f.rid)
            out.append(f"- {cite(f)}: {low(p.ind(f.attr).name)} {p.ind(f.attr).fmt(f.value)}, "
                       f"шло до штаба {f.t_rec - f.t_obs} мин")
    return "\n".join(out) + "\n"
