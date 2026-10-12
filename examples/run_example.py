"""Сквозной пример: десять донесений о пожаре у п. Кедровый, сводка на 16:00.

python examples/run_example.py

Что делает: читает донесения, раскладывает их в записи памяти, показывает,
что обнаружено, собирает черновик сводки и сравнивает с ожидаемым
результатом из examples/kedroviy/EXPECTED.md, который записан до запуска.
Значения показателей берутся из ручной разметки донесений (поле facts),
время наблюдения и пересказ разбираются из текста кодом. Результат пишется
в examples/kedroviy/OUTPUT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sova.baselines import last_received, majority_vote  # noqa: E402
from sova.extract import find_relay  # noqa: E402
from sova.memory import EvidenceMemory, Fact  # noqa: E402
from sova.profiles import FIRE  # noqa: E402
from sova.profiles.base import hhmm  # noqa: E402
from sova.reasoning import ask_queue, conclusions, needed  # noqa: E402
from sova.summary import cite, hours, render  # noqa: E402
from sova.timeparse import obs_time_quote, parse_obs_time  # noqa: E402

HERE = Path(__file__).parent / "kedroviy"
T = 16 * 60

# То же, что в таблице «Ожидаемая сводка» в EXPECTED.md (записан до запуска)
EXPECTED = {
    "distance_km": "требует уточнения", "area_ha": "не подтверждено", "wind": "не подтверждено", "road": "противоречие",
    "people_in_zone": "не подтверждено", "evacuated": "нет сведений", "personnel": "не подтверждено",
    "equipment": "не подтверждено", "casualties": "не подтверждено",
}
EXPECTED_STALE = {"people_in_zone"}  # остальные показанные сведения актуальны
EXPECTED_NOT_ASKED = {"hotspots", "reserve", "evacuated"}  # не нужны по профилю: не пробел и не вопрос
EXPECTED_QUOTES_OK = True  # каждая цитата дословно есть в оригинале донесения
EXPECTED_NOTES = ["Д-11"]
EXPECTED_RULES = {"distance_km": "нельзя определить до уточнения", "road": "нельзя определить до уточнения"}
EXPECTED_ASKS = [("distance_km", "Глава поселения"), ("road", "ГИБДД, Дорожная служба")]


def to_min(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def notes(rows: list[dict], upto: int | None = None) -> list[tuple[str, str, int, str]]:
    """Донесения, в которых по разметке нет ни одного утверждения по схеме."""
    return [(r["id"], r["source"], to_min(r["received"]), r["text"]) for r in rows
            if not r["facts"] and (upto is None or to_min(r["received"]) <= upto)]


def load(path: Path = HERE / "reports.jsonl") -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def to_facts(rows: list[dict], profile=FIRE, upto: int | None = None) -> list[Fact]:
    """Время наблюдения и пересказ разбирает код, значение, участок и цитату
    берём из ручной разметки (в системе их достаёт модель).
    Донесения, полученные после upto, в память не попадают."""
    facts = []
    for r in rows:
        t_rec = to_min(r["received"])
        if upto is not None and t_rec > upto:
            continue
        t_obs, unknown = parse_obs_time(r["text"], t_rec)
        cited = find_relay(r["text"]) or ""
        observer = r.get("observer", "")  # только если в тексте сказано, кто наблюдал
        for g in r["facts"]:
            v = tuple(g["value"]) if isinstance(g["value"], list) else g["value"]
            facts.append(Fact(len(facts), r["id"], r["source"], g["attr"], v, t_obs, t_rec, t_unknown=unknown,
                              cited=cited, relay_of=r.get("relay_of"), observer=observer,
                              subject=g.get("subject", ""), quote=g.get("quote", ""),
                              t_quote="" if unknown else obs_time_quote(r["text"])))
    return facts


def changes(mem: EvidenceMemory, attr: str, T: int) -> list[tuple[Fact, Fact]]:
    """Пары соседних наблюдений одного участка с известным временем, где значение
    другое, но совместимо с допустимой скоростью изменения: это изменение обстановки."""
    hist = [f for f in mem.history(attr, T) if not f.t_unknown and not f.cited]
    return [(a, b) for a, b in zip(hist, hist[1:])
            if a.value != b.value and not mem.other_object(a, b) and not mem.values_differ(a, b)]


def main() -> None:
    rows = load()
    facts = to_facts(rows, upto=T)
    mem = EvidenceMemory(FIRE)
    for f in facts:
        mem.add(f)
    for n in notes(rows, T):
        mem.add_note(*n)
    for r in rows:
        if to_min(r["received"]) <= T:
            mem.add_original(r["id"], r["source"], to_min(r["received"]), r["text"])
    st = mem.state(T)
    p = FIRE
    out = ["# Сквозной пример: п. Кедровый, сводка на 16:00", "",
           "Иллюстрация на одиннадцати вымышленных донесениях. Значение, участок и цитата взяты из ручной разметки, "
           "время наблюдения и пересказ разобраны кодом. Время местное. Ожидаемый результат в EXPECTED.md "
           "записан до запуска.", "", "## 1. Донесения", "",
           "| № | Получено | Источник | Текст |", "| --- | --- | --- | --- |"]
    out += [f"| {r['id']} | {r['received']} | {r['source']} | {r['text']} |" for r in rows]

    out += ["", "## 2. Записи памяти", "",
            "| Запись | Донесение | Показатель | Значение | Передал | Ссылается на | Наблюдал | Участок | "
            "Наблюдали | Получено |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in facts:
        ind = p.ind(f.attr)
        place = (f.subject or "не указан") if ind.needs_subject else ""
        out.append(f"| З-{f.fid + 1} | {f.rid} | {ind.name} | {ind.fmt(f.value)} | {f.source} | {f.cited} | "
                   f"{f.observer or (f.source + ', от себя' if not f.cited else 'не указан')} | {place} | {'не указано' if f.t_unknown else hhmm(f.t_obs)} | "
                   f"{hhmm(f.t_rec)} |")

    need = needed(mem, T, st)
    out += ["", "## 3. Что обнаружено", ""]
    for attr, s in st.items():
        ind = p.ind(attr)
        name = ind.name
        for a, b in changes(mem, attr, T):
            out.append(f"- {name}: изменение обстановки. {a.rid} на {hhmm(a.t_obs)}: {ind.fmt(a.value)}; "
                       f"{b.rid} на {hhmm(b.t_obs)}: {ind.fmt(b.value)}. Разница укладывается в допустимую "
                       f"скорость изменения (условную).")
        for a, b in s.conflicts:
            out.append(f"- {name}: противоречие. {cite(a)}: {ind.fmt(a.value)}; {cite(b)}: {ind.fmt(b.value)}. "
                       f"Время известно и одно, участок и условия одни, отправители разные. Хранятся обе стороны.")
        grouped: dict[str, tuple[list[str], tuple[str, ...]]] = {}
        for u in s.unclear:
            others, why = grouped.setdefault(u.a.rid, ([], u.reasons))
            if u.b is not None:
                others.append(u.b.rid)
        for rid, (others, why) in grouped.items():
            vs = f" с {', '.join(others)}" if others else ""
            out.append(f"- {name}: требует уточнения. {rid} расходится{vs}, но сопоставить нельзя: {', '.join(why)}.")
        for f in s.shown:
            if f.cited:
                out.append(f"- {name}: {f.rid} передал {f.source}, ссылается на: {f.cited}, кто наблюдал, "
                           f"не сказано. Вторым источником не считается.")
        late = [f for f in s.superseded if s.current and f.t_rec > max(g.t_rec for g in s.current)]
        for f in late:
            out.append(f"- {name}: {f.rid} пришло в {hhmm(f.t_rec)}, позже других, но наблюдение старое "
                       f"({hhmm(f.t_obs)}). Текущее значение не меняет, лежит в истории.")
        if s.shown and not s.fresh:
            out.append(f"- {name}: сведения устарели, условный срок для этого типа {hours(ind.ttl)}. Проверка: «{s.status}».")
        if s.status == "нет сведений" and attr in need:
            out.append(f"- {name}: сведений нет, а по профилю сводки они нужны ({need[attr]}).")

    out += ["", "## 4. Черновик сводки", "", render(mem, T, 15 * 60).replace("# Сводка", "### Сводка", 1)
            .replace("\n### ", "\n##### ").replace("\n## ", "\n#### ")]

    lr, mv = last_received(facts, T), majority_vote(mem, facts, T)
    out += ["## 5. Что дали бы простые способы", "",
            "| Способ | Расстояние | Откуда | Дорога | Откуда |", "| --- | --- | --- | --- | --- |"]
    for name, view in (("Последнее полученное", lr), ("Голосование источников", mv)):
        f, n = view["distance_km"]
        g, m = view["road"]
        conf = (f", голосов: {n}", f", голосов: {m}") if name.startswith("Голос") else ("", "")
        out.append(f"| {name} | {p.ind('distance_km').fmt(f.value)} | {cite(f)}{conf[0]} | {g.value} | "
                   f"{cite(g)}{conf[1]} |")

    out += ["", "## 6. Сравнение с ожидаемым результатом", "", "| Что проверяем | Ожидали | Получилось | Совпало |",
            "| --- | --- | --- | --- |"]
    ok = total = 0

    def row(what, want, got):
        nonlocal ok, total
        total += 1
        ok += want == got
        out.append(f"| {what} | {want} | {got} | {'да' if want == got else 'нет'} |")

    for attr, want in EXPECTED.items():
        row(f"Проверка: {p.ind(attr).name[0].lower() + p.ind(attr).name[1:]}", want, st[attr].status)
    stale = {a for a, s in st.items() if s.shown and not s.fresh}
    row("Устаревшие сведения", ", ".join(sorted(EXPECTED_STALE)), ", ".join(sorted(stale)))
    asked = {a.attr for a in ask_queue(mem, T)}
    row("Не спрашиваем, не нужно по профилю", ", ".join(sorted(EXPECTED_NOT_ASKED)),
        ", ".join(sorted(EXPECTED_NOT_ASKED - asked)))
    row("Не разобрано в поля", ", ".join(EXPECTED_NOTES), ", ".join(n[0] for n in mem.notes(T)))
    bad = [f.rid for f in mem.known(T) if mem.quote_ok(f) is not True]
    row("Все цитаты есть в оригинале", "да" if EXPECTED_QUOTES_OK else "нет", "да" if not bad else "нет: " + ", ".join(bad))
    for c in conclusions(mem, T):
        if c.rule.attr in EXPECTED_RULES:
            row(f"Правило сценария: {c.rule.name[0].lower() + c.rule.name[1:]}", EXPECTED_RULES[c.rule.attr], c.value)
    urgent = [(a.attr, a.whom) for a in ask_queue(mem, T) if a.priority == 1]
    for attr, whom in EXPECTED_ASKS:
        row(f"Срочный вопрос: {p.ind(attr).name[0].lower() + p.ind(attr).name[1:]}", whom,
            next((w for a, w in urgent if a == attr), "нет"))
    out += ["", f"Совпало {ok} из {total}."]
    text = "\n".join(out) + "\n"
    (HERE / "OUTPUT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
