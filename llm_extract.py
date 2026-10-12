"""Извлечение утверждений из донесений локальной моделью через Ollama.

В run_demo.py утверждения берутся из разметки. Здесь их достаёт из текста модель,
которая работает на компьютере без интернета, а результат сравнивается с
разметкой. Схема та же, что в облачной проверке experiment/v2: показатель,
значение, время наблюдения, на кого ссылается, кто наблюдал, участок и цитата.
Подсказка собирается из профиля ЧС, поэтому скрипт один на все виды ЧС.

Подготовка (один раз, с интернетом):
    ollama pull qwen3:8b
    python run_demo.py            # создаёт учебные донесения в data/
Запуск (интернет не нужен):
    python llm_extract.py --profile fire --model qwen3:8b

Результат: out/llm_<профиль>_<модель>.md с точностью и полнотой извлечения и
ошибками по полям (время, пересказ, цитата), и сводка из извлечённых утверждений.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from sova.extract import find_relay, parse_value
from sova.memory import EvidenceMemory, Fact
from sova.profiles import PROFILES
from sova.profiles.base import COMPASS, RELAY_NAMES, Profile
from sova.summary import render
from sova.timeparse import MSK_SHIFT, obs_time_quote, parse_obs_time, time_quote_for

ROOT = Path(__file__).parent
FIELDS = ("attr", "value", "time", "cited", "observer", "subject", "quote")


def describe(p: Profile) -> str:
    lines = []
    for key, ind in p.indicators.items():
        if ind.kind == "cat":
            what = "одно из: " + ", ".join(f'"{v}"' for v in ind.values)
        elif ind.kind == "wind":
            what = f'"<румб> <скорость>", румб из {", ".join(COMPASS)}, например "СЗ 10"'
        else:
            what = f"число{', ' + ind.unit if ind.unit else ''}"
        lines.append(f"- {key}: {ind.name.lower()}, {what}")
    return "\n".join(lines)


def system_prompt(p: Profile) -> str:
    shift = MSK_SHIFT // 60
    return (f"Ты извлекаешь утверждения из донесения в штаб ликвидации ЧС ({p.title.lower()}).\n"
            "Верни только JSON по схеме. Не добавляй того, чего нет в тексте, ничего не угадывай.\n"
            f"Показатели (attr):\n{describe(p)}\n"
            "Если сказано «пострадавших нет» или «эвакуация не начиналась», это значение 0.\n"
            "Правила заполнения полей:\n"
            f"1. time: время наблюдения ЧЧ:ММ местного времени (МСК+{shift}). Время по Москве (МСК) переводи "
            f"в местное: прибавь {shift} ч. Время отправки донесения («донесение от 15:50») это не время "
            "наблюдения. Если в тексте нет времени наблюдения, пустая строка.\n"
            "2. cited: на кого ссылается отправитель («по данным X», «со слов X», «X сообщил»), иначе пустая "
            "строка. Пересказ не второй независимый источник.\n"
            "3. observer: кто видел сам, только если это сказано прямо, иначе пустая строка.\n"
            "4. subject: участок или объект, к которому относится значение (например «кромка, посёлок»), "
            "если он назван; если сказано только «огонь в 2 км», пустая строка.\n"
            "5. quote: точный фрагмент текста со значением, слово в слово, без добавлений.\n"
            f"Источники для cited и observer: {', '.join(RELAY_NAMES)}.\n"
            "Если в донесении нет таких показателей, верни пустой список.")


def schema(p: Profile) -> dict:
    props = {k: {"type": "string"} for k in FIELDS}
    props["attr"] = {"type": "string", "enum": list(p.indicators)}
    return {"type": "object", "required": ["facts"], "properties": {"facts": {"type": "array", "items": {
        "type": "object", "required": list(FIELDS), "properties": props}}}}


def ask_ollama(p: Profile, model: str, text: str, host: str) -> list[dict]:
    payload = {"model": model, "stream": False, "format": schema(p), "think": False, "options": {"temperature": 0},
               "messages": [{"role": "system", "content": system_prompt(p)}, {"role": "user", "content": text}]}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    return json.loads(body["message"]["content"]).get("facts", [])


def source_name(s: str | None) -> str:
    """Приводит имя источника из ответа модели к справочному («главы администрации» -> «Глава поселения»)."""
    s = (s or "").strip()
    if not s:
        return ""
    low = s.lower()
    for name, gen in RELAY_NAMES.items():
        if name.lower() in low or gen.lower() in low or low in name.lower():
            return name
    if low.startswith("глав"):
        return "Глава поселения"
    return s


def to_minutes(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def model_time(f: dict) -> tuple[int | None, bool]:
    """(время в минутах или None, есть ли поле time в ответе). Поле есть, а времени нет:
    модель сказала, что время не указано. Поля нет: старый формат ответа."""
    t = str(f.get("time") or "").strip()
    if re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", t):
        return to_minutes(t), True
    return None, "time" in f


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)):
        return tuple(a) == tuple(b)
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 0.05
    return a == b


def _norm(t: str) -> str:
    return " ".join(t.split()).lower()


def evaluate(p: Profile, rows: list[dict], extracted: dict[str, list[dict]]) -> dict:
    """Точность и полнота по утверждениям и ошибки по полям на верно найденных утверждениях:
    время наблюдения, пересказ (на кого ссылается) и дословность цитаты.
    Поля считаются, только если модель их вернула."""
    tp = fp = fn = 0
    fld = {"time": [0, 0], "cited": [0, 0], "quote": [0, 0]}
    for row in rows:
        gold = [(g["attr"], g["value"], g.get("observed")) for g in row["gold"]]
        want_cited = find_relay(row["text"]) or ""
        for f in extracted.get(row["id"], []):
            v = parse_value(p, f["attr"], str(f["value"])) if f.get("attr") in p.indicators else None
            hit = next((g for g in gold if v is not None and g[0] == f["attr"] and _same(g[1], v)), None)
            if not hit:
                fp += 1
                continue
            tp += 1
            gold.remove(hit)
            if "time" in f:
                t, _ = model_time(f)
                ok = (t is None and not hit[2]) or (t is not None and hit[2] and t == to_minutes(hit[2]))
                fld["time"][0] += ok
                fld["time"][1] += 1
            if "cited" in f:
                fld["cited"][0] += source_name(f.get("cited")) == want_cited
                fld["cited"][1] += 1
            if "quote" in f:
                q = str(f.get("quote") or "")
                fld["quote"][0] += bool(q) and _norm(q) in _norm(row["text"])
                fld["quote"][1] += 1
        fn += len(gold)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "fields": {k: tuple(v) for k, v in fld.items()}}


def build_memory(p: Profile, rows: list[dict], extracted: dict[str, list[dict]]) -> EvidenceMemory:
    """Время, пересказ, наблюдатель, участок и цитата берутся из ответа модели.
    Если модель вернула только показатель и значение (старый формат), время и пересказ
    разбираются кодом из текста, а цитаты нет. Оригинал каждого донесения сохраняется,
    поэтому дословность цитаты проверяет память (quote_ok)."""
    mem = EvidenceMemory(p)
    fid = 0
    for row in rows:
        t_rec = to_minutes(row["received"])
        mem.add_original(row["id"], row["source"], t_rec, row["text"])
        regex_t, regex_unknown = parse_obs_time(row["text"], t_rec)
        regex_cited = find_relay(row["text"]) or ""
        for f in extracted.get(row["id"], []):
            if f.get("attr") not in p.indicators:
                continue
            v = parse_value(p, f["attr"], str(f["value"]))
            if v is None:
                continue
            t, named = model_time(f)
            if t is not None:
                t_obs, unknown, t_quote = t, False, time_quote_for(row["text"], t)
            elif named:  # модель сказала, что времени нет: не придумываем
                t_obs, unknown, t_quote = t_rec, True, ""
            else:  # старый формат ответа
                t_obs, unknown = regex_t, regex_unknown
                t_quote = "" if unknown else obs_time_quote(row["text"])
            cited = source_name(f.get("cited")) if "cited" in f else regex_cited
            if cited == row["source"]:
                cited = ""
            ind = p.ind(f["attr"])
            if "subject" in f:
                # в учебном наборе все замеры показателя об одном участке, поэтому
                # названный участок приводится к участку профиля; не названный остаётся пустым
                subject = ind.subject if str(f.get("subject") or "").strip() else ""
            else:
                subject = ind.subject
            mem.add(Fact(fid, row["id"], row["source"], f["attr"], v, t_obs, t_rec, t_unknown=unknown, cited=cited,
                         observer=source_name(f.get("observer")), subject=subject,
                         quote=str(f.get("quote") or ""), t_quote=t_quote))
            fid += 1
    return mem


def field_line(name: str, pair: tuple[int, int]) -> str:
    ok, n = pair
    return f"| {name} | {ok} из {n} |" if n else f"| {name} | модель поле не вернула |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="fire", choices=list(PROFILES))
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--corpus", default=None, help="по умолчанию data/<профиль>_seed*.jsonl из run_demo.py")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()
    p = PROFILES[args.profile]
    corpus = Path(args.corpus) if args.corpus else sorted((ROOT / "data").glob(f"{p.key}_seed*.jsonl"))[0]

    rows = [json.loads(line) for line in open(corpus, encoding="utf-8")]
    extracted, started = {}, time.time()
    for row in rows:
        extracted[row["id"]] = ask_ollama(p, args.model, row["text"], args.host)
        print(row["id"], extracted[row["id"]])
    elapsed = time.time() - started
    m = evaluate(p, rows, extracted)
    mem = build_memory(p, rows, extracted)
    facts = mem.known(24 * 60)
    bad_quotes = sum(1 for f in facts if mem.quote_ok(f) is False)

    tag = f"{p.key}_" + args.model.replace(":", "_").replace("/", "_")
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    (out / f"llm_svodka_{tag}_1730.md").write_text(render(mem, 17 * 60 + 30, 16 * 60 + 30), encoding="utf-8")
    report = "\n".join([
        f"# Извлечение утверждений моделью {args.model}, {p.title.lower()}", "",
        f"Донесений: {len(rows)}, время: {elapsed:.0f} с ({elapsed / len(rows):.1f} с на донесение)", "",
        "| Точность | Полнота | F1 | Верно | Лишних | Пропущено |", "| --- | --- | --- | --- | --- | --- |",
        f"| {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['tp']} | {m['fp']} | {m['fn']} |", "",
        "Поля на верно найденных утверждениях:", "", "| Поле | Верно |", "| --- | --- |",
        field_line("время наблюдения", m["fields"]["time"]),
        field_line("на кого ссылается (пересказ)", m["fields"]["cited"]),
        field_line("цитата дословно есть в тексте", m["fields"]["quote"]), "",
        f"Цитат, которых нет в оригинале, по проверке памяти: {bad_quotes} из {len(facts)}.", ""])
    (out / f"llm_{tag}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
