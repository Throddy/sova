"""Извлечение фактов из донесений локальной моделью через Ollama.

В run_demo.py факты берутся из разметки. Здесь их достаёт из текста модель,
которая работает на ноутбуке без интернета, а результат сравнивается с
разметкой. Подсказка для модели собирается из профиля ЧС, поэтому скрипт
один на все виды ЧС.

Подготовка (один раз, с интернетом):
    ollama pull qwen3:8b
Запуск (интернет не нужен):
    python llm_extract.py --profile fire --model qwen3:8b

Результат: out/llm_<профиль>_<модель>.md с точностью и полнотой извлечения,
долей верных ссылок и сводкой, собранной из извлечённых фактов.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from sova.extract import find_relay, parse_value
from sova.memory import EvidenceMemory, Fact
from sova.profiles import PROFILES
from sova.profiles.base import COMPASS, Profile
from sova.summary import render
from sova.timeparse import obs_time_quote, parse_obs_time

ROOT = Path(__file__).parent


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
    return (f"Ты извлекаешь факты из донесения в штаб ликвидации ЧС ({p.title.lower()}).\n"
            "Верни только JSON по схеме. Не добавляй фактов, которых нет в тексте, ничего не угадывай.\n"
            f"Показатели (attr):\n{describe(p)}\n"
            "Если сказано «пострадавших нет» или «эвакуация не начиналась», это значение 0.\n"
            "Если в донесении нет таких показателей, верни пустой список.")


def schema(p: Profile) -> dict:
    return {"type": "object", "required": ["facts"], "properties": {"facts": {"type": "array", "items": {
        "type": "object", "required": ["attr", "value"],
        "properties": {"attr": {"type": "string", "enum": list(p.indicators)}, "value": {"type": "string"}}}}}}


def ask_ollama(p: Profile, model: str, text: str, host: str) -> list[dict]:
    payload = {"model": model, "stream": False, "format": schema(p), "think": False, "options": {"temperature": 0},
               "messages": [{"role": "system", "content": system_prompt(p)}, {"role": "user", "content": text}]}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    return json.loads(body["message"]["content"]).get("facts", [])


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)):
        return tuple(a) == tuple(b)
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 0.05
    return a == b


def evaluate(p: Profile, rows: list[dict], extracted: dict[str, list[dict]]) -> dict:
    """Точность и полнота по фактам. Ссылка верна, если в донесении, на которое
    ссылается факт, действительно есть это значение этого показателя, то есть
    доля верных ссылок равна точности извлечения."""
    tp = fp = fn = 0
    for row in rows:
        gold = [(g["attr"], g["value"]) for g in row["gold"]]
        for f in extracted.get(row["id"], []):
            v = parse_value(p, f["attr"], str(f["value"])) if f.get("attr") in p.indicators else None
            hit = next((g for g in gold if v is not None and g[0] == f["attr"] and _same(g[1], v)), None)
            if hit:
                tp += 1
                gold.remove(hit)
            else:
                fp += 1
        fn += len(gold)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def to_minutes(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def build_memory(p: Profile, rows: list[dict], extracted: dict[str, list[dict]]) -> EvidenceMemory:
    mem = EvidenceMemory(p)
    fid = 0
    for row in rows:
        t_rec = to_minutes(row["received"])
        t_obs, unknown = parse_obs_time(row["text"], t_rec)
        cited = find_relay(row["text"]) or ""
        for f in extracted.get(row["id"], []):
            if f.get("attr") not in p.indicators:
                continue
            v = parse_value(p, f["attr"], str(f["value"]))
            if v is not None:
                mem.add(Fact(fid, row["id"], row["source"], f["attr"], v, t_obs, t_rec, t_unknown=unknown, cited=cited,
                             subject=p.ind(f["attr"]).subject, quote=row["text"],
                             t_quote="" if unknown else obs_time_quote(row["text"])))
                fid += 1
    return mem


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

    tag = f"{p.key}_" + args.model.replace(":", "_").replace("/", "_")
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    (out / f"llm_svodka_{tag}_1730.md").write_text(render(mem, 17 * 60 + 30, 16 * 60 + 30), encoding="utf-8")
    report = (f"# Извлечение фактов моделью {args.model}, {p.title.lower()}\n\n"
              f"Донесений: {len(rows)}, время: {elapsed:.0f} с ({elapsed / len(rows):.1f} с на донесение)\n\n"
              "| Точность (доля верных ссылок) | Полнота | F1 | Верно | Лишних | Пропущено |\n"
              "| --- | --- | --- | --- | --- | --- |\n"
              f"| {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['tp']} | {m['fp']} | {m['fn']} |\n")
    (out / f"llm_{tag}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
