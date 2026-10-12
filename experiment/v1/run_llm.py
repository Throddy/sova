"""Повтор эксперимента на локальной модели через Ollama, без интернета.

    ollama pull qwen3:8b                       # один раз, с интернетом
    python experiment/run_llm.py --model qwen3:8b
    python experiment/run.py --answers ollama_qwen3_8b

Модель получает те же подсказки, что и облачная (prompts.py): один запрос на кейс
для «простой LLM» и один запрос на донесение для извлечения фактов. Ответы
сохраняются в answers/ollama_<модель>_baseline.json и _extract.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from prompts import baseline_prompt, extract_prompt  # noqa: E402


def tm(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m


def ask(model: str, prompt: str, host: str) -> dict:
    payload = {"model": model, "stream": False, "format": "json", "think": False, "options": {"temperature": 0},
               "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        text = json.loads(resp.read())["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group()) if m else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()
    cases = json.load(open(HERE / "cases.json", encoding="utf-8"))["cases"]
    tag = "ollama_" + re.sub(r"[^A-Za-z0-9]+", "_", args.model).strip("_")
    baseline, extracted = {}, {}
    for c in cases:
        reps = [r for r in c["reports"] if tm(r["received"]) <= tm(c["T"])]
        baseline[c["id"]] = ask(args.model, baseline_prompt(c, reps), args.host)
        print(c["id"], "сводка готова")
        for r in reps:
            extracted[f"{c['id']}/{r['id']}"] = ask(args.model, extract_prompt(r), args.host)
    out = HERE / "answers"
    (out / f"{tag}_baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{tag}_extract.json").write_text(json.dumps(extracted, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово. Дальше: python experiment/run.py --answers {tag}")


if __name__ == "__main__":
    main()
