"""Нормализация времени наблюдения из текста донесения.

Время наблюдения берётся из текста, а не из момента получения.
Источники, которые пишут по Москве, переводятся в местное время (МСК+6).
Если времени в тексте нет, время не придумывается: утверждение получает
пометку «время не указано», а для упорядочивания берётся время получения.
Известно только, что наблюдали не позже получения.
"""
from __future__ import annotations

import re

MSK_SHIFT = 6 * 60  # Якутск = МСК+6

TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)(?!\d)")


def parse_obs_time(text: str, t_rec: int, tz_aware: bool = True) -> tuple[int, bool]:
    """Возвращает (время наблюдения в минутах местного времени, не указано ли оно)."""
    m = TIME_RE.search(text)
    if not m:
        return t_rec, True
    t = int(m.group(1)) * 60 + int(m.group(2))
    if tz_aware and "МСК" in text[m.end(): m.end() + 6]:
        t += MSK_SHIFT
    return t, False


def obs_time_quote(text: str) -> str:
    """Фрагмент текста, из которого взято время наблюдения: слово перед временем,
    само время и пометка МСК, если она есть. Пусто, если времени в тексте нет.
    Показывается в карточке основания рядом со временем наблюдения."""
    m = TIME_RE.search(text)
    if not m:
        return ""
    start = text.rfind(" ", 0, max(m.start() - 1, 0)) + 1
    end = m.end()
    msk = text.find("МСК", end, end + 6)
    if msk >= 0:
        end = msk + 3
    return text[start:end]
