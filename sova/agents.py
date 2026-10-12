"""Агенты по группам штаба.

Каждый агент отвечает за свои показатели, как группа штаба отвечает за своё
направление. Агенты не хранят своих копий обстановки и не передают друг
другу факты пересказом: все читают одну доказательную память, и всё, что
агент выдаёт, ссылается на номера донесений. Поэтому вывод одного агента не
может стать «подтверждением» для другого, как пересказ ЕДДС не становится
вторым источником.

Группа управления и взаимодействия собирает сводку, запросы и проекты форм.
В демонстраторе агенты детерминированные. В системе формулировки для людей
пишет локальная модель, а статусы и выводы остаются за правилами.
"""
from __future__ import annotations

from dataclasses import dataclass

from .memory import AttrState, EvidenceMemory
from .reasoning import Ask, Conclusion, ask_queue, conclusions


@dataclass
class GroupAgent:
    name: str
    attrs: tuple[str, ...]

    def situation(self, st: dict[str, AttrState]) -> list[AttrState]:
        return [st[a] for a in self.attrs]

    def decisions(self, cs: list[Conclusion]) -> list[Conclusion]:
        return [c for c in cs if c.rule.attr in self.attrs]

    def asks(self, queue: list[Ask]) -> list[Ask]:
        return [a for a in queue if a.attr in self.attrs]


def staff(mem: EvidenceMemory) -> list[GroupAgent]:
    return [GroupAgent(name, attrs) for name, attrs in mem.p.groups.items()]


def by_group(mem: EvidenceMemory, T: int) -> list[tuple[GroupAgent, list[AttrState], list[Conclusion], list[Ask]]]:
    st, cs, q = mem.state(T), conclusions(mem, T), ask_queue(mem, T)
    return [(g, g.situation(st), g.decisions(cs), g.asks(q)) for g in staff(mem)]
