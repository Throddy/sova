"""Профиль: ландшафтный пожар идёт к посёлку (сценарий первого этапа из темы)."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .base import START, Indicator, Profile, Rule, Source, SummaryProfile

PLACE = "п. Кедровый"
POP = 350

IND = {i.key: i for i in [
    Indicator("distance_km", "Расстояние от кромки до посёлка", unit="км", decimals=1, tol_abs=0.8, tol_rel=0.2,
              rate=0.02, ttl=120, critical=True, ask="РТП, оператор БПЛА",
              needs_subject=True, subject="кромка, посёлок", matters="от него зависит решение об эвакуации",
              phrases=(f"кромка пожара в {{v}} от {PLACE}", f"до {PLACE} от кромки {{v}}", "фронт огня в {v} от крайних домов")),
    Indicator("area_ha", "Площадь пожара", unit="га", tol_abs=40, tol_rel=0.25, rate=3, ttl=120, ask="Авиалесоохрана",
              phrases=("площадь около {v}", "пройдено огнём {v}")),
    Indicator("wind", "Ветер", kind="wind", ttl=120, ask="метеостанция, РТП", phrases=("ветер {v}",)),
    Indicator("personnel", "Личный состав", unit="чел.", tol_abs=5, tol_rel=0.25, rate=0.5, ttl=180, ask="РТП",
              phrases=("работают {v}", "задействовано л/с {v}")),
    Indicator("equipment", "Техника", unit="ед.", tol_abs=2, tol_rel=0.25, rate=0.03, ttl=180, ask="РТП",
              phrases=("техники {v}", "на тушении {v} техники")),
    Indicator("evacuated", "Эвакуировано", unit="чел.", tol_abs=20, tol_rel=0.2, rate=3, ttl=120, critical=True,
              ask="ЕДДС улуса", phrases=("эвакуировано {v}", "вывезено жителей: {v}"), zero="эвакуация не начиналась"),
    Indicator("people_in_zone", "Люди в посёлке сейчас", unit="чел.", tol_abs=20, tol_rel=0.15, rate=3, ttl=180,
              matters="сколько людей может понадобиться вывезти",
              ask="глава поселения", phrases=("в посёлке остаётся {v}",)),
    Indicator("road", "Дорога на райцентр", kind="cat", values=("открыта", "закрыта"), ttl=180, critical=True,
              needs_subject=True, subject="трасса на райцентр, весь транспорт", matters="от неё зависит путь вывоза",
              ask="ЕДДС улуса, ГИБДД", phrases=("дорога на райцентр {v}", "трасса на райцентр {v}")),
    Indicator("hotspots", "Термоточки ИСДМ", unit="шт.", tol_abs=5, tol_rel=0.3, rate=0.2, ttl=240,
              ask="ИСДМ-Рослесхоз", phrases=("термоточек {v}",)),
    Indicator("casualties", "Пострадавшие", unit="чел.", rate=0.02, ttl=180, critical=True, ask="РТП, ФАП",
              phrases=("пострадавших {v}",), zero="пострадавших нет"),
    Indicator("reserve", "Резерв сил и средств", unit="ед.", ttl=240, ask="ЕДДС улуса"),
]}


@dataclass
class FireTruth:
    d0: float
    v1: float
    v2: float
    a0: float
    g1: float
    g2: float
    t_shift: int
    evac_start: int
    evac_rate: float
    road_close: int

    def value(self, attr: str, t: int):
        h, hs = (t - START) / 60, (self.t_shift - START) / 60
        if attr == "distance_km":
            d = self.d0 - self.v1 * h if h <= hs else self.d0 - self.v1 * hs - self.v2 * (h - hs)
            return max(d, 0.6)
        if attr == "area_ha":
            return self.a0 + self.g1 * h if h <= hs else self.a0 + self.g1 * hs + self.g2 * (h - hs)
        if attr == "wind":
            return ("З", 5) if t < self.t_shift else ("СЗ", 10)
        if attr == "personnel":
            return 12 if t < 12 * 60 + 30 else 24 if t < 15 * 60 + 35 else 39
        if attr == "equipment":
            return 3 if t < 12 * 60 + 30 else 5 if t < 15 * 60 + 35 else 8
        if attr == "evacuated":
            return 0 if t < self.evac_start else int(min(POP - 30, 40 + self.evac_rate * (t - self.evac_start)))
        if attr == "people_in_zone":
            return POP - self.value("evacuated", t)
        if attr == "road":
            return "открыта" if t < self.road_close else "закрыта"
        if attr == "hotspots":
            return int(self.value("area_ha", t) / 25)
        return 0


def make_truth(rng: random.Random) -> FireTruth:
    return FireTruth(d0=rng.uniform(6.0, 7.0), v1=rng.uniform(0.25, 0.4), v2=rng.uniform(0.9, 1.1),
                     a0=rng.uniform(200, 300), g1=rng.uniform(20, 40), g2=rng.uniform(60, 90),
                     t_shift=rng.randint(13 * 60 + 45, 14 * 60 + 45),
                     evac_start=rng.randint(15 * 60 + 20, 15 * 60 + 35), evac_rate=rng.uniform(1.6, 2.2),
                     road_close=rng.randint(15 * 60 + 50, 16 * 60 + 10))


GLAVA = f"Глава администрации {PLACE}: "
SOURCES = [
    Source("РТП", "Донесение РТП. ", ("distance_km", "area_ha", "personnel", "equipment", "casualties"),
           ("10:20", "11:15", "12:20", "13:20", "14:40", "15:30", "16:10", "17:00", "17:40"), (10, 25),
           {"distance_km": 0.15, "area_ha": 25}),
    Source("Глава поселения", GLAVA, ("distance_km", "evacuated", "people_in_zone"), ("12:45", "13:30"), (5, 12),
           {"distance_km": 0.3}, notime=0.3),
    Source("Глава поселения", GLAVA, ("evacuated", "people_in_zone"), ("15:55", "16:35", "17:25"), (5, 15),
           {"evacuated": 10, "people_in_zone": 10}, notime=0.3),
    Source("Глава поселения", GLAVA, ("road",), ("10:35",), (5, 15), notime=0.5),
    Source("ЕДДС", "ЕДДС улуса. ", ("personnel", "equipment"), ("12:00", "14:00", "16:00"), (30, 60), msk=True, relays=True),
    Source("ЕДДС", "ЕДДС улуса. ", ("evacuated",), ("15:40", "16:20", "17:50"), (20, 40), {"evacuated": 5}, msk=True),
    Source("ЕДДС", "ЕДДС улуса. ", ("road",), ("12:00", "16:20"), (30, 50), msk=True),
    Source("ИСДМ", "ИСДМ-Рослесхоз, сеанс спутника. ", ("hotspots", "distance_km"), ("11:10", "13:10", "15:50", "17:20"),
           (60, 110), {"distance_km": 0.5, "hotspots": 2}),
    Source("Авиалесоохрана", "Авиалесоохрана, облёт Ми-8. ", ("distance_km", "area_ha"), ("11:40", "15:20"), (20, 40),
           {"distance_km": 0.15, "area_ha": 25}),
    Source("Метеостанция", "Метеостанция: ", ("wind",), ("11:00", "14:00"), (10, 20)),
    Source("БПЛА", "Оператор БПЛА: ", ("distance_km", "area_ha"), ("14:05", "16:40", "17:45"), (8, 15),
           {"distance_km": 0.1, "area_ha": 15}),
    Source("112", "Звонок на 112 от жителя: ", (), ("14:25", "16:50"), (3, 8)),
    Source("112", "Звонок на 112 от жителя: ", ("road",), ("12:10",), (3, 8), notime=1.0),
    Source("ГИБДД", "ГИБДД: ", ("road",), (), (10, 30), events=True),
]

RULES = [
    # правила учебного сценария; пороги условные, их задаёт Академия
    Rule("Эвакуация, если до кромки 3 км и меньше", "distance_km", "<=", 3.0, "порог достигнут",
         "порог не достигнут", "нельзя определить до уточнения"),
    Rule("Вывоз по дороге на райцентр, если она открыта", "road", "==", "закрыта", "дорога закрыта",
         "дорога открыта", "нельзя определить до уточнения"),
]

FORMS = [
    ("2/ЧС", "1.10", "area_ha"), ("2/ЧС", "2.2", "wind"), ("2/ЧС", "3.1", "casualties"),
    ("3/ЧС", "2.1", "people_in_zone"), ("3/ЧС", "3.1", "casualties"), ("3/ЧС", "4.1", "evacuated"),
    ("4/ЧС", "1", "personnel"), ("4/ЧС", "1", "equipment"), ("4/ЧС", "4", "reserve"),
]

GROUPS = {
    "Мониторинг обстановки": ("distance_km", "area_ha", "wind", "hotspots"),
    "Защита населения": ("evacuated", "people_in_zone", "road"),
    "Силы и средства": ("personnel", "equipment", "reserve"),
    "Медицинская группа": ("casualties",),
}

# Профиль первого этапа. Получатель, назначение, состав и условия предварительные, согласовать с Академией.
# Добавить другой вид ЧС можно новым профилем, но первый этап обещает только этот.
SUMMARY = SummaryProfile(
    "Оперативная сводка: ландшафтный пожар угрожает посёлку",
    "начальник штаба",
    "видеть, где огонь, свободна ли дорога и сколько людей в посёлке, чтобы решать про эвакуацию",
    required=("distance_km", "wind", "road", "people_in_zone", "casualties"),
    conditional=(("evacuated", "если эвакуация объявлена или начата", "reported"),),
    note=("Пострадавшие: три состояния, сообщается о пострадавших, сообщается об их отсутствии, сведений нет; "
          "отсутствие сообщений не значит, что пострадавших нет. До объявления эвакуации могут понадобиться "
          "сведения о готовности: транспорт, пункты размещения."),
)

PROFILE = Profile("fire", "Ландшафтный пожар у посёлка", PLACE, IND, SOURCES, RULES, FORMS, make_truth, GROUPS,
                  ("сильный дым, просят разъяснить, что делать", "просят прислать автобус для пожилых жителей",
                   "соседи говорят, что скоро всех вывезут, это правда?"), SUMMARY)
