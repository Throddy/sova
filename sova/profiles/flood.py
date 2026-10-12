"""Профиль: затор льда на реке и подтопление села (весенний ледоход в Якутии)."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .base import START, Indicator, Profile, Rule, Source

PLACE = "с. Приречное"
POP = 420
CRITICAL_LEVEL = 850  # условная опасная отметка поста, см

IND = {i.key: i for i in [
    Indicator("water_level", "Уровень воды у поста", unit="см", tol_abs=15, rate=0.8, ttl=90, critical=True,
              ask="гидропост, оперативная группа", phrases=("уровень воды у поста {v}", "уровень реки у села {v}")),
    Indicator("flooded_houses", "Подтоплено жилых домов", tol_abs=5, tol_rel=0.2, rate=0.5, ttl=120, critical=True,
              ask="глава поселения, оперативная группа", phrases=("подтоплено домов: {v}", "вода зашла во дворы, домов: {v}")),
    Indicator("ice_jam", "Затор ниже села", kind="cat", values=("держится", "прорван"), monotone=True, ttl=180, critical=True,
              ask="оператор БПЛА, авиаразведка", phrases=("затор ниже села {v}",)),
    Indicator("evacuated", "Эвакуировано", unit="чел.", tol_abs=20, tol_rel=0.2, rate=3, ttl=120, critical=True,
              ask="ЕДДС улуса", phrases=("эвакуировано {v}", "вывезено в ПВР {v}"), zero="эвакуация не начиналась"),
    Indicator("people_in_zone", "Население в зоне ЧС", unit="чел.", tol_abs=20, tol_rel=0.15, rate=3, ttl=180,
              ask="глава поселения", phrases=("в зоне подтопления остаётся {v}",)),
    Indicator("road", "Дорога на райцентр", kind="cat", values=("открыта", "закрыта"), ttl=180, critical=True,
              ask="ЕДДС улуса, дорожная служба", phrases=("дорога на райцентр {v}",)),
    Indicator("air_temp", "Температура воздуха", unit="°C", tol_abs=3, rate=0.05, ttl=240, ask="метеостанция",
              phrases=("температура воздуха {v}",)),
    Indicator("personnel", "Личный состав", unit="чел.", tol_abs=5, tol_rel=0.25, rate=0.5, ttl=180,
              ask="оперативная группа", phrases=("на месте работают {v}",)),
    Indicator("equipment", "Техника", unit="ед.", tol_abs=2, tol_rel=0.25, rate=0.03, ttl=180, ask="оперативная группа",
              phrases=("техники и плавсредств {v}",)),
    Indicator("casualties", "Пострадавшие", unit="чел.", rate=0.02, ttl=240, critical=True, ask="ФАП, ЕДДС",
              phrases=("пострадавших {v}",), zero="пострадавших нет"),
    Indicator("reserve", "Резерв сил и средств", unit="ед.", ttl=240, ask="ЕДДС улуса"),
]}


@dataclass
class FloodTruth:
    l0: float
    r1: float
    r2: float
    t_jam: int
    t_break: int
    evac_rate: float

    def level(self, t: int) -> float:
        h = (t - START) / 60
        hj, hb = (self.t_jam - START) / 60, (self.t_break - START) / 60
        if h <= hj:
            return self.l0 + self.r1 * h
        peak = self.l0 + self.r1 * hj + self.r2 * (min(h, hb) - hj)
        return peak if h <= hb else peak - 35 * (h - hb)

    def evac_start(self) -> int:
        for t in range(START, 24 * 60):
            if self.level(t) >= CRITICAL_LEVEL - 10:
                return t
        return 24 * 60

    def value(self, attr: str, t: int):
        if attr == "water_level":
            return self.level(t)
        if attr == "flooded_houses":
            return max(0.0, (self.level(t) - 800) * 0.9)
        if attr == "ice_jam":
            return "прорван" if t >= self.t_break else "держится"
        if attr == "evacuated":
            s = self.evac_start()
            return 0 if t < s else int(min(POP - 20, 30 + self.evac_rate * (t - s)))
        if attr == "people_in_zone":
            return POP - self.value("evacuated", t)
        if attr == "road":
            # дорогу закрывают, когда её заливает, и до конца дня не открывают
            return "закрыта" if any(self.level(x) >= 830 for x in range(START, t + 1, 5)) else "открыта"
        if attr == "air_temp":
            return 4 + 3 * (t - START) / 480
        if attr == "personnel":
            return 15 if t < 13 * 60 else 30 if t < 15 * 60 + 30 else 45
        if attr == "equipment":
            return 4 if t < 13 * 60 else 7 if t < 15 * 60 + 30 else 10
        return 0


def make_truth(rng: random.Random) -> FloodTruth:
    return FloodTruth(l0=rng.uniform(730, 760), r1=rng.uniform(5, 9), r2=rng.uniform(24, 32),
                      t_jam=rng.randint(12 * 60 + 30, 13 * 60 + 30), t_break=rng.randint(16 * 60 + 30, 17 * 60 + 30),
                      evac_rate=rng.uniform(1.5, 2.2))


GLAVA = f"Глава администрации {PLACE}: "
SOURCES = [
    Source("Гидропост", "Гидропост Росгидромета. ", ("water_level",),
           ("10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00"), (5, 15), {"water_level": 2}),
    Source("Оперативная группа", "Оперативная группа МЧС. ", ("water_level", "flooded_houses", "personnel", "equipment", "casualties"),
           ("10:40", "12:10", "13:40", "15:10", "16:40"), (10, 25), {"water_level": 10, "flooded_houses": 3}),
    Source("Глава поселения", GLAVA, ("flooded_houses", "evacuated", "people_in_zone"), ("14:30", "15:30", "16:30", "17:30"),
           (5, 15), {"flooded_houses": 6, "evacuated": 10, "people_in_zone": 10}, notime=0.3),
    Source("Глава поселения", GLAVA, ("road", "evacuated", "people_in_zone"), ("11:00", "13:30"), (5, 15), notime=0.5),
    Source("Глава поселения", GLAVA, ("road",), ("15:45",), (5, 15), notime=0.5),
    Source("ЕДДС", "ЕДДС улуса. ", ("evacuated", "people_in_zone"), ("15:00", "16:00", "17:00"), (20, 50),
           {"evacuated": 8, "people_in_zone": 8}, msk=True, relays=True),
    Source("ЕДДС", "ЕДДС улуса. ", ("personnel", "equipment"), ("12:30", "15:30"), (30, 60), msk=True),
    Source("БПЛА", "Оператор БПЛА: ", ("ice_jam", "flooded_houses"), ("12:00", "14:30", "17:15"), (8, 15), {"flooded_houses": 3}),
    Source("Авиаразведка", "Авиаразведка, вертолёт Ми-8: ", ("ice_jam",), ("13:00", "17:40"), (20, 40)),
    Source("Метеостанция", "Метеостанция: ", ("air_temp",), ("11:00", "14:00"), (10, 20)),
    Source("112", "Звонок на 112 от жителя: ", (), ("13:20", "16:10"), (3, 8)),
    Source("Дорожная служба", "Дорожная служба: ", ("road",), (), (10, 30), events=True),
]

RULES = [
    Rule("Эвакуация села", "water_level", ">=", CRITICAL_LEVEL, "нужна по плану", "пока не нужна",
         "готовить, решить после уточнения уровня"),
    Rule("Вывоз по дороге на райцентр", "road", "==", "закрыта", "невозможен, нужны плавсредства", "возможен",
         "проверить дорогу до начала вывоза"),
]

FORMS = [
    ("2/ЧС", "2.1", "air_temp"), ("2/ЧС", "3.1", "casualties"), ("3/ЧС", "2.1", "people_in_zone"),
    ("3/ЧС", "3.1", "casualties"), ("3/ЧС", "4.1", "evacuated"), ("4/ЧС", "1", "personnel"),
    ("4/ЧС", "1", "equipment"), ("4/ЧС", "4", "reserve"),
]

GROUPS = {
    "Мониторинг обстановки": ("water_level", "ice_jam", "flooded_houses", "air_temp"),
    "Защита населения": ("evacuated", "people_in_zone", "road"),
    "Силы и средства": ("personnel", "equipment", "reserve"),
    "Медицинская группа": ("casualties",),
}

PROFILE = Profile("flood", "Затор льда и подтопление села", PLACE, IND, SOURCES, RULES, FORMS, make_truth, GROUPS,
                  ("вода подходит к дому, просят лодку", "спрашивают, куда вывозят и что брать с собой",
                   "соседи говорят, что затор будут взрывать, это правда?"))
