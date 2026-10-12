"""Профиль: авария на котельной в сильный мороз, посёлок связан с райцентром только зимником."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .base import START, Indicator, Profile, Rule, Source

PLACE = "п. Северный"
HOUSES = 180

IND = {i.key: i for i in [
    Indicator("indoor_temp", "Температура в квартирах", unit="°C", tol_abs=3, rate=0.03, ttl=120, critical=True,
              ask="котельная, управляющая компания", phrases=("в квартирах {v}", "температура в домах около {v}")),
    Indicator("houses_cold", "Домов без отопления", tol_abs=10, tol_rel=0.15, rate=0.8, ttl=120, critical=True,
              ask="котельная", phrases=("без отопления домов: {v}",)),
    Indicator("warming_people", "Людей в пунктах обогрева", unit="чел.", tol_abs=10, tol_rel=0.2, rate=2, ttl=120,
              ask="глава поселения", phrases=("в пунктах обогрева {v}",)),
    Indicator("evacuated", "Вывезено в райцентр", unit="чел.", tol_abs=5, tol_rel=0.2, rate=1, ttl=180, critical=True,
              ask="ЕДДС улуса", phrases=("вывезено в райцентр {v}",), zero="в райцентр пока никого не вывозили"),
    Indicator("winter_road", "Зимник до райцентра", kind="cat", values=("открыт", "закрыт"), ttl=180, critical=True,
              ask="дорожная служба, ЕДДС", phrases=("зимник {v}",)),
    Indicator("air_temp", "Температура воздуха", unit="°C", tol_abs=3, rate=0.05, ttl=240, ask="метеостанция",
              phrases=("на улице {v}",)),
    Indicator("casualties", "Пострадавшие (обморожения)", unit="чел.", tol_abs=0, rate=0.02, ttl=240, critical=True,
              ask="фельдшер ФАП", phrases=("обратились с обморожениями {v}",), zero="с обморожениями никто не обращался"),
    Indicator("personnel", "Личный состав", unit="чел.", tol_abs=4, tol_rel=0.25, rate=0.3, ttl=180,
              ask="оперативная группа", phrases=("работают {v}",)),
    Indicator("equipment", "Техника", unit="ед.", tol_abs=2, tol_rel=0.25, rate=0.03, ttl=180, ask="оперативная группа",
              phrases=("техники {v}",)),
    Indicator("fuel_days", "Запас топлива", unit="сут.", ttl=240, ask="котельная"),
    Indicator("reserve", "Резерв сил и средств", unit="ед.", ttl=240, ask="ЕДДС улуса"),
]}


@dataclass
class HeatTruth:
    t0: float
    fall: float
    t_fix: int
    t_close: int
    air: float

    def indoor(self, t: int) -> float:
        h, hf = (t - START) / 60, (self.t_fix - START) / 60
        low = self.t0 - self.fall * min(h, hf)
        return low if h <= hf else low + 1.5 * (h - hf)

    def value(self, attr: str, t: int):
        if attr == "indoor_temp":
            return self.indoor(t)
        if attr == "houses_cold":
            return HOUSES if t < self.t_fix else max(0, HOUSES - 0.7 * (t - self.t_fix))
        if attr == "warming_people":
            return 20 if self.indoor(t) > 13 else min(260, 20 + 1.2 * max(0, (13 - self.indoor(t)) * 40))
        if attr == "evacuated":
            start = 13 * 60
            return 0 if t < start else int(0.4 * (min(t, self.t_close) - start))
        if attr == "winter_road":
            return "открыт" if t < self.t_close else "закрыт"
        if attr == "air_temp":
            return self.air
        if attr == "casualties":
            return 0 if t < 13 * 60 + 30 else 1 if t < 15 * 60 + 30 else 2 if t < 17 * 60 else 3
        if attr == "personnel":
            return 8 if t < 12 * 60 else 16 if t < 15 * 60 else 22
        if attr == "equipment":
            return 2 if t < 12 * 60 else 4 if t < 15 * 60 else 6
        return 0


def make_truth(rng: random.Random) -> HeatTruth:
    return HeatTruth(t0=rng.uniform(15, 17), fall=rng.uniform(1.2, 1.8), t_fix=rng.randint(16 * 60, 17 * 60),
                     t_close=rng.randint(14 * 60, 15 * 60), air=rng.uniform(-48, -42))


GLAVA = f"Глава администрации {PLACE}: "
SOURCES = [
    Source("Котельная", "Дежурный котельной. ", ("indoor_temp", "houses_cold"),
           ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:30", "17:30"), (10, 20),
           {"indoor_temp": 1, "houses_cold": 6}),
    Source("Глава поселения", GLAVA, ("warming_people", "houses_cold", "evacuated"), ("11:10", "13:10", "15:10", "17:10"), (5, 15),
           {"warming_people": 10, "houses_cold": 10}, notime=0.3),
    Source("Оперативная группа", "Оперативная группа МЧС. ", ("indoor_temp", "personnel", "equipment"),
           ("12:15", "14:15", "16:15"), (10, 25), {"indoor_temp": 1}),
    Source("ЕДДС", "ЕДДС улуса. ", ("evacuated", "warming_people"), ("13:30", "15:30", "17:30"), (20, 50),
           {"warming_people": 8}, msk=True, relays=True),
    Source("ЕДДС", "ЕДДС улуса. ", ("winter_road",), ("13:00", "16:00"), (20, 40), msk=True),
    Source("Фельдшер", "Фельдшер ФАП: ", ("casualties",), ("14:00", "16:30"), (10, 30), notime=0.5),
    Source("Дорожная служба", "Дорожная служба: ", ("winter_road",), ("11:30",), (15, 40), events=True),
    Source("Метеостанция", "Метеостанция: ", ("air_temp",), ("10:00", "13:00", "16:00"), (10, 20), {"air_temp": 1}),
    Source("112", "Звонок на 112 от жителя: ", (), ("12:20", "15:40"), (3, 8)),
]

RULES = [
    Rule("Пункты обогрева и вывоз маломобильных", "indoor_temp", "<=", 12, "нужны по плану", "пока не нужны",
         "готовить, решить после уточнения температуры"),
    Rule("Подвоз и вывоз по зимнику", "winter_road", "==", "закрыт", "невозможен, нужна авиация", "возможен",
         "проверить зимник до отправки колонны"),
]

FORMS = [
    ("2/ЧС", "2.1", "air_temp"), ("2/ЧС", "3.1", "casualties"), ("3/ЧС", "3.1", "casualties"),
    ("3/ЧС", "3.4", "warming_people"), ("3/ЧС", "4.1", "evacuated"), ("4/ЧС", "1", "personnel"),
    ("4/ЧС", "1", "equipment"), ("4/ЧС", "4", "reserve"),
]

GROUPS = {
    "Мониторинг обстановки": ("indoor_temp", "houses_cold", "air_temp", "fuel_days"),
    "Защита населения": ("warming_people", "evacuated", "winter_road"),
    "Силы и средства": ("personnel", "equipment", "reserve"),
    "Медицинская группа": ("casualties",),
}

PROFILE = Profile("heat", "Авария на котельной в мороз", PLACE, IND, SOURCES, RULES, FORMS, make_truth, GROUPS,
                  ("в доме холодно, пожилая соседка не выходит на связь", "спрашивают, где пункт обогрева",
                   "соседи говорят, что котельную до завтра не починят, это правда?"))
