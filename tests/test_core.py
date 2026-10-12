import socket

import pytest

from sova import forms
from sova.baselines import last_received, majority_vote
from sova.evaluate import run_seed
from sova.extract import find_relay
from sova.memory import EvidenceMemory, Fact
from sova.profiles import FIRE, FLOOD, HEAT, PROFILES
from sova.profiles.base import RELAY_NAMES, generate
from sova.reasoning import ask_queue, conclusions, draft_request
from sova.timeparse import obs_time_quote, parse_obs_time

H = 60


EDGE = "кромка, посёлок"


def fact(fid, rid, source, attr, value, t_obs, t_rec, cited="", subject=EDGE, unknown=False, observer=""):
    """Утверждение для тестов. Если времени нет, t_obs игнорируется и берётся время получения."""
    return Fact(fid, rid, source, attr, value, t_rec if unknown else t_obs, t_rec, t_unknown=unknown,
                cited=cited, observer=observer, subject=subject, quote=f"{attr} {value}")


def memory(profile, *facts):
    mem = EvidenceMemory(profile)
    for f in facts:
        mem.add(f)
    return mem


def evac(mem, T, **kw):
    return next(c for c in conclusions(mem, T, **kw) if c.rule.attr in ("distance_km", "water_level", "indoor_temp"))


def test_nothing_is_deleted():
    mem = memory(FIRE, *[fact(i, f"Д-{i}", "РТП", "distance_km", 5.0 - i * 0.1, 10 * H + i * 30, 10 * H + i * 30 + 10)
                         for i in range(5)])
    assert len(mem.known(20 * H)) == 5


def test_late_old_observation_does_not_replace_fresh_one():
    fresh = fact(0, "Д-1", "РТП", "distance_km", 3.0, 14 * H, 14 * H + 10)
    late = fact(1, "Д-2", "ИСДМ", "distance_km", 6.0, 12 * H, 14 * H + 30)  # наблюдение старше, пришло позже
    st = memory(FIRE, fresh, late).state(14 * H + 40)["distance_km"]
    assert [f.rid for f in st.current] == ["Д-1"]
    assert [f.rid for f in st.superseded] == ["Д-2"]
    assert last_received([fresh, late], 14 * H + 40)["distance_km"][0].rid == "Д-2"


def test_contradiction_keeps_both_sides_and_flags_rule_input():
    mem = memory(FIRE, fact(0, "Д-1", "РТП", "distance_km", 5.0, 13 * H + 20, 13 * H + 40),
                 fact(1, "Д-2", "Глава поселения", "distance_km", 1.5, 13 * H + 30, 13 * H + 40))
    st = mem.state(13 * H + 45)["distance_km"]
    assert st.status == "противоречие"
    assert {f.rid for pair in st.conflicts for f in pair} == {"Д-1", "Д-2"}
    assert evac(mem, 13 * H + 45).recheck


def test_natural_change_is_not_a_contradiction_and_not_a_confirmation():
    mem = memory(FIRE, fact(0, "Д-1", "РТП", "distance_km", 5.0, 14 * H, 14 * H + 10),
                 fact(1, "Д-2", "БПЛА", "distance_km", 4.6, 14 * H + 25, 14 * H + 30))
    s = mem.state(14 * H + 40)["distance_km"]
    assert s.status == "не подтверждено" and not s.conflicts and not s.unclear  # раннее значение позднее не подтверждает


def test_same_time_compatible_reports_agree_but_independence_is_not_claimed():
    mem = memory(FIRE, fact(0, "Д-1", "РТП", "distance_km", 4.1, 15 * H + 10, 15 * H + 22),
                 fact(1, "Д-2", "Авиалесоохрана", "distance_km", 4.0, 15 * H + 15, 15 * H + 30))
    s = mem.state(15 * H + 40)["distance_km"]
    assert s.status == "согласуется" and "не проверена" in s.why


def test_freshness_is_separate_from_check():
    mem = memory(FIRE, fact(0, "Д-1", "Глава поселения", "people_in_zone", 350, 10 * H + 30, 10 * H + 42))
    s = mem.state(16 * H)["people_in_zone"]
    assert s.status == "не подтверждено" and not s.fresh


def test_relay_is_not_independent_confirmation():
    orig = fact(0, "Д-1", "Глава поселения", "distance_km", 2.0, 13 * H + 30, 13 * H + 35)
    relay = fact(1, "Д-2", "ЕДДС", "distance_km", 2.0, 13 * H + 30, 13 * H + 45, cited="Глава поселения")
    rtp = fact(2, "Д-3", "РТП", "distance_km", 5.8, 13 * H + 24, 13 * H + 40)
    mem = memory(FIRE, orig, relay, rtp)
    s = mem.state(13 * H + 50)["distance_km"]
    assert s.status == "противоречие" and not s.confirmed
    assert not mem.agree(orig, relay)  # один и тот же источник по существу
    winner, votes = majority_vote(mem, [orig, relay, rtp], 13 * H + 50)["distance_km"]
    assert winner.value == 2.0 and votes == 2  # голосование считает пересказ вторым голосом


def test_relay_without_observer_needs_clarification_not_contradiction():
    relay = fact(0, "Д-1", "ЕДДС", "distance_km", 2.0, 15 * H, 15 * H + 20, cited="Глава поселения")
    rtp = fact(1, "Д-2", "РТП", "distance_km", 4.1, 15 * H + 10, 15 * H + 22)
    s = memory(FIRE, relay, rtp).state(15 * H + 30)["distance_km"]
    assert s.status == "требует уточнения" and not s.conflicts
    assert "не указано, кто наблюдал" in s.unclear[0].reasons


def test_single_source_jump_needs_clarification():
    mem = memory(FLOOD, fact(0, "Д-1", "Гидропост", "water_level", 760, 12 * H, 12 * H + 10),
                 fact(1, "Д-2", "Гидропост", "water_level", 900, 13 * H, 13 * H + 10))  # +140 см за час
    s = mem.state(13 * H + 15)["water_level"]
    assert s.status == "требует уточнения" and s.jump and not s.conflicts
    assert "требует уточнения" in evac(mem, 13 * H + 15).recheck


def test_jump_is_accepted_when_independently_confirmed():
    mem = memory(FLOOD, fact(0, "Д-1", "Гидропост", "water_level", 760, 12 * H, 12 * H + 10),
                 fact(1, "Д-2", "Гидропост", "water_level", 900, 13 * H, 13 * H + 10),
                 fact(2, "Д-3", "Оперативная группа", "water_level", 905, 13 * H + 5, 13 * H + 12))
    assert mem.state(13 * H + 15)["water_level"].status == "согласуется"


def test_irreversible_category_cannot_go_back():
    mem = memory(FLOOD, fact(0, "Д-1", "БПЛА", "ice_jam", "прорван", 14 * H, 14 * H + 10),
                 fact(1, "Д-2", "Авиаразведка", "ice_jam", "держится", 15 * H, 15 * H + 10))
    assert mem.state(15 * H + 15)["ice_jam"].status == "требует уточнения"


def test_same_time_same_object_category_is_contradiction():
    closed = fact(0, "Д-1", "ГИБДД", "road", "закрыта", 15 * H + 20, 15 * H + 25)
    opened = fact(1, "Д-2", "Дорожная служба", "road", "открыта", 15 * H + 20, 15 * H + 35)
    s = memory(FIRE, closed, opened).state(15 * H + 40)["road"]
    assert s.status == "противоречие"


def test_category_change_at_different_times_is_not_flagged():
    mem = memory(FIRE, fact(0, "Д-1", "ГИБДД", "road", "открыта", 13 * H, 13 * H + 5),
                 fact(1, "Д-2", "ГИБДД", "road", "закрыта", 15 * H, 15 * H + 5))
    s = mem.state(15 * H + 10)["road"]
    assert s.status == "не подтверждено" and [f.value for f in s.current] == ["закрыта"]


def test_different_flanks_are_different_objects():
    north = fact(0, "Д-1", "РТП", "distance_km", 4.0, 15 * H, 15 * H + 10, subject="северная кромка")
    south = fact(1, "Д-2", "БПЛА", "distance_km", 6.5, 15 * H + 5, 15 * H + 12, subject="южная кромка")
    mem = memory(FIRE, north, south)
    s = mem.state(15 * H + 20)["distance_km"]
    assert s.status == "не подтверждено" and not s.conflicts and not s.unclear
    assert "северная кромка" in forms._value(mem, s) and "южная кромка" in forms._value(mem, s)


def test_missing_object_needs_clarification():
    rtp = fact(0, "Д-1", "РТП", "distance_km", 4.0, 15 * H, 15 * H + 10)
    vague = fact(1, "Д-2", "Глава поселения", "distance_km", 2.0, 15 * H + 5, 15 * H + 12, subject="")
    s = memory(FIRE, rtp, vague).state(15 * H + 20)["distance_km"]
    assert s.status == "требует уточнения" and s.unclear[0].reasons == ("не указан участок",)


def test_danger_is_not_lifted_by_one_unconfirmed_report():
    mem = memory(FIRE, fact(0, "Д-1", "ЕДДС", "road", "закрыта", 14 * H, 14 * H + 20),
                 fact(1, "Д-2", "112", "road", "открыта", 15 * H, 15 * H + 5))
    road = next(c for c in conclusions(mem, 15 * H + 10) if c.rule.attr == "road")
    assert road.value == road.rule.watch and "неподтверждённым" in road.recheck


def test_drift_forecast_is_off_by_default():
    mem = memory(FIRE, fact(0, "Д-1", "РТП", "distance_km", 4.4, 15 * H, 15 * H + 10),
                 fact(1, "Д-2", "БПЛА", "distance_km", 3.4, 16 * H, 16 * H + 10))  # 1 км в час
    assert evac(mem, 17 * H).recheck is None
    assert "прогноз" in evac(mem, 17 * H, use_drift=True).recheck


def test_staff_decision_resolves_dispute_and_is_logged():
    mem = memory(HEAT, fact(0, "Д-1", "Оперативная группа", "indoor_temp", 3, 12 * H + 13, 12 * H + 30),
                 fact(1, "Д-2", "Котельная", "indoor_temp", 13, 12 * H + 29, 12 * H + 40))
    assert mem.state(13 * H)["indoor_temp"].status == "противоречие"
    mem.decide("indoor_temp", "Д-2", "начальник штаба", 13 * H + 5, "перезвонили в котельную")
    assert mem.state(13 * H)["indoor_temp"].status == "противоречие"  # решение принято позже
    s = mem.state(13 * H + 10)["indoor_temp"]
    assert s.status == "принято штабом" and s.decision.by == "начальник штаба"
    assert [f.rid for f in evac(mem, 13 * H + 10).basis] == ["Д-2"]


def test_review_shows_what_was_known_then_and_now():
    early = fact(0, "Д-1", "РТП", "distance_km", 4.0, 14 * H, 14 * H + 10)
    late = fact(1, "Д-2", "ИСДМ", "distance_km", 2.5, 14 * H + 20, 15 * H + 30)  # пришло через 70 минут
    mem = memory(FIRE, early, late)
    then = mem.state(14 * H + 30)["distance_km"]
    now = mem.state(14 * H + 30, known_at=16 * H)["distance_km"]
    assert [f.rid for f in then.current] == ["Д-1"]
    assert {f.rid for f in now.current} == {"Д-1", "Д-2"}


def test_ask_queue_puts_rule_inputs_first_and_drafts_question():
    mem = memory(FIRE, fact(0, "Д-1", "РТП", "distance_km", 5.0, 13 * H + 20, 13 * H + 40),
                 fact(1, "Д-2", "Глава поселения", "distance_km", 1.5, 13 * H + 30, 13 * H + 40))
    q = ask_queue(mem, 13 * H + 45)
    assert q[0].attr == "distance_km" and q[0].priority == 1
    text = draft_request(mem, q[0], 13 * H + 45)
    assert "Д-1" in text and "Д-2" in text and text.startswith("Кому: РТП")


def test_form_deadlines_are_in_local_time():
    assert forms.next_deadline(12 * H) == 13 * H  # 07:00 МСК
    assert forms.next_deadline(14 * H) == 25 * H  # 19:00 МСК, то есть 01:00 следующего дня
    fields = forms.build(memory(FIRE), 12 * H)
    assert forms.readiness(fields) == (0, 0, len(FIRE.forms))


@pytest.mark.parametrize("key", list(PROFILES))
def test_same_core_works_for_every_profile(key):
    p = PROFILES[key]
    _, reports, pairs = generate(p, 0)
    assert len(reports) > 25 and 7 <= len(pairs) <= 8
    rows = [r for r in run_seed(p, 0) if r["system"] == "СОВА"]
    found = sum(r["pairs_hit"] for r in rows) / sum(r["pairs_active"] for r in rows)
    assert found > 0.9


def test_every_relay_name_is_recognised():
    for source, genitive in RELAY_NAMES.items():
        assert find_relay(f"ЕДДС улуса. По данным {genitive}, на 13:32 ...") == source
    assert find_relay("Донесение РТП. На 13:20 кромка в 5,8 км") is None


@pytest.mark.parametrize("text,expected", [
    ("ЕДДС улуса. По состоянию на 08:20 МСК эвакуировано 40 чел.", 14 * H + 20),
    ("Донесение РТП. В 14:20 кромка в 3,0 км", 14 * H + 20),
])
def test_time_is_local(text, expected):
    assert parse_obs_time(text, t_rec=15 * H)[0] == expected


def test_missing_time_is_not_invented():
    t, unknown = parse_obs_time("Глава: сейчас вода у крайних домов", t_rec=15 * H)
    assert unknown and t == 15 * H


@pytest.mark.parametrize("text,expected", [
    ("Авиалесоохрана, облёт Ми-8. На 15:30 кромка в 3,6 км от посёлка", "На 15:30"),
    ("ЕДДС: по данным РТП в 09:40 МСК вода у моста", "в 09:40 МСК"),
    ("15:10 кромка в 4 км", "15:10"),
    ("Глава: сейчас вода у крайних домов", ""),
])
def test_time_fragment_is_taken_from_text(text, expected):
    """Время наблюдения в карточке показывается вместе с фрагментом текста, откуда оно взято."""
    q = obs_time_quote(text)
    assert q == expected and q in text


def test_runs_without_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("сеть запрещена")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert all(run_seed(p, 1) for p in PROFILES.values())


def test_report_without_time_never_confirms():
    rtp = fact(0, "Д-1", "РТП", "distance_km", 4.0, 15 * H + 30, 15 * H + 35)
    glava = fact(1, "Д-2", "Глава поселения", "distance_km", 4.0, None, 15 * H + 40, unknown=True)
    s = memory(FIRE, rtp, glava).state(15 * H + 45)["distance_km"]
    assert s.status == "не подтверждено" and not s.confirmed and [f.rid for f in s.loose] == ["Д-2"]


def test_report_without_time_is_a_question_not_a_contradiction():
    rtp = fact(0, "Д-1", "РТП", "distance_km", 4.1, 15 * H + 10, 15 * H + 22)
    glava = fact(1, "Д-2", "Глава поселения", "distance_km", 2.0, None, 15 * H + 41, unknown=True, subject="")
    mem = memory(FIRE, rtp, glava)
    s = mem.state(15 * H + 45)["distance_km"]
    assert s.status == "требует уточнения" and not s.conflicts
    assert set(s.unclear[0].reasons) == {"не указано время наблюдения", "не указан участок"}
    ask = next(a for a in ask_queue(mem, 15 * H + 45) if a.kind == "detail")
    assert ask.whom == "Глава поселения" and "время наблюдения" in ask.need
    assert "Прошу уточнить время наблюдения" in draft_request(mem, ask, 15 * H + 45)


def test_report_without_time_received_before_latest_observation_is_history():
    old = fact(0, "Д-1", "Глава поселения", "distance_km", 7.0, None, 14 * H, unknown=True)
    rtp = fact(1, "Д-2", "РТП", "distance_km", 4.1, 15 * H + 10, 15 * H + 22)
    s = memory(FIRE, old, rtp).state(15 * H + 30)["distance_km"]
    assert s.status == "не подтверждено" and [f.rid for f in s.superseded] == ["Д-1"]


def test_quote_check_catches_added_words():
    mem = memory(FIRE)
    mem.add_original("Д-1", "РТП", 14 * H + 10, "РТП: на 14:00 северная кромка в 4,0 км от посёлка, южная в 6,5 км.")
    ok = Fact(0, "Д-1", "РТП", "distance_km", 6.5, 14 * H, 14 * H + 10, subject=EDGE, quote="южная в 6,5 км")
    added = Fact(1, "Д-1", "РТП", "distance_km", 6.5, 14 * H, 14 * H + 10, subject=EDGE,
                 quote="южная в 6,5 км от посёлка")  # модель дописала слова
    assert mem.quote_ok(ok) is True and mem.quote_ok(added) is False


def test_no_reports_about_casualties_is_a_gap_not_absence():
    from sova.reasoning import needed
    mem = memory(FIRE, fact(0, "Д-1", "РТП", "distance_km", 5.0, 14 * H, 14 * H + 10))
    st = mem.state(14 * H + 20)
    assert "casualties" in needed(mem, 14 * H + 20, st) and st["casualties"].status == "нет сведений"
    assert "evacuated" not in needed(mem, 14 * H + 20, st)  # эвакуация не объявлена и не начата
