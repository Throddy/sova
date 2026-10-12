from llm_extract import build_memory, evaluate, system_prompt
from sova.extract import parse_value
from sova.profiles import FIRE, FLOOD, HEAT


def test_parse_values_for_all_kinds():
    assert parse_value(FIRE, "distance_km", "3,2 км") == 3.2
    assert parse_value(FIRE, "wind", "СЗ 10 м/с") == ("СЗ", 10)
    assert parse_value(FIRE, "road", "закрыта (задымление)") == "закрыта"
    assert parse_value(FLOOD, "ice_jam", "прорван") == "прорван"
    assert parse_value(HEAT, "air_temp", "-44 °C") == -44
    assert parse_value(FIRE, "evacuated", "около 120 чел.") == 120


def test_prompt_is_built_from_profile():
    text = system_prompt(FLOOD)
    assert "water_level" in text and "затор" in text.lower()


def test_evaluate_counts_hits_and_misses():
    rows = [{"id": "Д-01", "text": "Донесение РТП. На 10:16 кромка в 6,5 км", "received": "10:30", "source": "РТП",
             "gold": [{"attr": "distance_km", "value": 6.5, "observed": "10:16", "planted": None},
                      {"attr": "area_ha", "value": 240, "observed": "10:16", "planted": None}]}]
    extracted = {"Д-01": [{"attr": "distance_km", "value": "6,5"}, {"attr": "personnel", "value": "12"}]}
    m = evaluate(FIRE, rows, extracted)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    mem = build_memory(FIRE, rows, extracted)
    assert mem.state(11 * 60)["distance_km"].current[0].t_obs == 10 * 60 + 16


def _row(text, received="16:00", source="ЕДДС", gold=None, rid="Д-01"):
    return {"id": rid, "text": text, "received": received, "source": source, "relay_of": None,
            "gold": gold or [{"attr": "distance_km", "value": 3.0, "observed": "15:30", "planted": None}]}


def test_model_time_beats_header_time():
    """«Донесение от 15:50» это время отправки, а не наблюдения."""
    row = _row("Донесение РТП от 15:50. На 15:30 кромка в 3 км от посёлка.", source="РТП")
    claim = {"attr": "distance_km", "value": "3", "time": "15:30", "cited": "", "observer": "",
             "subject": "кромка, посёлок", "quote": "кромка в 3 км от посёлка"}
    mem = build_memory(FIRE, [row], {"Д-01": [claim]})
    f = mem.known(24 * 60)[0]
    assert f.t_obs == 15 * 60 + 30 and not f.t_unknown and f.t_quote == "На 15:30"
    assert evaluate(FIRE, [row], {"Д-01": [claim]})["fields"]["time"] == (1, 1)


def test_relay_named_by_model_is_not_second_source():
    """«со слов главы» и «глава сообщил» код по тексту не ловит, поэтому их отмечает модель."""
    for text in ("ЕДДС улуса: со слов главы, огонь в 2 км от посёлка.", "ЕДДС улуса: глава сообщил, что огонь в 2 км."):
        row = _row(text, gold=[{"attr": "distance_km", "value": 2.0, "observed": None, "planted": None}])
        claim = {"attr": "distance_km", "value": "2", "time": "", "cited": "главы администрации", "observer": "",
                 "subject": "", "quote": "огонь в 2 км"}
        f = build_memory(FIRE, [row], {"Д-01": [claim]}).known(24 * 60)[0]
        assert f.cited == "Глава поселения" and not f.observed and f.t_unknown and f.subject == ""


def test_invented_quote_is_caught_by_memory():
    row = _row("ЕДДС улуса. На 15:30 кромка в 3 км.")
    claim = {"attr": "distance_km", "value": "3", "time": "15:30", "cited": "", "observer": "",
             "subject": "кромка, посёлок", "quote": "кромка в 3 км от посёлка"}
    mem = build_memory(FIRE, [row], {"Д-01": [claim]})
    assert mem.quote_ok(mem.known(24 * 60)[0]) is False
    assert evaluate(FIRE, [row], {"Д-01": [claim]})["fields"]["quote"] == (0, 1)


def test_prompt_names_new_fields():
    text = system_prompt(FIRE)
    assert "со слов" in text and "МСК" in text and "слово в слово" in text
