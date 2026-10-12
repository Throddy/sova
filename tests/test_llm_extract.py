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
