"""Прогноз DT для циклов без него: что уходит в службу и что считается заново.

Шаг пишет в чужую базу и запускает расчёт без врача, поэтому здесь проверяется в
первую очередь то, где он обязан ОСТАНОВИТЬСЯ: на неразрешённой подстановке, на
цикле без пункции, на неизменившемся входе.
"""
from datetime import datetime

from ews import dt_refresh

RULES = dt_refresh.Rules(amh_constant=2.0, bmi_constant=24, attempt=1,
                         sperm_source="ejaculate", afc_from_follicles=True,
                         okk_for_thawed_oocytes=True, exclude_methods=("NO", "CRYO"))


def _fields(**values):
    base = {"ID": "AB1234567", "Дата OPU": "2026-09-30T19:00:00.000Z", "cycle_uid": "CYC-1",
            "Возраст": 33, "AMH": 3.1, "КАФ": 12, "BMI": 22.5, "№ попытки": 2, "Сперма": "N",
            "Фолликулы": 14, "ОКК": 10, "MII": 8, "Инсеминировано": 8, "2pN": 6,
            "Бласт": None, "Хор.бласт": None, "Метод": "ICSI"}
    base.update(values)
    return base


def test_real_values_win_over_constants():
    snap = dt_refresh.build_snapshot(_fields(), RULES)
    assert snap.patient["amh"] == 3.1 and snap.patient["bmi"] == 22.5
    assert "amh_imputed" not in snap.flags and snap.skip is None


def test_every_imputation_is_flagged():
    snap = dt_refresh.build_snapshot(
        _fields(AMH=None, BMI=None, КАФ=None, **{"№ попытки": None, "Сперма": None}), RULES)
    assert snap.patient["amh"] == 2.0 and snap.patient["bmi"] == 24
    assert snap.patient["afc"] == 14 and snap.patient["attempt"] == 1
    assert snap.patient["sperm_source"] == "ejaculate"
    assert {"amh_imputed", "bmi_imputed", "afc_from_follicles", "attempt_imputed",
            "sperm_source_imputed"} <= set(snap.flags)


def test_without_permission_nothing_is_imputed():
    """Служба не подставляет сама — и шаг не подставляет без решения владельца."""
    snap = dt_refresh.build_snapshot(_fields(AMH=None), dt_refresh.Rules())
    assert "amh" not in snap.patient and snap.skip and "amh" in snap.skip


def test_unknown_stage_is_omitted_and_zero_is_kept():
    snap = dt_refresh.build_snapshot(_fields(**{"2pN": 0}), RULES)
    assert snap.patient["known_pn2"] == 0
    assert "known_blasts" not in snap.patient


def test_no_insemination_does_not_read_as_failed_fertilisation():
    snap = dt_refresh.build_snapshot(_fields(Инсеминировано=0, **{"2pN": 0}), RULES)
    assert "known_pn2" not in snap.patient
    assert "no_insemination_stages_dropped" in snap.flags


def test_no_retrieval_and_cryo_transfer_are_not_scored():
    for method in ("NO", "Cryo"):
        assert dt_refresh.build_snapshot(_fields(Метод=method), RULES).skip


def test_thawed_oocytes_raise_okk_with_a_flag():
    snap = dt_refresh.build_snapshot(_fields(ОКК=6, MII=8), RULES)
    assert snap.patient["known_okk"] == 8 and "okk_raised_for_thawed" in snap.flags


def test_dt_key_uses_clinic_calendar_date():
    """19:00Z — это уже следующие сутки в Ташкенте: ключ берёт местную дату."""
    when = dt_refresh.local_date("2026-09-30T19:00:00.000Z", 5)
    assert dt_refresh.dt_key("ab 1234567", when) == "AB1234567#20261001"


def _select(state, dt_value, **changes):
    fields = _fields(**changes)
    key = dt_refresh.dt_key(fields["ID"], dt_refresh.local_date(fields["Дата OPU"], 5))
    rows = {} if dt_value is None else {key: {"id": "rec1", "fields": {
        "DIGITAL TWIN (посл.)": dt_value}}}
    return dt_refresh.select([fields], rows, state, RULES, 5)


def test_cycle_without_dt_is_scored():
    targets, _ = _select({}, None)
    assert [t.reason for t in targets] == ["без DT"]


def test_pipeline_forecast_is_baselined_not_rescored():
    """Первый проход снимает отпечаток прогноза пайплайна и ничего не считает."""
    state = {}
    targets, counts = _select(state, 0.41)
    assert targets == [] and counts["исходный отпечаток"] == 1
    assert state["scored"]["CYC-1"]["baseline"] is True


def test_new_stage_triggers_rescoring_unchanged_does_not():
    state = {}
    _select(state, 0.41)
    assert _select(state, 0.41)[0] == []
    targets, _ = _select(state, 0.41, Бласт=3, **{"Хор.бласт": 1})
    assert [t.reason for t in targets] == ["новый этап/правка"]


def test_rejected_input_is_not_resent_until_it_changes():
    state = {}
    first, _ = _select(state, None)
    state["scored"]["CYC-1"] = {"fingerprint": first[0].snapshot.fingerprint,
                                "invalid": "cascade:known_mii"}
    assert _select(state, None)[0] == []
    assert _select(state, None, MII=7)[0]


def test_nothing_identifying_goes_to_the_service():
    snap = dt_refresh.build_snapshot(_fields(Имя="Иванова"), RULES)
    assert set(snap.patient) <= {"age", "amh", "afc", "bmi", "attempt", "sperm_source",
                                 "follicles", "known_okk", "known_mii", "known_pn2",
                                 "known_blasts", "known_good"}
