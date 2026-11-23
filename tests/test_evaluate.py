"""Оценка на закрытии: единственная точка, где в системе возникают тревоги.

Всё, что здесь проверяется, — это либо неподвижное решение владельца, либо гейт,
защищающий самый дорогой сценарий (остановленный лот среды по неоткалиброванному
порогу). Ни то, ни другое не видно по выходным числам: система, поднимающая alert
там, где не должна, выглядит ровно как система, поднимающая alert по делу.
"""
import json

import pandas as pd
import pytest

from conftest import cycle_frame, cycle_row, make_config

from ews import evaluate, verdict


# Пороги подобраны так, чтобы события были ДЕТЕРМИНИРОВАННЫМИ. Цикл 0 бластоцист из
# 10 зигот при базовой доле 0.60 даёт вклад -10*ln(0.7) = 3.567 — он же и выбран
# опорной величиной ниже.
CROSSING = {"bl_total": 0.0, "two_pn": 10.0}

# Цикл, проваливший все три карты сразу, при живых знаменателях у каждой:
# MIIR 10/20 (база 0.80) -> 3.29, FR 5/10 (0.80) -> 1.64, BlR 0/5 (0.60) -> 1.78.
# Знаменатель одной метрики — числитель предыдущей, поэтому «обнулить всё» нельзя:
# при mii=0 у FR не остаётся наблюдения вовсе, и контур молчит не от здоровья.
ALL_CONTOURS = {"occ": 20.0, "mii": 10.0, "two_pn": 5.0, "bl_total": 0.0}


def _thresholds(watch=1.0, alert=3.0, metrics=("FR", "BlR", "MIIR")):
    return {"h_watch": {m: watch for m in metrics},
            "h_alert": {m: alert for m in metrics}}


def _evaluate(cfg, clinic, baselines, row, thresholds=None, states=None, **kw):
    states = {} if states is None else states
    return evaluate.evaluate_cycle(cfg, clinic, pd.Series(row), baselines,
                                   thresholds or _thresholds(), states, **kw)


# --- Что вообще попадает в оценку -----------------------------------------

def test_only_closure_cycles_are_evaluated(cfg, clinic, baselines):
    """T0 и T+1 — чистая регистрация. Неподвижное решение владельца.

    Более ранняя тревога — преждевременная паника, которая обесценивает систему
    быстрее, чем пропущенный сигнал.
    """
    frame = cycle_frame([
        cycle_row("OPU-1", checkpoint="opu"),
        cycle_row("FERT-1", checkpoint="fertilisation"),
        cycle_row("CLOSED-1", checkpoint="closure"),
    ])
    out = evaluate.apply_evaluations(cfg, clinic, frame, baselines,
                                     _thresholds(), {})
    assert [e.cycle_uid for e in out] == ["CLOSED-1"]


def test_already_evaluated_cycles_are_not_fed_twice(cfg, clinic, baselines):
    """Цикл оценивается один раз: карта секвенциальна.

    Повторная подача сдвинула бы статистику тем же свидетельством второй раз, а
    тревога появилась бы заново с новой датой.
    """
    frame = cycle_frame([cycle_row("A"), cycle_row("B", date_opu="2026-06-02")])
    states: dict = {}
    first = evaluate.apply_evaluations(cfg, clinic, frame, baselines,
                                       _thresholds(), states)
    observations = states["BlR"].observations
    second = evaluate.apply_evaluations(cfg, clinic, frame, baselines, _thresholds(),
                                        states, already_evaluated={"A", "B"})
    assert len(first) == 2 and second == []
    assert states["BlR"].observations == observations


def test_cycles_are_fed_in_closure_order(cfg, clinic, baselines):
    """Порядок — по дате ЗАКРЫТИЯ, и он хронологический.

    Перестановка циклов изменила бы траекторию карты, то есть и момент срабатывания.
    """
    frame = cycle_frame([
        cycle_row("LATE", date_opu="2026-06-10"),
        cycle_row("EARLY", date_opu="2026-06-01"),
    ])
    out = evaluate.apply_evaluations(cfg, clinic, frame, baselines, _thresholds(), {})
    assert [e.cycle_uid for e in out] == ["EARLY", "LATE"]


def test_empty_frame_yields_no_evaluations(cfg, clinic, baselines):
    """Кадр без единого цикла — результат «нечего оценивать», а не сбой.

    У пустого кадра нет колонки closure_due вовсе, и сортировка по ней роняла
    прогон на первой же попытке запустить новую клинику.
    """
    empty = pd.DataFrame(columns=["cycle_uid", "date_opu", "checkpoint"])
    assert evaluate.apply_evaluations(cfg, clinic, empty, baselines,
                                      _thresholds(), {}) == []


def test_frame_without_checkpoint_column_is_not_a_crash(cfg, clinic, baselines):
    assert evaluate.apply_evaluations(cfg, clinic, pd.DataFrame(), baselines,
                                      _thresholds(), {}) == []


# --- Гейт alert ------------------------------------------------------------

def test_draft_never_produces_alert(settings, tmp_path, clinic, baselines):
    """В черновом окружении alert невозможен физически, а не по договорённости."""
    cfg = make_config(settings, tmp_path, environment="draft", site="home")
    result = _evaluate(cfg, clinic, baselines, cycle_row("A", **CROSSING))
    assert result.level == evaluate.LEVEL_WATCH
    assert any("понижен до watch" in note for note in result.notes)


def test_production_with_own_calibration_reaches_alert(cfg, clinic, baselines):
    result = _evaluate(cfg, clinic, baselines, cycle_row("A", **CROSSING))
    assert result.level == evaluate.LEVEL_ALERT


def test_uncalibrated_thresholds_block_alert(settings, tmp_path, clinic, baselines):
    """Пороги без бэкфита не открывают alert даже в боевом окружении."""
    settings["thresholds"]["calibrated"] = False
    cfg = make_config(settings, tmp_path)
    result = _evaluate(cfg, clinic, baselines, cycle_row("A", **CROSSING))
    assert result.level == evaluate.LEVEL_WATCH


# --- Эскалация по обстановке окна -----------------------------------------

def _watching(cfg, clinic, baselines):
    """Цикл, доведённый ровно до watch: карта сработала, alert ещё нет."""
    result = _evaluate(cfg, clinic, baselines, cycle_row("A", **CROSSING),
                       thresholds=_thresholds(watch=1.0, alert=99.0))
    assert result.level == evaluate.LEVEL_WATCH
    return result


def test_strong_context_escalates_watch_to_alert(cfg, clinic, baselines):
    """Сильное независимое подтверждение поднимает watch до alert.

    Маршрут описан во всех документах системы, но выполняется он ТОЛЬКО этим
    вызовом: обстановка окна (среда, лоты) собирается уже после оценки — знать, что
    цикл сработал, надо раньше, чем идти за логами. Без пересмотра уровня здесь
    быстрый маршрут не срабатывал бы никогда.
    """
    result = _watching(cfg, clinic, baselines)
    result.corroboration_rank = evaluate.RANK_STRONG
    assert evaluate.escalate_with_context(cfg, result) is True
    assert result.level == evaluate.LEVEL_ALERT


def test_moderate_corroboration_never_escalates(cfg, clinic, baselines):
    """Другая метрика того же контура — умеренный источник, до alert не поднимает."""
    result = _watching(cfg, clinic, baselines)
    result.corroboration_rank = evaluate.RANK_MODERATE
    assert evaluate.escalate_with_context(cfg, result) is False
    assert result.level == evaluate.LEVEL_WATCH


def test_coverage_drop_blocks_escalation(cfg, clinic, baselines):
    """Просевшее покрытие конверта запрещает эскалацию.

    Падение покрытия ставит под вопрос саму опору — состав пациенток или дрейф
    модели (§4). Поднимать по ней тревогу до разбора значит принимать решение по
    измерителю, о котором известно, что он поехал.
    """
    result = _watching(cfg, clinic, baselines)
    result.corroboration_rank = evaluate.RANK_STRONG
    assert evaluate.escalate_with_context(cfg, result, coverage_blocks=True) is False
    assert result.level == evaluate.LEVEL_WATCH
    assert any("заблокирована" in note for note in result.notes)


def test_escalation_respects_the_alert_gate(settings, tmp_path, clinic, baselines):
    """Сильное подтверждение не обходит гейт: в черновике alert по-прежнему нет."""
    cfg = make_config(settings, tmp_path, environment="draft", site="home")
    result = _watching(cfg, clinic, baselines)
    result.corroboration_rank = evaluate.RANK_STRONG
    assert evaluate.escalate_with_context(cfg, result) is False
    assert result.level == evaluate.LEVEL_WATCH


def test_escalation_is_idempotent(cfg, clinic, baselines):
    """Повторный вызов на уже поднятой тревоге ничего не добавляет.

    Прогон может пройти по одному и тому же циклу дважды (replay поверх истории);
    вторая запись «повышено до alert» в карточке читалась бы как второе событие.
    """
    result = _watching(cfg, clinic, baselines)
    result.corroboration_rank = evaluate.RANK_STRONG
    evaluate.escalate_with_context(cfg, result)
    notes = list(result.notes)
    assert evaluate.escalate_with_context(cfg, result) is False
    assert result.notes == notes


def test_ok_cycle_is_never_escalated(cfg, clinic, baselines):
    """Обстановка окна не создаёт тревогу там, где карта молчала.

    Среда — обстановка, а не причина: она может усилить сигнал, но не породить его.
    """
    result = _evaluate(cfg, clinic, baselines, cycle_row("A"))
    assert result.level == evaluate.LEVEL_OK
    result.corroboration_rank = evaluate.RANK_STRONG
    assert evaluate.escalate_with_context(cfg, result) is False
    assert result.level == evaluate.LEVEL_OK


# --- Подтверждение --------------------------------------------------------

def test_all_contours_at_once_is_not_triple_confirmation(cfg, clinic, baselines):
    """Сработали все контуры сразу — общий фактор, а не тройное подтверждение.

    Это понижает уверенность в локализации, а не повышает: разбирается по §4
    (состав пациенток / дрейф модели / общелабораторное событие).
    """
    row = cycle_row("A", **ALL_CONTOURS)
    result = _evaluate(cfg, clinic, baselines, row,
                       thresholds=_thresholds(watch=1.0, alert=99.0))
    triggered = {m for m, info in result.charts.items() if info.get("crossed")}
    assert triggered == {"FR", "BlR", "MIIR"}, triggered
    assert result.corroboration_rank == evaluate.RANK_NONE
    assert any("все контуры" in note for note in result.notes)
    assert any(s["type"] == "all_contours_at_once" for s in result.corroboration)


def test_contour_lists_every_triggered_chart(cfg, clinic, baselines):
    """Контур — по всем сработавшим картам, а не по первой попавшейся."""
    row = cycle_row("A", **ALL_CONTOURS)
    result = _evaluate(cfg, clinic, baselines, row,
                       thresholds=_thresholds(watch=1.0, alert=99.0))
    assert set(result.contour.split("+")) == {"culture", "doctor", "embryologist"}


def test_chart_corroboration_alone_is_never_strong(cfg, clinic, baselines):
    """Карты сами по себе сильного подтверждения не дают — ни в каком сочетании.

    Ранг strong зарезервирован за независимым источником (смена лота, эпизод среды).
    Пара CUSUM+EWMA на одной метрике не даёт даже умеренного: один ряд.
    """
    row = cycle_row("A", **CROSSING)
    result = _evaluate(cfg, clinic, baselines, row)
    assert result.corroboration_rank in (evaluate.RANK_NONE, evaluate.RANK_MODERATE)


# --- Ожидание карты -------------------------------------------------------

def test_risk_adjustment_is_off_by_default(cfg, clinic, baselines):
    """Пока гейт закрыт, ожидание берётся из базовой доли даже при наличии конверта.

    Гейт закрыт по ИЗМЕРЕНИЮ, а не из осторожности: прямая подстановка P50/n
    ухудшила разнородность втрое (METHODOLOGY §5.8).
    """
    assert cfg.settings["envelope"]["risk_adjustment_enabled"] is False
    envelope = {"blasts_after_fert_p50": 2.0, "blasts_after_fert_p50__mode": "prospective"}
    result = _evaluate(cfg, clinic, baselines, cycle_row("A"), envelope=envelope)
    assert result.charts["BlR"]["expected_source"] == "baseline"
    assert result.charts["BlR"]["expected"] == pytest.approx(0.60)


def test_envelope_is_used_when_risk_adjustment_is_enabled(settings, tmp_path, clinic,
                                                          baselines):
    settings["envelope"]["risk_adjustment_enabled"] = True
    cfg = make_config(settings, tmp_path)
    envelope = {"blasts_after_fert_p50": 4.0,
                "blasts_after_fert_p50__mode": "retrospective"}
    result = _evaluate(cfg, clinic, baselines, cycle_row("A"), envelope=envelope)
    assert result.charts["BlR"]["expected_source"] == "envelope:retrospective"
    assert result.charts["BlR"]["expected"] == pytest.approx(0.5)   # 4 из 8 зигот


def test_expected_probability_is_clipped(settings, tmp_path, clinic, baselines):
    """Ожидание не бывает 0 или 1: там вырождается биномиальная дисперсия."""
    settings["envelope"]["risk_adjustment_enabled"] = True
    cfg = make_config(settings, tmp_path)
    envelope = {"blasts_after_fert_p50": 999.0,
                "blasts_after_fert_p50__mode": "prospective"}
    result = _evaluate(cfg, clinic, baselines, cycle_row("A"), envelope=envelope)
    assert result.charts["BlR"]["expected"] == pytest.approx(0.99)


def test_metric_without_baseline_gets_no_chart(cfg, clinic, baselines):
    """Без опорного параметра карта не строится: центра нет."""
    result = _evaluate(cfg, clinic, {"BlR": baselines["BlR"]}, cycle_row("A"))
    assert set(result.charts) == {"BlR"}


# --- Состояние карт -------------------------------------------------------

def test_chart_state_survives_a_round_trip():
    state = evaluate.ChartState(metric="BlR", statistic=1.25, observations=40,
                                signals=1, watch_episodes=2, above_watch=True,
                                peak=3.5)
    state.record("2026-06-07", 1.25)
    restored = evaluate.ChartState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()


def test_trajectory_is_bounded():
    """Траектория ограничена: состояние карт не должно расти без предела."""
    state = evaluate.ChartState(metric="BlR")
    for index in range(evaluate.ChartState.TRAJECTORY_LIMIT + 50):
        state.record("2026-06-07", float(index))
    assert len(state.trajectory) == evaluate.ChartState.TRAJECTORY_LIMIT
    assert state.peak == float(evaluate.ChartState.TRAJECTORY_LIMIT + 49)


def test_state_file_keeps_the_evaluated_set(cfg, clinic, tmp_path):
    """Множество уже оценённых циклов обязано пережить перезапуск.

    Без него каждый прогон прокручивал бы всю историю заново: на первом же dry-run
    число наблюдений выросло с 630 до 1260.
    """
    path = tmp_path / "state" / "ews_charts_state.json"
    states = {"BlR": evaluate.ChartState(metric="BlR", statistic=2.0, observations=10)}
    evaluate.save_state(path, states, cfg, clinic, {"A", "B"})
    restored, evaluated = evaluate.load_state(path)
    assert evaluated == {"A", "B"}
    assert restored["BlR"].statistic == 2.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["environment"] == "production" and payload["site"] == "lab"


def test_missing_state_file_is_a_first_run(tmp_path):
    states, evaluated = evaluate.load_state(tmp_path / "нет.json")
    assert states == {} and evaluated == set()


def test_watch_episode_is_not_repeated_on_every_cycle(cfg, clinic, baselines):
    """Пока статистика висит наверху, новых эпизодов watch не возникает.

    Иначе одна затяжная экскурсия давала бы watch на каждом закрывшемся цикле, и
    измеренная калибровкой частота «24 в год» перестала бы соответствовать тому,
    что видит человек: симуляция считала именно пересечения.
    """
    frame = cycle_frame([
        cycle_row(f"C{index}", date_opu=f"2026-06-0{index + 1}", **CROSSING)
        for index in range(4)
    ])
    states: dict = {}
    out = evaluate.apply_evaluations(cfg, clinic, frame, baselines,
                                     _thresholds(watch=1.0, alert=999.0), states)
    crossings = [e.cycle_uid for e in out if e.charts["BlR"]["crossed"] == "watch"]
    assert crossings == ["C0"], crossings
    assert states["BlR"].watch_episodes == 1


# --- Вердикт --------------------------------------------------------------

def test_verdict_prefers_protocol_over_casemix(cfg, clinic, baselines):
    """Из нескольких вердиктов действием цикла становится самый дорогой.

    Если хоть одна метрика требует разбора протокола, карточка обязана говорить об
    этом, а не о более мягком «тренде» соседней.
    """
    baselines = dict(baselines)
    baselines["BlR"] = {**baselines["BlR"], "status": "below"}
    literature = pd.DataFrame([{"metric": "BlR", "gap_pp": -20.0},
                               {"metric": "FR", "gap_pp": 5.0}])
    row = cycle_row("A", **ALL_CONTOURS)
    result = _evaluate(cfg, clinic, baselines, row,
                       thresholds=_thresholds(watch=1.0, alert=99.0))
    evaluate.attach_verdict(result, baselines, literature)
    assert result.action == verdict.ACTION_REVIEW_PROTOCOL
    assert "разбор протокола" in result.verdict_summary


def test_silent_charts_get_no_verdict(cfg, clinic, baselines):
    """Молчащая карта вердикта не порождает: событие не наступило."""
    result = _evaluate(cfg, clinic, baselines, cycle_row("A"))
    evaluate.attach_verdict(result, baselines, None)
    assert result.verdicts == [] and result.action == verdict.ACTION_TREND


def test_baselines_from_manifest_reads_metrics(tmp_path):
    path = tmp_path / "incontrol_parameters.json"
    path.write_text(json.dumps({"metrics": [{"metric": "BlR", "rate": 0.6}]}),
                    encoding="utf-8")
    assert evaluate.baselines_from_manifest(path)["BlR"]["rate"] == 0.6
    assert evaluate.baselines_from_manifest(tmp_path / "нет.json") == {}


# --- Отложенная доводка по среде -------------------------------------------
#
# Второй такт: обстановка окна приходит ПОЗЖЕ оценки. До этого цикл попадал в
# `already_evaluated` и не пересматривался никогда — маршрут `watch -> alert` по
# сильному подтверждению не срабатывал ни разу за всю историю площадки, хотя
# документы его описывали. Решение владельца 2026-09-10: доводить, но только вверх.

def _journal(level="watch", miri_status="pending", cycle_uid="CYC-AAA111"):
    return pd.DataFrame([{
        "alert_id": "two_line-00001", "cycle_uid": cycle_uid,
        "raised_at": "2026-06-07", "level": level, "chart": "BlR",
        "corroboration": "[]", "corroboration_rank": "none",
        "miri_status": miri_status, "action": "casemix", "notes": "",
    }])


class _Context:
    """Обстановка окна в том виде, в каком её отдаёт confounders.collect."""

    def __init__(self, status, unusual, minutes=900.0, lots=5):
        self.miri_status = status
        self.critical_minutes = minutes
        self.lot_changes = lots
        self._unusual = unusual

    def is_unusual(self, thresholds):
        return self._unusual


def _patch(monkeypatch, contexts, thresholds=None):
    from ews import confounders
    monkeypatch.setattr(confounders, "collect",
                        lambda cfg, clinic, frame: contexts)
    monkeypatch.setattr(confounders, "load_baseline",
                        lambda cfg, clinic: (
                            thresholds if thresholds is not None
                            else {"critical_minutes": 692.0, "lot_changes": 2.0},
                            "границы от 2026-09-07"))


def test_pending_card_is_raised_when_the_window_turns_out_unusual(
        cfg, clinic, monkeypatch):
    """Логи догнали окно, обстановка необычна — карточка становится alert."""
    frame = cycle_frame([cycle_row("CYC-AAA111")])
    _patch(monkeypatch, {"CYC-AAA111": _Context("enriched", True)})

    updated, upgrades = evaluate.deferred_upgrades(cfg, clinic, frame, _journal())

    assert len(upgrades) == 1
    row = updated.iloc[0]
    assert row["level"] == evaluate.LEVEL_ALERT
    assert row["corroboration_rank"] == evaluate.RANK_STRONG
    assert row["miri_status"] == "enriched"
    assert row["upgraded_at"]
    assert "повышено до alert отложенной доводкой" in row["notes"]


def test_ordinary_window_only_records_the_status(cfg, clinic, monkeypatch):
    """Обстановка разобрана и оказалась обычной — уровень не трогается.

    Источник, присутствующий всегда, не подтверждает ничего; здесь он просто
    перестаёт быть неизвестным.
    """
    frame = cycle_frame([cycle_row("CYC-AAA111")])
    _patch(monkeypatch, {"CYC-AAA111": _Context("enriched", False)})

    updated, upgrades = evaluate.deferred_upgrades(cfg, clinic, frame, _journal())

    assert upgrades == []
    assert updated.iloc[0]["level"] == evaluate.LEVEL_WATCH
    assert updated.iloc[0]["miri_status"] == "enriched"


def test_window_still_uncovered_is_left_alone(cfg, clinic, monkeypatch):
    """Логи ещё не покрыли окно — карточка ждёт следующей выгрузки, не меняясь."""
    frame = cycle_frame([cycle_row("CYC-AAA111")])
    _patch(monkeypatch, {"CYC-AAA111": _Context("pending", True)})

    updated, upgrades = evaluate.deferred_upgrades(cfg, clinic, frame, _journal())

    assert upgrades == []
    assert updated.iloc[0]["miri_status"] == "pending"


def test_window_that_logs_will_never_cover_stops_being_pending(
        cfg, clinic, monkeypatch):
    """`unavailable` записывается тоже.

    Иначе карточка с окном, которое выгрузки уже не догонят, осталась бы в
    `pending` навсегда и проверялась бы на каждом прогоне до конца времён.
    """
    frame = cycle_frame([cycle_row("CYC-AAA111")])
    _patch(monkeypatch, {"CYC-AAA111": _Context("unavailable", True)})

    updated, _ = evaluate.deferred_upgrades(cfg, clinic, frame, _journal())
    assert updated.iloc[0]["miri_status"] == "unavailable"
    assert updated.iloc[0]["level"] == evaluate.LEVEL_WATCH


def test_deferred_upgrade_never_lowers_a_level(cfg, clinic, monkeypatch):
    """Уже поднятая тревога доводкой не снимается — неподвижное правило."""
    frame = cycle_frame([cycle_row("CYC-AAA111")])
    _patch(monkeypatch, {"CYC-AAA111": _Context("enriched", False)})

    updated, _ = evaluate.deferred_upgrades(cfg, clinic, frame,
                                            _journal(level="alert"))
    assert updated.iloc[0]["level"] == evaluate.LEVEL_ALERT


def test_coverage_block_stops_the_deferred_upgrade(cfg, clinic, monkeypatch):
    """Просевшее покрытие проспективного конверта запирает и этот маршрут.

    Под вопросом сама опора — состав пациенток или дрейф модели (§4). Причина
    остаётся в карточке: подтверждение найдено, но уровень не поднят.
    """
    frame = cycle_frame([cycle_row("CYC-AAA111")])
    _patch(monkeypatch, {"CYC-AAA111": _Context("enriched", True)})

    updated, upgrades = evaluate.deferred_upgrades(cfg, clinic, frame, _journal(),
                                                   coverage_blocks=True)
    assert upgrades == []
    assert updated.iloc[0]["level"] == evaluate.LEVEL_WATCH
    assert updated.iloc[0]["corroboration_rank"] == evaluate.RANK_STRONG
    assert "заблокирована" in updated.iloc[0]["notes"]


def test_draft_environment_cannot_be_upgraded(draft_cfg, clinic, monkeypatch):
    """Черновой стенд не поднимает alert ни одним из маршрутов."""
    frame = cycle_frame([cycle_row("CYC-AAA111")])
    _patch(monkeypatch, {"CYC-AAA111": _Context("enriched", True)})

    updated, upgrades = evaluate.deferred_upgrades(draft_cfg, clinic, frame,
                                                   _journal())
    assert upgrades == []
    assert updated.iloc[0]["level"] == evaluate.LEVEL_WATCH
    assert "недоступен" in updated.iloc[0]["notes"]


def test_deferred_upgrade_creates_no_cards(cfg, clinic, monkeypatch):
    """Доводка не создаёт тревог: она работает только по уже записанным строкам."""
    frame = cycle_frame([cycle_row("CYC-AAA111"), cycle_row("CYC-ZZZ999")])
    _patch(monkeypatch, {"CYC-AAA111": _Context("enriched", True),
                         "CYC-ZZZ999": _Context("enriched", True)})

    updated, _ = evaluate.deferred_upgrades(cfg, clinic, frame, _journal())
    assert len(updated) == 1
    assert set(updated["cycle_uid"]) == {"CYC-AAA111"}


def test_empty_journal_needs_no_work(cfg, clinic):
    frame = cycle_frame([cycle_row("CYC-AAA111")])
    updated, upgrades = evaluate.deferred_upgrades(cfg, clinic, frame,
                                                   pd.DataFrame())
    assert updated.empty and upgrades == []
