"""Совместное прочтение трёх контуров: что тревога ЗНАЧИТ.

Неподвижное решение владельца (2026-08-13), задающее смысл всей системы: если
лаборатория укладывается в консенсус и в литературное ожидание, просадка
относительно собственной истории — не провал и не повод разбирать протокол.

Записывать это кодом обязательно потому, что карта относительного контура
центрирована на нашем СОБСТВЕННОМ уровне, а он у этой лаборатории выше и консенсуса,
и литературы. Лаборатория, работающая лучше среднего, получала бы тревоги чаще
прочих — и без этого слоя они читались бы как «стало плохо», хотя означают «стало
иначе, чем у нас же раньше».

Тесты этого файла раньше жили в `test_charts.py`; перенесены сюда, потому что к
секвенциальным картам они отношения не имеют.
"""
import pandas as pd

from ews import verdict


def _state(**kw) -> verdict.ContourState:
    base = dict(metric="BlR", relative_triggered=True, absolute_status="benchmark",
                literature_gap_pp=0.0)
    base.update(kw)
    return verdict.ContourState(**base)


# --- Карта сработала -------------------------------------------------------

def test_trend_when_above_consensus():
    """Просадка относительно себя при уровне выше консенсуса — НЕ разбор протокола."""
    result = verdict.read(_state())
    assert result.action == verdict.ACTION_REVIEW_CASEMIX
    assert "не провал" in result.reasoning


def test_protocol_review_when_below_competency_and_literature():
    result = verdict.read(_state(absolute_status="below", literature_gap_pp=-12.0))
    assert result.action == verdict.ACTION_REVIEW_PROTOCOL
    assert "все три контура" in result.reasoning


def test_competency_breach_alone_is_enough_for_protocol():
    """Ниже уровня компетентности — достаточное основание и без литературы.

    Литература — третья опора, а не обязательное условие: она может не подтвердить
    расхождение просто потому, что формула не различает эту когорту.
    """
    result = verdict.read(_state(absolute_status="below", literature_gap_pp=+3.0))
    assert result.action == verdict.ACTION_REVIEW_PROTOCOL
    assert "литература расхождения не" in result.reasoning


def test_intermediate_level_is_casemix_not_protocol():
    """Между компетентностью и benchmark — наблюдать, но не разбирать протокол."""
    result = verdict.read(_state(absolute_status="competency"))
    assert result.action == verdict.ACTION_REVIEW_CASEMIX
    assert "запаса нет" in result.reasoning


def test_literature_tolerance_is_respected():
    """Небольшой отрыв от литературы признаком проблемы не считается.

    Три пункта ниже литературного ожидания при допуске в пять — это «мы выше
    внешних опор», а при допуске в один — уже «запаса нет». Действие в обоих
    случаях одно, но читающему говорится разное, и это разница по существу.
    """
    lenient = verdict.read(_state(literature_gap_pp=-3.0))
    strict = verdict.read(_state(literature_gap_pp=-3.0), literature_tolerance_pp=-1.0)
    assert lenient.action == strict.action == verdict.ACTION_REVIEW_CASEMIX
    assert "не провал" in lenient.reasoning
    assert "запаса нет" in strict.reasoning


def test_missing_literature_does_not_change_the_verdict():
    """Нет литературного приора — вердикт выносится по двум оставшимся контурам."""
    assert verdict.read(_state(literature_gap_pp=None)).action == \
        verdict.ACTION_REVIEW_CASEMIX


# --- Карта молчит ----------------------------------------------------------

def test_silent_chart_never_triggers_protocol_review():
    """Молчащая карта не даёт разбора протокола, даже при низком уровне.

    Низкий уровень без изменения — давняя разница, а не событие раннего
    предупреждения. Смешивать их значит превращать систему в постоянный сигнал.
    """
    result = verdict.read(_state(relative_triggered=False, absolute_status="below",
                                 literature_gap_pp=-15.0))
    assert result.action == verdict.ACTION_STANDING_GAP


def test_silent_chart_below_literature_only_is_still_standing_gap():
    result = verdict.read(_state(relative_triggered=False, absolute_status="benchmark",
                                 literature_gap_pp=-15.0))
    assert result.action == verdict.ACTION_STANDING_GAP


def test_silent_chart_in_norm_is_trend():
    result = verdict.read(_state(relative_triggered=False))
    assert result.action == verdict.ACTION_TREND and result.reasoning == ""


# --- Сборка состояний ------------------------------------------------------

def test_build_states_maps_unknown_status_to_below():
    """Всё, что не benchmark и не competency, — ниже компетентности.

    `status` приходит из `embryo.kpi`; шкала там шире трёх значений, и молча
    проглотить незнакомое означало бы считать неизвестное нормой.
    """
    states = verdict.build_states(
        {"BlR": {}}, {"BlR"},
        {"BlR": {"status": "ниже компетентности"}},
        pd.DataFrame([{"metric": "BlR", "gap_pp": -9.0}]))
    assert states[0].absolute_status == "below"
    assert states[0].literature_gap_pp == -9.0
    assert states[0].relative_triggered is True


def test_build_states_without_baseline_leaves_status_unknown():
    states = verdict.build_states({"BlR": {}}, set(), {}, None)
    assert states[0].absolute_status is None
    assert states[0].relative_triggered is False


def test_build_states_ignores_empty_literature():
    states = verdict.build_states({"BlR": {}}, {"BlR"}, {}, pd.DataFrame())
    assert states[0].literature_gap_pp is None


# --- Сводка ----------------------------------------------------------------

def test_summary_reports_the_most_expensive_action():
    """Одна строка на тревогу называет самое дорогое из требуемых действий."""
    verdicts = [
        verdict.Verdict("FR", verdict.ACTION_REVIEW_CASEMIX, "", ""),
        verdict.Verdict("BlR", verdict.ACTION_REVIEW_PROTOCOL, "", ""),
    ]
    assert "разбор протокола: BlR" in verdict.summarise(verdicts)


def test_summary_of_casemix_says_it_is_not_a_failure():
    verdicts = [verdict.Verdict("BlR", verdict.ACTION_REVIEW_CASEMIX, "", "")]
    assert "не провал" in verdict.summarise(verdicts)


def test_summary_of_standing_gap_says_it_is_not_an_event():
    verdicts = [verdict.Verdict("BlR", verdict.ACTION_STANDING_GAP, "", "")]
    assert "не событие" in verdict.summarise(verdicts)


def test_empty_summary_is_the_normal_case():
    assert verdict.summarise([]) == "в норме по всем контурам"
