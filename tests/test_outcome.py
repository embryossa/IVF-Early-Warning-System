"""Медленный контур: прогноз исхода против факта.

Тревог не поднимает и поднимать не должен: исход зависит от переноса, эндометрия и
десятка вещей вне лаборатории. Его роль — подтверждать или опровергать тренды
быстрого контура. Поэтому здесь проверяется главным образом то, где он обязан
МОЛЧАТЬ: на малых числах, на неизвестных исходах, на пересекающихся интервалах.
"""
import pandas as pd
import pytest

from ews import outcome


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _cohort(n_known: int, successes: int, label_date="2026-06-01") -> pd.DataFrame:
    rows = [{"date_opu": label_date, "transfer_outcome": 1 if i < successes else 0,
             "digital_twin": 0.45} for i in range(n_known)]
    return _frame(rows)


# --- Знаменатель -----------------------------------------------------------

def test_denominator_counts_only_known_outcomes():
    """Считать частоту беременности по всем циклам значило бы делить на популяцию,
    которой исход ещё не мог случиться."""
    frame = _frame([{"transfer_outcome": 1}, {"transfer_outcome": 0},
                    {"transfer_outcome": None}, {"transfer_outcome": ""}])
    window = outcome.summarise(frame)
    assert window.n_cycles == 4 and window.n_known == 2
    assert window.observed == pytest.approx(0.5)


def test_text_outcomes_are_not_counted():
    """Текстовая отметка исхода наблюдением не является и нулём не становится."""
    frame = _frame([{"transfer_outcome": "Preg"}, {"transfer_outcome": "нет"}])
    assert outcome.summarise(frame).n_known == 0


def test_no_known_outcome_says_so():
    window = outcome.summarise(_frame([{"transfer_outcome": None}]))
    assert window.observed is None and "не известен" in window.note


def test_missing_columns_do_not_crash():
    """Кадр без колонок исхода — пустая таблица, а не повод падать."""
    window = outcome.summarise(pd.DataFrame(columns=["cycle_uid"]))
    assert window.n_cycles == 0 and window.n_known == 0


def test_compare_periods_survives_a_frame_without_dates():
    windows = outcome.compare_periods(pd.DataFrame(columns=["cycle_uid"]),
                                      pd.Timestamp("2026-05-01"))
    assert len(windows) == 2 and all(w.n_known == 0 for w in windows)


# --- Интервал и согласие с моделью ----------------------------------------

def test_model_prediction_is_checked_against_the_interval():
    window = outcome.summarise(_cohort(40, 16))          # 40%
    assert window.agrees(0.40) is True
    assert window.agrees(0.90) is False
    assert window.agrees(None) is None


def test_calibration_note_uses_digital_twin_only():
    """Опора медленного контура — только DT (решение владельца 2026-10-17)."""
    note = outcome.calibration_note(outcome.summarise(_cohort(40, 16)))
    assert "DIGITAL TWIN" in note and "PRAI" not in note


def test_calibration_note_without_models_says_so():
    frame = _frame([{"transfer_outcome": 1}, {"transfer_outcome": 0}])
    assert "модельных вероятностей" in outcome.calibration_note(
        outcome.summarise(frame))


# --- Вердикт контура -------------------------------------------------------

def test_verdict_is_withheld_on_small_samples():
    """Молчание на малых числах лучше вывода: исход известен у трети циклов.

    Это самый частый честный ответ контура, и он обязан быть отличим от «изменений
    нет».
    """
    windows = [outcome.summarise(_cohort(5, 2), "до окна"),
               outcome.summarise(_cohort(4, 1), "опорное окно")]
    good, note = outcome.confirms_process_trend(windows)
    assert good is None and "недостаточно" in note


def test_overlapping_intervals_do_not_confirm_a_trend():
    windows = [outcome.summarise(_cohort(40, 16), "до окна"),
               outcome.summarise(_cohort(40, 15), "опорное окно")]
    good, note = outcome.confirms_process_trend(windows)
    assert good is False and "пересекаются" in note


def test_separated_intervals_confirm_a_trend():
    windows = [outcome.summarise(_cohort(120, 84), "до окна"),      # 70%
               outcome.summarise(_cohort(120, 24), "опорное окно")]  # 20%
    good, note = outcome.confirms_process_trend(windows)
    assert good is True and "снизился" in note


def test_periods_are_split_at_the_reference_window_start():
    frame = pd.concat([_cohort(30, 10, "2026-01-15"), _cohort(30, 20, "2026-06-15")],
                      ignore_index=True)
    before, after = outcome.compare_periods(frame, pd.Timestamp("2026-05-01"))
    assert before.n_known == 30 and after.n_known == 30
    assert before.observed < after.observed


def test_wilson_is_undefined_on_impossible_counts():
    assert all(pd.isna(v) for v in outcome._wilson(5, 0))
    assert all(pd.isna(v) for v in outcome._wilson(11, 10))
