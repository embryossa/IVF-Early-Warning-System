"""Литературный контур: опора, которая НЕ двигается вместе с нами.

Собственная базовая линия самореферентна: если лаборатория медленно сползает, а
опорное окно переобъявляется, карты не увидят ничего. Венский порог не сползает, но
он один на всех пациенток. Литературный приор не двигается вместе с нами И учитывает
пациентку — ради этого он и заведён.

Отдельно проверяется когортная арифметика: доля считается как Σ числителей /
Σ знаменателей, а не средним пациентских долей. Это правило проекта, и нарушение
его не видно по числу — оно просто немного другое.
"""
import numpy as np
import pandas as pd
import pytest

from ews import prior


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- Формулы ---------------------------------------------------------------

def test_blastulation_is_flat_below_forty():
    """Ниже 40 лет формула индивидуальной дискриминации не даёт вовсе.

    Это свойство источника, а не недосмотр: Romanski 2022 даёт плоскую 0.70 до 40.
    Знать об этом обязательно — иначе приор BlR принимают за персональный прогноз.
    """
    rates = prior.blastulation_rate(pd.Series([25.0, 32.0, 39.0]))
    assert rates.nunique() == 1 and rates.iloc[0] == pytest.approx(0.70)


def test_blastulation_declines_after_forty():
    older = prior.blastulation_rate(pd.Series([45.0])).iloc[0]
    assert older < 0.70


def test_blastulation_is_clipped():
    assert prior.blastulation_rate(pd.Series([80.0])).iloc[0] >= 0.30


def test_rates_stay_probabilities():
    frame = _frame([{"age": age, "amh": 2.0} for age in (25, 35, 45)])
    rates = prior.expected_rates(frame)
    assert ((rates > 0) & (rates < 1)).all().all()


# --- Импутация AMH ---------------------------------------------------------

def test_amh_constant_fills_only_the_gaps():
    """Реальный AMH всегда важнее константы владельца.

    Константа — доменное решение для пропусков, а не оценка по пациентке; сплошная
    подстановка вырождала бы формулу зрелости в фиксированный уровень.
    """
    frame = _frame([{"age": 33.0, "amh": 6.0}, {"age": 33.0, "amh": None}])
    rates = prior.expected_rates(frame, amh_constant=2.0)
    assert rates["MIIR"].iloc[0] != rates["MIIR"].iloc[1]


def test_without_amh_constant_maturity_is_not_computed():
    frame = _frame([{"age": 33.0, "amh": None}])
    assert pd.isna(prior.expected_rates(frame)["MIIR"].iloc[0])


# --- Когортный разрыв ------------------------------------------------------

def test_cohort_rate_is_sum_over_sum_not_mean_of_ratios():
    """Σ числителей / Σ знаменателей — правило проекта, а не вариант оформления.

    Здесь среднее долей дало бы 0.75, когортная доля — 0.60. Крупный цикл обязан
    весить больше мелкого.
    """
    frame = _frame([
        {"age": 33.0, "amh": 2.0, "two_pn": 1.0, "mii": 1.0},     # доля 1.00
        {"age": 33.0, "amh": 2.0, "two_pn": 2.0, "mii": 4.0},     # доля 0.50
    ])
    row = prior.gap_table(frame).set_index("metric").loc["FR"]
    assert row["observed"] == pytest.approx(0.6)


def test_rows_with_ratio_above_one_are_excluded():
    """MII>ОКК — добавленные размороженные ооциты, а не порча данных.

    В когортную долю такая строка не идёт: доля вне [0,1] не сравнима с приором.
    """
    frame = _frame([
        {"age": 33.0, "amh": 2.0, "mii": 13.0, "occ": 12.0},
        {"age": 33.0, "amh": 2.0, "mii": 8.0, "occ": 10.0},
    ])
    row = prior.gap_table(frame).set_index("metric").loc["MIIR"]
    assert row["n_cycles"] == 1 and row["observed"] == pytest.approx(0.8)


def test_gap_is_our_rate_minus_literature_in_points():
    frame = _frame([{"age": 33.0, "amh": 2.0, "two_pn": 7.0, "mii": 10.0}])
    row = prior.gap_table(frame).set_index("metric").loc["FR"]
    assert row["gap_pp"] == pytest.approx((row["observed"] - row["literature"]) * 100,
                                          abs=0.05)


def test_metric_without_usable_rows_says_so_instead_of_vanishing():
    """Невозможность посчитать приор — это результат, и он обязан быть виден.

    Метрика, тихо выпавшая из таблицы, читается как «всё в порядке».
    """
    frame = _frame([{"age": None, "amh": None, "two_pn": 7.0, "mii": 10.0}])
    row = prior.gap_table(frame).set_index("metric").loc["FR"]
    assert row["literature"] is None and row["gap_pp"] is None
    assert "приор не посчитан" in row["note"]


def test_gap_table_survives_a_frame_without_columns():
    """Кадр без колонок — обычное дело на пустой таблице, а не исключительный случай.

    Раньше здесь падало на `.fillna` у скаляра: сообщение про numpy.float64 не
    содержало ни имени колонки, ни намёка на то, что циклов просто нет.
    """
    gaps = prior.gap_table(pd.DataFrame(columns=["cycle_uid"]), amh_constant=2.0)
    assert len(gaps) == len(prior.PARTS)
    assert gaps["observed"].isna().all()
    assert (gaps["n_cycles"] == 0).all()


def test_every_metric_keeps_its_source():
    """Происхождение числа не теряется: у каждой строки есть источник формулы."""
    frame = _frame([{"age": 33.0, "amh": 2.0, "two_pn": 7.0, "mii": 10.0,
                     "occ": 12.0, "bl_total": 5.0}])
    gaps = prior.gap_table(frame)
    assert gaps["source"].notna().all()
    assert set(gaps["metric"]) == set(prior.PARTS)


# --- Пригодность для риск-коррекции ---------------------------------------

def test_prior_without_discrimination_is_not_usable_for_risk_adjustment():
    """Разрыв уровня — не препятствие, отсутствие дискриминации — препятствие.

    Систематическое смещение снимается сдвигом; приор, не различающий пациенток,
    подставлять вместо базовой доли бессмысленно.
    """
    gaps = pd.DataFrame([{"metric": "MIIR", "corr_rate": 0.02},
                         {"metric": "FR", "corr_rate": 0.55},
                         {"metric": "BlR", "corr_rate": np.nan}])
    usable = prior.usable_for_risk_adjustment(gaps)
    assert usable == {"MIIR": False, "FR": True, "BlR": False}


def test_no_gaps_means_no_verdict():
    assert prior.usable_for_risk_adjustment(pd.DataFrame()) == {}
