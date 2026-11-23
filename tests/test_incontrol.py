"""Опорные параметры: центр относительного контура и его оговорки.

Задача модуля — не оспорить объявленное владельцем опорное окно, а зафиксировать
его последствия. Главное из них: если внутри окна метрика значимо отличается от
того, что было до него, окно вобрало уже случившийся сдвиг в норму, и карта на этой
метрике перестала его видеть. Оговорка обязана дожить до манифеста калибровки —
иначе «в контроле» читается как «в порядке» ровно там, где это неверно.
"""
import numpy as np
import pandas as pd
import pytest

from ews import incontrol


def _kpi(metric="BlR", numerator=180.0, denominator=300.0, direction="lower",
         competency=0.40, benchmark=0.60, status="benchmark") -> pd.DataFrame:
    return pd.DataFrame([{
        "metric": metric, "numerator": numerator, "denominator": denominator,
        "rate": numerator / denominator if denominator else float("nan"),
        "direction": direction, "competency": competency,
        "benchmark": benchmark, "status": status,
    }])


# --- Доверительный интервал ------------------------------------------------

def test_wilson_brackets_the_point_estimate():
    low, high = incontrol._wilson(180, 300)
    assert low < 0.6 < high
    assert 0.0 <= low and high <= 1.0


def test_wilson_is_undefined_when_numerator_exceeds_denominator():
    """ClR>100% и MII>ОКК — штатные ситуации, но интервал на них не определён.

    Правило проекта: доля остаётся, ci_low/ci_high = NaN. Посчитать «что-нибудь»
    здесь означало бы выдать за интервал число без смысла.
    """
    low, high = incontrol._wilson(13, 12)
    assert np.isnan(low) and np.isnan(high)


def test_wilson_is_undefined_without_denominator():
    assert all(np.isnan(v) for v in incontrol._wilson(0, 0))


def test_wilson_handles_the_boundaries():
    """Ноль успехов и сплошные успехи — законные наблюдения, а не ошибка."""
    assert incontrol._wilson(0, 50)[0] == 0.0
    assert incontrol._wilson(50, 50)[1] == 1.0


# --- Сравнение с периодом до окна -----------------------------------------

def test_significant_worsening_is_marked_absorbed():
    """Окно вобрало сдвиг — и это обязано быть сказано вслух.

    Карта после этого центрирована на новом уровне и прошлый сдвиг не увидит; она
    сработает только на дальнейшем ухудшении.
    """
    window = _kpi(numerator=150.0, denominator=300.0)          # 0.50
    prior = _kpi(numerator=240.0, denominator=300.0)           # 0.80
    base = incontrol.build_baselines(window, prior)[0]
    assert base.shift_absorbed is True
    assert base.delta_pp == pytest.approx(-30.0) and base.p_value < 0.05
    notes = incontrol.caveats([base])
    assert any("вобрало уже случившийся сдвиг" in note for note in notes)


def test_improvement_is_not_absorbed():
    """Улучшение оговорки не требует: карта не слепнет к ухудшению от того, что
    лаборатория стала работать лучше."""
    window = _kpi(numerator=240.0, denominator=300.0)
    prior = _kpi(numerator=150.0, denominator=300.0)
    base = incontrol.build_baselines(window, prior)[0]
    assert base.delta_pp == pytest.approx(30.0) and base.shift_absorbed is False


def test_direction_upper_reverses_what_counts_as_worsening():
    """У метрик верхнего контроля (DegR, 3pNR) ухудшение — это РОСТ доли."""
    window = _kpi(metric="DegR", numerator=60.0, denominator=300.0, direction="upper",
                  benchmark=0.10, competency=0.20, status="below")
    prior = _kpi(metric="DegR", numerator=15.0, denominator=300.0, direction="upper",
                 benchmark=0.10, competency=0.20, status="benchmark")
    base = incontrol.build_baselines(window, prior)[0]
    assert base.delta_pp > 0 and base.shift_absorbed is True


def test_insignificant_change_is_not_absorbed():
    window = _kpi(numerator=181.0, denominator=300.0)
    prior = _kpi(numerator=180.0, denominator=300.0)
    base = incontrol.build_baselines(window, prior)[0]
    assert base.shift_absorbed is False


def test_without_prior_period_there_is_no_comparison():
    base = incontrol.build_baselines(_kpi(), None)[0]
    assert base.prior_rate is None and base.p_value is None
    assert base.shift_absorbed is False


def test_fisher_is_skipped_when_the_table_is_impossible():
    """Числитель больше знаменателя — таблица 2x2 не существует, тест не считается."""
    assert incontrol._fisher(13, 12, 10, 12) is None
    assert incontrol._fisher(None, 12, 10, 12) is None


# --- Оговорки об абсолютном уровне ----------------------------------------

def test_no_caveat_when_benchmark_lies_inside_the_interval():
    """Точечная оценка на волосок ниже порога — не «ниже порога».

    Иначе оговорка срабатывала бы почти на каждой метрике и обесценила бы весь
    список: BlR 0.590 при CI 0.534-0.643 не является «ниже 0.60» ни в каком
    полезном смысле.
    """
    window = _kpi(numerator=177.0, denominator=300.0, benchmark=0.60)   # 0.590
    base = incontrol.build_baselines(window, None)[0]
    assert base.ci_low < 0.60 < base.ci_high
    assert incontrol.caveats([base]) == []


def test_caveat_when_benchmark_lies_outside_the_interval():
    window = _kpi(numerator=90.0, denominator=300.0, benchmark=0.60, status="below")
    notes = incontrol.caveats(incontrol.build_baselines(window, None))
    assert any("лежит вне интервала" in note for note in notes)


def test_caveat_is_skipped_without_a_benchmark():
    window = _kpi(benchmark=None)
    assert incontrol.caveats(incontrol.build_baselines(window, None)) == []


# --- Манифест -------------------------------------------------------------

def test_manifest_carries_caveats_forward():
    """Оговорки обязаны дожить до манифеста: дальше их читает калибровка."""
    window = _kpi(numerator=150.0, denominator=300.0)
    prior = _kpi(numerator=240.0, denominator=300.0)
    manifest = incontrol.as_manifest(
        incontrol.build_baselines(window, prior),
        window_start="2026-05-01", window_end=None, n_cycles=37,
        n_embryos=312.0, declared_by="owner")
    assert manifest["baseline_caveats"]
    assert manifest["reference_window"]["declared_by"] == "owner"
    assert manifest["reference_window"]["end"], "открытое окно закрывается сегодняшним днём"
    assert manifest["metrics"][0]["metric"] == "BlR"
