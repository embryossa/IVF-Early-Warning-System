"""Проверки секвенциального слоя: свойства, которые обязаны держаться."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import charts


def test_weight_drift_signs():
    """В контроле снос отрицательный, при настроенном сдвиге — положительный."""
    p0, R = 0.59, 0.5
    in_control = charts.cusum_weight(np.array([1.0]), np.array([p0]), R)[0] * p0 + \
        charts.cusum_weight(np.array([0.0]), np.array([p0]), R)[0] * (1 - p0)
    shifted_p = p0 * R / (1 - p0 + R * p0)
    out_control = charts.cusum_weight(np.array([1.0]), np.array([p0]), R)[0] * shifted_p + \
        charts.cusum_weight(np.array([0.0]), np.array([p0]), R)[0] * (1 - shifted_p)
    assert in_control < 0, "карта обязана дрейфовать вниз, пока всё в норме"
    assert out_control > 0, "карта обязана копить при сдвиге, на который настроена"


def test_cusum_never_negative():
    rng = np.random.default_rng(1)
    outcome = rng.random(500) < 0.6
    path = charts.cusum_path(outcome, np.full(500, 0.6), 0.5)
    assert (path >= 0).all()


def _alerts_at(h, rng_seed=2):
    rng = np.random.default_rng(rng_seed)
    runner = charts.CusumRunner()
    for _ in range(300):
        trials = 8
        successes = int(rng.binomial(trials, 0.45))   # хуже базовой 0.59
        runner.step_block(successes, trials, 0.59, 0.5, None, h)
    return runner.alert_signals


def test_signal_count_monotone_in_h():
    """Чем выше порог, тем меньше сигналов — на этом стоит бисекция в solve_thresholds."""
    counts = [_alerts_at(h) for h in (1, 2, 4, 8, 16)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > 0, "при сильном сдвиге низкий порог обязан сигналить"


def test_block_update_equals_sum_of_bernoulli_weights():
    """Поцикловое обновление даёт тот же ВКЛАД, что и сумма по эмбрионам.

    Различается только траектория: у блока нет внутрицикловых пиков, которых
    в реальности никто не наблюдает (порядок эмбрионов не известен).
    """
    successes, trials, p, R = 5.0, 8.0, 0.59, 0.5
    outcomes = np.concatenate([np.ones(int(successes)), np.zeros(int(trials - successes))])
    per_embryo = charts.cusum_weight(outcomes, np.full(len(outcomes), p), R).sum()
    block = charts.CusumRunner()
    block.step_block(successes, trials, p, R, None, None)
    assert abs(block.statistic - max(0.0, per_embryo)) < 1e-9


def test_watch_episode_is_edge_triggered():
    """Пока статистика висит выше порога, новых эпизодов watch не возникает."""
    runner = charts.CusumRunner()
    events = [runner.step_block(0, 6, 0.59, 0.5, 1.0, 99.0) for _ in range(5)]
    assert events.count("watch") == 1, events
    assert runner.watch_episodes == 1


def test_upper_control_direction():
    """Для верхнего контроля успех — нежелательное событие, R>1 копит при росте."""
    p0, R = 0.05, 2.0
    weight_event = charts.cusum_weight(np.array([1.0]), np.array([p0]), R)[0]
    weight_none = charts.cusum_weight(np.array([0.0]), np.array([p0]), R)[0]
    assert weight_event > 0 and weight_none < 0


def test_parameter_uncertainty_widens_outcomes():
    """Розыгрыш истинной доли обязан менять картину против фиксированной."""
    rng = np.random.default_rng(3)
    fixed = charts.simulate_false_alarm_rate(
        charts.fixed_sampler(0.59), 0.59, 0.5, 3.0, horizon=1000, replicates=300, rng=rng)
    uncertain = charts.simulate_false_alarm_rate(
        charts.jeffreys_sampler(184, 312), 0.59, 0.5, 3.0, horizon=1000, replicates=300, rng=rng)
    assert uncertain.never_signalled > fixed.never_signalled, (
        "при неизвестном центре доля 'тихих' лет обязана расти: часть лабораторий "
        "окажется лучше базовой оценки и не сигналит вовсе"
    )


@pytest.mark.parametrize("target", [0.5, 2.0, 8.0])
def test_solve_h_hits_target(target):
    rng = np.random.default_rng(4)
    h, result = charts.solve_h(charts.jeffreys_sampler(184, 312), 0.59, 0.5,
                               target_rate=target, horizon=1095, replicates=400, rng=rng)
    assert abs(result.rate - target) / target < 0.35, (h, result.rate)


def test_beta_binomial_caps_confident_misses():
    """Уверенный и ошибшийся прогноз не должен весить как двадцать наблюдений.

    Именно этим бернуллиевская подстановка P50/n испортила калибровку: при p=0.99
    и n=7 дисперсия вырождается, и промах даёт вклад 124 при медианном 5.
    """
    successes, trials, mean = 4.0, 7.0, 0.99
    bernoulli = abs(successes * np.log(0.5) - trials * np.log(1 - mean + 0.5 * mean))
    beta_binom = abs(charts.beta_binomial_weight(successes, trials, mean,
                                                 concentration=3.0, odds_ratio=0.5))
    assert beta_binom < bernoulli, (beta_binom, bernoulli)


def test_wide_interval_shrinks_toward_baseline():
    """Чем шире интервал конверта, тем меньше карта доверяет его центру."""
    narrow = charts.concentration_from_interval(p10=4.5, p50=5.0, p90=5.5, trials=8)
    wide = charts.concentration_from_interval(p10=1.0, p50=5.0, p90=8.0, trials=8)
    assert narrow > wide, (narrow, wide)
    assert wide >= 2.0, "концентрация не должна опускаться ниже равномерной"

    # При низкой концентрации вес приближается к нулю: карта почти не реагирует.
    confident = abs(charts.beta_binomial_weight(2, 8, 0.6, narrow, 0.5))
    uncertain = abs(charts.beta_binomial_weight(2, 8, 0.6, wide, 0.5))
    assert uncertain < confident


def test_degenerate_interval_gives_floor_concentration():
    assert charts.concentration_from_interval(5.0, 5.0, 5.0, 8) >= 2.0
    assert charts.concentration_from_interval(0.0, 0.0, 0.0, 0) == 2.0


# --- Совместное прочтение контуров -----------------------------------------

def _state(**kw):
    from ews.verdict import ContourState
    base = dict(metric="BlR", relative_triggered=True, absolute_status="benchmark",
                literature_gap_pp=0.0)
    base.update(kw)
    return ContourState(**base)


def test_trend_when_above_consensus():
    """Просадка относительно себя при уровне выше консенсуса — НЕ разбор протокола.

    Неподвижное решение владельца: лаборатория, укладывающаяся в консенсус и в
    литературу, не может считаться провалившей протокол только потому, что стала
    отличаться от собственной истории.
    """
    from ews import verdict

    result = verdict.read(_state())
    assert result.action == verdict.ACTION_REVIEW_CASEMIX
    assert "не провал" in result.reasoning


def test_protocol_review_only_when_below_competency():
    from ews import verdict

    result = verdict.read(_state(absolute_status="below", literature_gap_pp=-12.0))
    assert result.action == verdict.ACTION_REVIEW_PROTOCOL


def test_silent_chart_never_triggers_protocol_review():
    """Молчащая карта не может дать разбор протокола, даже если уровень низкий.

    Низкий уровень без изменения — давняя разница, а не событие раннего
    предупреждения; смешивать их значит превращать систему в постоянный сигнал.
    """
    from ews import verdict

    result = verdict.read(_state(relative_triggered=False, absolute_status="below",
                                 literature_gap_pp=-15.0))
    assert result.action == verdict.ACTION_STANDING_GAP


def test_literature_tolerance_is_respected():
    from ews import verdict

    mild = verdict.read(_state(literature_gap_pp=-3.0))
    assert mild.action == verdict.ACTION_REVIEW_CASEMIX
