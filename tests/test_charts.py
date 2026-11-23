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


def test_alert_crossing_resets_the_statistic():
    """После сигнала alert счётчик обнуляется — тревога разобрана, копим заново.

    Watch устроен иначе и статистику НЕ сбрасывает: это два разных события, и
    калибровка считает их той же реализацией, что и боевой прогон.
    """
    runner = charts.CusumRunner()
    assert runner.step_block(0, 20, 0.6, 0.5, 1.0, 3.0) == "alert"
    assert runner.statistic == 0.0
    assert runner.alert_signals == 1 and runner.above_watch is False


def test_watch_episode_closes_only_at_zero():
    """Эпизод watch закрывается разрядкой в ноль, а не возвратом под порог.

    Иначе на чередующихся исходах статистика болталась бы вокруг порога и рождала
    новый «эпизод» на каждом колебании.
    """
    runner = charts.CusumRunner()
    assert runner.step_block(0, 6, 0.59, 0.5, 1.0, 99.0) == "watch"
    assert runner.above_watch is True
    for _ in range(20):                      # длинная серия успехов разряжает карту
        runner.step_block(6, 6, 0.59, 0.5, 1.0, 99.0)
    assert runner.statistic == 0.0 and runner.above_watch is False
    assert runner.step_block(0, 6, 0.59, 0.5, 1.0, 99.0) == "watch"
    assert runner.watch_episodes == 2


def test_simulation_refuses_impossible_block_sizes():
    """Ни одного положительного размера цикла — это ошибка, а не бесконечный цикл.

    Набор наблюдений отбрасывает неположительные размеры и берёт следующий; без
    положительных он не закончится никогда, и задача планировщика упрётся в
    двухчасовой лимит с отчётом «модель считает долго».
    """
    with pytest.raises(ValueError, match="положительного размера"):
        charts.simulate_event_rates(charts.fixed_sampler(0.6), 0.6, 0.5, 1.0, 3.0,
                                    horizon=10, replicates=1,
                                    rng=np.random.default_rng(0),
                                    block_sizes=np.array([0, -1]))


def test_alert_crossing_keeps_the_value_it_crossed_on():
    """Статистика обнуляется, `attained` — нет.

    Карточка тревоги обязана нести величину, НА КОТОРОЙ сработала карта. Раньше её
    читали из `statistic` уже ПОСЛЕ обнуления, и в боевом журнале лежала карточка
    уровня alert со статистикой 0.0 — число, опровергающее собственную тревогу.
    Пик и траектория страдали так же: панель печатала «сигналов alert 1, пик 6.25
    при пороге 7.44», то есть пик ниже порога, который был пересечён.
    """
    runner = charts.CusumRunner()
    event = runner.step_block(successes=0, trials=40, expected=0.8,
                              odds_ratio=0.5, h_watch=1.0, h_alert=5.0)
    assert event == "alert"
    assert runner.statistic == 0.0        # карта перезапущена — это верно
    assert runner.attained > 5.0          # но экскурсия сохранена


def test_attained_equals_statistic_when_nothing_crossed():
    """Без сигнала два поля совпадают: новое поле не меняет обычный ход карты."""
    runner = charts.CusumRunner()
    for _ in range(3):
        runner.step_block(successes=3, trials=4, expected=0.8, odds_ratio=0.5,
                          h_watch=100.0, h_alert=200.0)
        assert runner.attained == runner.statistic


def test_watch_crossing_does_not_reset_anything():
    """`watch` карту не перезапускает — семантика уровней не тронута."""
    runner = charts.CusumRunner()
    event = runner.step_block(successes=0, trials=10, expected=0.8,
                              odds_ratio=0.5, h_watch=1.0, h_alert=1000.0)
    assert event == "watch"
    assert runner.statistic == runner.attained > 1.0
