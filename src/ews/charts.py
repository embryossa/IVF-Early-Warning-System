"""Секвенциальные карты EWS: risk-adjusted CUSUM.

Steiner SH, Cook RJ, Farewell VT, Treasure T. Monitoring surgical performance using
risk-adjusted cumulative sum charts. Biostatistics 2000;1(4):441-452.

Единица наблюдения — эмбрион (или ооцит), не цикл: при ~70 наблюдениях в месяц на
уровне циклов мощности не остаётся вовсе.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cusum_weight(outcome: np.ndarray, expected: np.ndarray, odds_ratio: float) -> np.ndarray:
    """Вклад наблюдения в статистику CUSUM при сдвиге шансов в ``odds_ratio`` раз.

    Логарифм отношения правдоподобий Бернулли: под H0 вероятность успеха ``expected``,
    под H1 шансы умножены на R, то есть p1 = R*p / (1 - p + R*p).

        W = y * log(R) - log(1 - p + R * p)

    Для детекции ухудшения метрики нижнего контроля берётся R < 1 (например 0.5 —
    половинные шансы успеха): тогда W > 0 на неудаче и W < 0 на успехе, и статистика
    копится ровно там, где успехов меньше ожидаемого.
    """
    p = np.clip(np.asarray(expected, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(outcome, dtype=float)
    return y * np.log(odds_ratio) - np.log(1.0 - p + odds_ratio * p)


def cusum_path(outcome: np.ndarray, expected: np.ndarray,
               odds_ratio: float, start: float = 0.0) -> np.ndarray:
    """Траектория C_i = max(0, C_{i-1} + W_i) без перезапуска при сигнале."""
    weights = cusum_weight(outcome, expected, odds_ratio)
    path = np.empty(len(weights), dtype=float)
    current = float(start)
    for index, weight in enumerate(weights):
        current = max(0.0, current + weight)
        path[index] = current
    return path


def run_length(outcome: np.ndarray, expected: np.ndarray, odds_ratio: float,
               h: float) -> int | None:
    """Число наблюдений до первого пересечения порога; None — сигнала не было."""
    weights = cusum_weight(outcome, expected, odds_ratio)
    current = 0.0
    for index, weight in enumerate(weights):
        current = max(0.0, current + weight)
        if current > h:
            return index + 1
    return None


def beta_binomial_weight(successes: float, trials: float, mean: float,
                         concentration: float, odds_ratio: float) -> float:
    """Вклад цикла при НЕТОЧНО известном риске: бета-биномиальное правдоподобие.

    Обычный бернуллиевский вес берёт ``p`` как известную величину. Для риска,
    пришедшего из прогноза, это неверно и дорого: при уверенном прогнозе
    (``p`` около 0 или 1) биномиальная дисперсия ``n·p·(1-p)`` вырождается, и один
    промах модели весит как двадцать обычных наблюдений. Измерено на опорном окне:
    прямая подстановка ``P50/n`` подняла X²/df с 2.33 до 11.92 (см. METHODOLOGY §5.8).

    Здесь риск цикла считается величиной ``p ~ Beta(a, b)`` со средним ``mean`` и
    концентрацией ``concentration = a + b``, взятой из ширины интервала P10-P90:
    узкий интервал — высокая концентрация и поведение, близкое к бернуллиевскому;
    широкий — низкая, и карта автоматически сдвигается к базовой доле.

    Отношение правдоподобий двух бета-биномиальных моделей — с исходными шансами и
    со сдвинутыми в ``odds_ratio`` раз:

        W = log BetaBinom(x | n, a1, b1) - log BetaBinom(x | n, a0, b0)

    Биномиальный коэффициент сокращается, поэтому считаются только бета-функции.
    """
    from scipy.special import betaln  # noqa: PLC0415

    mean = min(max(float(mean), 1e-4), 1 - 1e-4)
    concentration = max(float(concentration), 2.0)
    shifted_mean = mean * odds_ratio / (1.0 - mean + odds_ratio * mean)

    a0, b0 = mean * concentration, (1.0 - mean) * concentration
    a1, b1 = shifted_mean * concentration, (1.0 - shifted_mean) * concentration

    log_h1 = betaln(successes + a1, trials - successes + b1) - betaln(a1, b1)
    log_h0 = betaln(successes + a0, trials - successes + b0) - betaln(a0, b0)
    return float(log_h1 - log_h0)


def concentration_from_interval(p10: float, p50: float, p90: float,
                                trials: float) -> float:
    """Концентрация Beta из ширины предиктивного интервала конверта.

    Интервал P10-P90 счётной величины переводится в долю делением на знаменатель;
    его полуширина сопоставляется с sd бета-распределения. Для Beta(a,b) при
    концентрации ``k``: var = p(1-p)/(k+1), то есть ``k = p(1-p)/var - 1``.

    Нижняя граница 2 — вырожденный, максимально неинформативный случай (равномерное
    распределение риска); он же безопасный: карта тогда почти не отличает сдвиг и
    молчит, вместо того чтобы кричать на уверенном промахе модели.
    """
    if trials <= 0:
        return 2.0
    p = min(max(p50 / trials, 1e-4), 1 - 1e-4)
    half_width = max((p90 - p10) / (2.0 * trials), 1e-6)
    # P10-P90 покрывает 1.2816 sd в каждую сторону при нормальном приближении.
    sd = half_width / 1.2816
    variance = max(sd * sd, 1e-9)
    return max(p * (1.0 - p) / variance - 1.0, 2.0)


@dataclass
class CusumRunner:
    """Счётчик событий карты. ОДНА реализация на симуляцию и на боевой прогон.

    Разделять их нельзя: если калибровка считает одно событие, а рантайм поднимает
    другое, измеренная частота ложных тревог не имеет отношения к тому, что увидит
    лаборатория. Здесь зафиксированы обе семантики, и они различны:

    * `alert` — пересечение h_alert, статистика ПЕРЕЗАПУСКАЕТСЯ (после разбора
      тревоги счётчик обнуляется);
    * `watch` — пересечение h_watch снизу вверх, статистика НЕ перезапускается.
      Эпизод закрывается, только когда статистика разрядилась в ноль. Закрывать его
      по простому возврату ниже порога нельзя: на чередующихся исходах статистика
      болтается вокруг порога и породила бы новый «эпизод» на каждом колебании.
    """
    statistic: float = 0.0
    above_watch: bool = False
    observations: int = 0
    last_zero_index: int = 0
    watch_episodes: int = 0
    alert_signals: int = 0

    def step(self, outcome: float, expected: float, odds_ratio: float,
             h_watch: float | None, h_alert: float | None) -> str | None:
        p = min(max(float(expected), 1e-6), 1 - 1e-6)
        self.statistic = max(0.0, self.statistic
                             + outcome * np.log(odds_ratio)
                             - np.log(1.0 - p + odds_ratio * p))
        self.observations += 1
        if self.statistic == 0.0:
            self.last_zero_index = self.observations

        if h_alert is not None and self.statistic > h_alert:
            self.alert_signals += 1
            self.statistic = 0.0
            self.above_watch = False
            self.last_zero_index = self.observations
            return "alert"
        if h_watch is not None:
            if self.statistic > h_watch and not self.above_watch:
                self.above_watch = True
                self.watch_episodes += 1
                return "watch"
            if self.statistic == 0.0:
                self.above_watch = False
        return None

    def step_block(self, successes: float, trials: float, expected: float,
                   odds_ratio: float, h_watch: float | None,
                   h_alert: float | None) -> str | None:
        """Одно обновление на ЦИКЛ: суммарный вклад всех его эмбрионов сразу.

        Так устроено поступление информации: исходы эмбрионов одного цикла
        становятся известны одновременно, на закрытии. Пошагово внутри цикла карту
        двигать нельзя — порядок эмбрионов не наблюдаем, а любой фиксированный
        порядок создаёт артефакт. Разворачивание «сначала все успехи, потом все
        неудачи» давало максимальную экскурсию в конце каждого цикла и завышало
        частоту срабатываний в 2.7 раза против калиброванной.

        Вклад тот же самый — сумма бернуллиевских весов, — но траектория не имеет
        внутрицикловых пиков, которых в реальности никто не наблюдает.
        """
        p = min(max(float(expected), 1e-6), 1 - 1e-6)
        weight = (successes * np.log(odds_ratio)
                  - trials * np.log(1.0 - p + odds_ratio * p))
        self.statistic = max(0.0, self.statistic + weight)
        self.observations += int(trials)
        if self.statistic == 0.0:
            self.last_zero_index = self.observations

        if h_alert is not None and self.statistic > h_alert:
            self.alert_signals += 1
            self.statistic = 0.0
            self.above_watch = False
            self.last_zero_index = self.observations
            return "alert"
        if h_watch is not None:
            if self.statistic > h_watch and not self.above_watch:
                self.above_watch = True
                self.watch_episodes += 1
                return "watch"
            if self.statistic == 0.0:
                self.above_watch = False
        return None

    def run(self, outcomes: np.ndarray, expected: np.ndarray, odds_ratio: float,
            h_watch: float | None, h_alert: float | None) -> list[str]:
        events = []
        for outcome, exp in zip(outcomes, expected):
            event = self.step(float(outcome), float(exp), odds_ratio, h_watch, h_alert)
            if event:
                events.append(event)
        return events


@dataclass
class RateResult:
    h: float
    rate: float              # ожидаемое число сигналов за горизонт
    never_signalled: float   # доля прогонов без единого сигнала
    n_replicates: int
    horizon: int


def simulate_event_rates(p_true_sampler, p_chart: float, odds_ratio: float,
                         h_watch: float | None, h_alert: float | None,
                         horizon: int, replicates: int, rng: np.random.Generator,
                         block_sizes: np.ndarray | None = None,
                         dispersion: float = 0.0) -> dict[str, float]:
    """Ожидаемое число событий каждого уровня за ``horizon`` наблюдений (год).

    Оба порога активны одновременно, как в бою: перезапуск по alert влияет на то,
    сколько эпизодов watch успеет случиться, и считать их по отдельности значило бы
    моделировать процесс, которого не существует.

    ``block_sizes`` — эмпирическое распределение размеров цикла (знаменателей).
    Симуляция обязана двигать карту так же, как рантайм: по одному обновлению на
    цикл. Поцикловые блоки — не деталь реализации, а свойство поступления данных, и
    от них заметно зависит частота срабатываний.

    ``dispersion`` — внутриклассовая корреляция rho сверхбиномиальной вариации между
    циклами. У этой лаборатории она измерена и велика (X2/df = 1.8-2.8): доля успеха
    заметно различается от пациентки к пациентке сверх биномиального шума. Если её не
    моделировать, карта видит разброс, которого нет в модели, и пересекает порог
    втрое чаще предсказанного. Правильное лекарство — риск-коррекция p_i по конверту
    (она эту разнородность поглощает); пока конверт не зарегистрирован для всех
    циклов, вариация вносится в саму калибровку явно.
    """
    watch = np.zeros(replicates)
    alert = np.zeros(replicates)
    sizes = None if block_sizes is None or len(block_sizes) == 0 else np.asarray(block_sizes)

    for index in range(replicates):
        p_true = float(p_true_sampler(rng))
        runner = CusumRunner()
        if sizes is None:
            outcomes = (rng.random(horizon) < p_true).astype(float)
            runner.run(outcomes, np.full(horizon, p_chart), odds_ratio, h_watch, h_alert)
        else:
            observed = 0
            while observed < horizon:
                trials = int(sizes[rng.integers(len(sizes))])
                if trials <= 0:
                    continue
                if dispersion > 0:
                    # Бета-биномиальная модель цикла: rho = 1/(a+b+1).
                    concentration = max((1.0 - dispersion) / dispersion, 1e-6)
                    p_cycle = float(rng.beta(p_true * concentration,
                                             (1 - p_true) * concentration))
                else:
                    p_cycle = p_true
                successes = int(rng.binomial(trials, p_cycle))
                runner.step_block(successes, trials, p_chart, odds_ratio,
                                  h_watch, h_alert)
                observed += trials
        watch[index] = runner.watch_episodes
        alert[index] = runner.alert_signals
    return {"watch": float(watch.mean()), "alert": float(alert.mean()),
            "quiet_years": float((watch + alert == 0).mean())}


def solve_thresholds(p_true_sampler, p_chart: float, odds_ratio: float,
                     target_watch: float, target_alert: float, horizon: int,
                     replicates: int, rng: np.random.Generator,
                     bounds: tuple[float, float] = (0.2, 60.0),
                     tolerance: float = 0.05, max_iter: int = 20,
                     block_sizes: np.ndarray | None = None,
                     dispersion: float = 0.0) -> dict:
    """Подобрать пару (h_watch, h_alert) под целевые частоты СВОИХ событий.

    Порядок важен: сначала h_alert (его события редки и от h_watch практически не
    зависят — watch статистику не сбрасывает), затем h_watch при уже
    зафиксированном h_alert.
    """
    low, high = bounds
    h_alert = float("nan")
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        rates = simulate_event_rates(p_true_sampler, p_chart, odds_ratio,
                                     None, mid, horizon, replicates, rng,
                                     block_sizes=block_sizes, dispersion=dispersion)
        h_alert = mid
        if target_alert and abs(rates["alert"] - target_alert) / target_alert <= tolerance:
            break
        if rates["alert"] > target_alert:
            low = mid
        else:
            high = mid

    low, high = bounds[0], h_alert
    h_watch = float("nan")
    final: dict[str, float] = {}
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        rates = simulate_event_rates(p_true_sampler, p_chart, odds_ratio,
                                     mid, h_alert, horizon, replicates, rng,
                                     block_sizes=block_sizes, dispersion=dispersion)
        h_watch, final = mid, rates
        if target_watch and abs(rates["watch"] - target_watch) / target_watch <= tolerance:
            break
        if rates["watch"] > target_watch:
            low = mid
        else:
            high = mid

    return {"h_watch": h_watch, "h_alert": h_alert, "rates": final}


def simulate_false_alarm_rate(p_true_sampler, p_chart: float, odds_ratio: float,
                              h: float, horizon: int, replicates: int,
                              rng: np.random.Generator) -> RateResult:
    """Ожидаемое число сигналов за ``horizon`` наблюдений (то есть за год).

    Считается именно ЧАСТОТА, а не средняя длина серии до первого сигнала. Разница
    принципиальна, как только истинная доля разыгрывается: у прогонов, где реальность
    оказалась лучше базовой оценки, время до сигнала фактически бесконечно, и среднее
    упирается в горизонт симуляции вместо того, чтобы отражать реальность. По
    неравенству Йенсена E[1/ARL] != 1/E[ARL], а цель поставлена именно в терминах
    E[1/ARL] — «сколько ложных тревог в год увидит лаборатория».

    ``p_chart`` — доля, которую карта СЧИТАЕТ базовой (оценка по опорному окну).
    ``p_true_sampler`` даёт истинную долю прогона: розыгрыш из апостериорного
    распределения учитывает, что центр известен неточно.
    """
    expected = np.full(horizon, p_chart, dtype=float)
    counts = np.empty(replicates, dtype=float)
    for index in range(replicates):
        p_true = float(p_true_sampler(rng))
        outcomes = (rng.random(horizon) < p_true).astype(float)
        runner = CusumRunner()
        runner.run(outcomes, expected, odds_ratio, None, h)
        counts[index] = runner.alert_signals
    return RateResult(h=h, rate=float(counts.mean()),
                      never_signalled=float((counts == 0).mean()),
                      n_replicates=replicates, horizon=horizon)


def simulate_detection_delay(p_shifted: float, p_chart: float, odds_ratio: float,
                             h: float, horizon: int, replicates: int,
                             rng: np.random.Generator) -> tuple[float, float]:
    """(среднее, медиана) числа наблюдений до сигнала при НАСТОЯЩЕМ сдвиге.

    Здесь цензурирования почти нет: карта настроена ровно на этот сдвиг, и серии
    короткие. Прогоны без сигнала учитываются как горизонт — это завышает оценку
    задержки, то есть ошибается в консервативную сторону.
    """
    expected = np.full(horizon, p_chart, dtype=float)
    lengths = np.empty(replicates, dtype=float)
    for index in range(replicates):
        outcome = rng.random(horizon) < p_shifted
        length = run_length(outcome, expected, odds_ratio, h)
        lengths[index] = horizon if length is None else length
    return float(lengths.mean()), float(np.median(lengths))


def solve_h(p_true_sampler, p_chart: float, odds_ratio: float,
            target_rate: float, horizon: int, replicates: int,
            rng: np.random.Generator, h_bounds: tuple[float, float] = (0.2, 60.0),
            tolerance: float = 0.03, max_iter: int = 22) -> tuple[float, RateResult]:
    """Подобрать h под целевую ЧАСТОТУ ложных сигналов за горизонт.

    Частота монотонно убывает по h, поэтому бисекция корректна. ``horizon`` — это
    объём наблюдений за год по данной метрике: цель задана в тревогах на год.
    """
    low, high = h_bounds
    best: RateResult | None = None
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        result = simulate_false_alarm_rate(p_true_sampler, p_chart, odds_ratio, mid,
                                           horizon, replicates, rng)
        best = result
        if target_rate and abs(result.rate - target_rate) / target_rate <= tolerance:
            break
        if result.rate > target_rate:
            low = mid          # сигналов слишком много -> порог поднять
        else:
            high = mid
    return (best.h if best else float("nan")), best


def jeffreys_sampler(successes: float, total: float):
    """Розыгрыш истинной доли из апостериорного Beta(x+1/2, n-x+1/2).

    Приор Джеффриса — стандартный неинформативный выбор для биномиальной доли; он же
    не схлопывается в точку при x=0 или x=n, что для DegR/3pNR принципиально.
    """
    alpha = max(successes + 0.5, 1e-6)
    beta = max(total - successes + 0.5, 1e-6)

    def sample(rng: np.random.Generator) -> float:
        return float(rng.beta(alpha, beta))

    return sample


def fixed_sampler(value: float):
    """Истинная доля известна точно — для сравнения «сколько стоит неопределённость»."""

    def sample(_: np.random.Generator) -> float:
        return float(value)

    return sample
