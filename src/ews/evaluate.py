"""Оценка на закрытии цикла: единственная точка, где возникают тревоги.

T0 и T+1 сюда не приходят вовсе — по неподвижному решению владельца они только
регистрируют данные. Здесь: обновление карт по эмбрионам, локализация контура,
подтверждение и решение ok / watch / alert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts, verdict as verdict_mod
from .clinic import Clinic
from .config import Config
from .cycles import METRIC_PARTS, metric_counts

LEVEL_OK = "ok"
LEVEL_WATCH = "watch"
LEVEL_ALERT = "alert"

RANK_NONE = "none"
RANK_MODERATE = "moderate"
RANK_STRONG = "strong"

# Метрика -> стадия конверта, дающая её риск-скорректированное ожидание.
# Числитель метрики = предсказанная стадия, знаменатель = фактический.
#   MIIR = MII/ОКК      -> mii                (регистрируется на T0)
#   FR   = 2PN/MII      -> pn2                (регистрируется на T0)
#   BlR  = Bl/2PN       -> blasts_after_fert  (регистрируется на T+1)
# TGBDR здесь СОЗНАТЕЛЬНО нет: стадия `good` обусловлена числом бластоцист, то есть
# ровно той величиной, которую мы мониторим. Такой «конверт» не является прогнозом
# исхода культивирования, и риск-коррекции для TGBDR не существует — метрика живёт
# только в абсолютном контуре.
ENVELOPE_STAGE = {"MIIR": "mii", "FR": "pn2", "BlR": "blasts_after_fert"}


@dataclass
class ChartState:
    """Состояние одной карты. Переживает перезапуск — иначе карта не секвенциальна.

    `above_watch` хранится вместе со статистикой намеренно: тревога уровня watch —
    это ЭПИЗОД (пересечение порога снизу вверх), а не свойство каждого цикла, пока
    статистика висит наверху. Без этого одна затяжная экскурсия давала бы watch на
    каждом закрывшемся цикле подряд, журнал превращался бы в шум, а измеренная
    калибровкой частота «24 в год» перестала бы соответствовать тому, что видит
    человек: симуляция считала именно пересечения.
    """
    metric: str
    statistic: float = 0.0
    observations: int = 0
    last_zero_index: int = 0        # оценка точки начала накопления
    signals: int = 0                # пересечений h_alert
    watch_episodes: int = 0         # пересечений h_watch снизу вверх
    above_watch: bool = False
    # Пик и траектория. Снимок статистики на конец периода почти всегда близок к
    # нулю — карта обнуляется после сигнала и разряжается в спокойные периоды.
    # Без пика и траектории панель показывает «0.00 при пороге 1.78» и читается как
    # «мы далеко от беды», хотя означает только «прямо сейчас ничего не копится».
    peak: float = 0.0
    trajectory: list = field(default_factory=list)   # (дата, статистика) по циклам

    TRAJECTORY_LIMIT = 400

    def record(self, when, value: float) -> None:
        """Точка траектории на закрытие цикла. Хранится ограниченное число."""
        self.peak = max(self.peak, value)
        self.trajectory.append([str(when)[:10], round(value, 3)])
        if len(self.trajectory) > self.TRAJECTORY_LIMIT:
            del self.trajectory[0]

    def to_dict(self) -> dict:
        return {"metric": self.metric, "statistic": round(self.statistic, 4),
                "observations": self.observations,
                "last_zero_index": self.last_zero_index, "signals": self.signals,
                "watch_episodes": self.watch_episodes, "above_watch": self.above_watch,
                "peak": round(self.peak, 4), "trajectory": self.trajectory}

    @classmethod
    def from_dict(cls, raw: dict) -> "ChartState":
        return cls(metric=raw["metric"], statistic=float(raw.get("statistic", 0.0)),
                   observations=int(raw.get("observations", 0)),
                   last_zero_index=int(raw.get("last_zero_index", 0)),
                   signals=int(raw.get("signals", 0)),
                   watch_episodes=int(raw.get("watch_episodes", 0)),
                   above_watch=bool(raw.get("above_watch", False)),
                   peak=float(raw.get("peak", 0.0)),
                   trajectory=list(raw.get("trajectory", [])))


@dataclass
class CycleEvaluation:
    cycle_uid: str
    closure_date: pd.Timestamp
    level: str = LEVEL_OK
    metrics: dict = field(default_factory=dict)      # metric -> {rate, num, den, ...}
    charts: dict = field(default_factory=dict)       # metric -> {statistic, crossed}
    contour: str | None = None
    corroboration: list = field(default_factory=list)
    corroboration_rank: str = RANK_NONE
    miri_status: str = "unavailable"
    notes: list = field(default_factory=list)
    # Совместное прочтение контуров: что тревога ЗНАЧИТ и что с ней делать.
    # Без этого слоя просадка относительно собственной истории читается как
    # «стало плохо», хотя у лаборатории выше консенсуса означает «стало иначе».
    verdicts: list = field(default_factory=list)
    verdict_summary: str = ""
    action: str = verdict_mod.ACTION_TREND


def load_state(path: Path) -> tuple[dict[str, ChartState], set[str]]:
    """Состояние карт и множество уже оценённых циклов.

    Второе обязательно: карта секвенциальна, и цикл, поданный в неё дважды, дважды
    двинет статистику. При суточном расписании без этой памяти каждый прогон
    прокручивал бы всю историю заново — за первый же dry-run число наблюдений
    выросло с 630 до 1260.
    """
    if not path.exists():
        return {}, set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    states = {name: ChartState.from_dict(entry)
              for name, entry in raw.get("charts", {}).items()}
    return states, set(raw.get("evaluated_cycles") or [])


def save_state(path: Path, states: dict[str, ChartState], cfg: Config,
               clinic: Clinic, evaluated: set[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clinic": clinic.code,
        "environment": cfg.environment,
        "site": cfg.site,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "charts": {name: state.to_dict() for name, state in states.items()},
        "evaluated_cycles": sorted(evaluated or []),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _contour_of(cfg: Config, metric: str) -> str | None:
    for contour, metrics in cfg.settings["localisation"]["contours"].items():
        if metric in metrics:
            return contour
    return None


def _expected_probability(row: pd.Series, metric: str, denominator: float,
                          baseline_rate: float, envelope: dict | None,
                          risk_adjustment_enabled: bool = False) -> tuple[float, str]:
    """Риск-скорректированное ожидание для одного цикла.

    Если для цикла есть замороженный конверт, ожидание берётся из него: P50 стадии,
    делённое на фактический знаменатель. Это и есть риск-коррекция — она снимает
    влияние состава пациенток. Без конверта берётся базовая доля опорного окна, и
    факт подмены записывается: карта на базовой доле измеряет «не как у нас обычно»,
    а не «хуже, чем ожидалось для ЭТИХ пациенток».
    """
    if not risk_adjustment_enabled:
        # ГЕЙТ ЗАКРЫТ по умолчанию, и это измеренное решение, а не осторожность.
        # Прямая подстановка p_i = P50/знаменатель УХУДШИЛА разнородность вместо
        # того, чтобы её снять: X2/df вырос 2.33 -> 11.92 (MIIR) и 2.76 -> 14.48
        # (BlR) на опорном окне. Причина не в качестве прогноза — P50 хорошо
        # согласован с фактом (корреляция 0.86-0.97, смещение <1 эмбриона), — а в
        # том, что ТОЧЕЧНАЯ оценка используется как ИЗВЕСТНАЯ вероятность. У `mii`
        # 49% циклов дают p_i >= 0.95 (модель ограничена сверху числом ОКК), и там
        # биномиальная дисперсия n·p·(1-p) вырождается: один промах даёт вклад 124
        # при медианном 5. Ширина P10-P90 существует ровно для того, чтобы выразить
        # эту неопределённость, и выбрасывать её нельзя.
        # См. docs/METHODOLOGY.md §5.8.
        return float(np.clip(baseline_rate, 0.01, 0.99)), "baseline"

    stage = ENVELOPE_STAGE.get(metric)
    if envelope and stage:
        p50 = envelope.get(f"{stage}_p50")
        if p50 is not None and not pd.isna(p50) and denominator > 0:
            mode = envelope.get(f"{stage}_p50__mode") or "prospective"
            return (float(np.clip(p50 / denominator, 0.01, 0.99)),
                    f"envelope:{mode}")
    return float(np.clip(baseline_rate, 0.01, 0.99)), "baseline"


def evaluate_cycle(cfg: Config, clinic: Clinic, row: pd.Series,
                   baselines: dict[str, dict], thresholds: dict,
                   states: dict[str, ChartState],
                   envelope: dict | None = None,
                   miri_status: str = "unavailable") -> CycleEvaluation:
    """Оценить один закрытый цикл и продвинуть карты."""
    evaluation = CycleEvaluation(cycle_uid=row["cycle_uid"],
                                 closure_date=row.get("closure_due"),
                                 miri_status=miri_status)
    cusum_cfg = cfg.settings["charts"]["cusum"]
    h_watch, h_alert = thresholds.get("h_watch", {}), thresholds.get("h_alert", {})

    # 1. Абсолютный контур: все метрики, у которых есть наблюдение.
    for metric in METRIC_PARTS:
        counts = metric_counts(row, metric)
        base = baselines.get(metric)
        if counts is None:
            continue
        numerator, denominator = counts
        entry = {"numerator": numerator, "denominator": denominator,
                 "rate": numerator / denominator}
        if base:
            entry.update({"baseline_rate": base.get("rate"),
                          "competency": base.get("competency"),
                          "benchmark": base.get("benchmark"),
                          "direction": base.get("direction")})
        evaluation.metrics[metric] = entry

    # 2. Относительный контур: карты.
    crossed_watch: list[str] = []
    for metric in clinic.cusum_charts:
        base = baselines.get(metric)
        counts = metric_counts(row, metric)
        state = states.setdefault(metric, ChartState(metric=metric))
        if base is None or counts is None or not np.isfinite(base.get("rate", np.nan)):
            continue
        numerator, denominator = counts
        odds_ratio = (cusum_cfg["odds_ratio_deterioration"]
                      if base["direction"] == "lower"
                      else cusum_cfg["odds_ratio_upper_control"])
        expected, source = _expected_probability(
            row, metric, denominator, base["rate"], envelope,
            risk_adjustment_enabled=bool(
                cfg.settings.get("envelope", {}).get("risk_adjustment_enabled", False)))

        # Одно обновление на цикл: вклад всех его эмбрионов суммой. Мощность
        # по-прежнему даёт число эмбрионов (оно в весе), но двигать карту внутри
        # цикла нельзя — порядок эмбрионов не наблюдаем.
        threshold_alert, threshold_watch = h_alert.get(metric), h_watch.get(metric)
        runner = charts.CusumRunner(
            statistic=state.statistic, above_watch=state.above_watch,
            observations=state.observations, last_zero_index=state.last_zero_index,
            watch_episodes=state.watch_episodes, alert_signals=state.signals)
        event = runner.step_block(numerator, denominator, expected, odds_ratio,
                                  threshold_watch, threshold_alert)
        state.statistic = runner.statistic
        state.above_watch = runner.above_watch
        state.observations = runner.observations
        state.last_zero_index = runner.last_zero_index
        state.watch_episodes = runner.watch_episodes
        state.signals = runner.alert_signals
        crossed = LEVEL_ALERT if event == "alert" else LEVEL_WATCH if event == "watch" else None

        # В пик и траекторию идёт ДОСТИГНУТОЕ значение, а не остаток после
        # обнуления. Иначе экскурсия, ради которой карта и существует, не попадает
        # никуда: `statistic` после сигнала alert равна нулю, и отчёт печатал
        # «сигналов 1, пик 6.25 при пороге 7.44» — пик ниже порога, который был
        # пересечён. На шагах без сигнала attained и statistic совпадают.
        state.record(row.get("closure_due"), runner.attained)
        evaluation.charts[metric] = {
            "statistic": round(runner.attained, 3),
            # Остаток после обнуления. Нужен отдельно: карточка тревоги обязана
            # нести величину, НА КОТОРОЙ сработала карта, а панель — то, с чего
            # карта продолжит копить завтра.
            "statistic_after": round(state.statistic, 3),
            "peak": round(state.peak, 3),
            "observations": state.observations,
            "expected_source": source,
            "expected": round(expected, 3),
            "crossed": crossed,
            "above_watch": state.above_watch,
            "onset_observation": state.last_zero_index,
            "contour": _contour_of(cfg, metric),
        }
        if crossed == LEVEL_ALERT:
            evaluation.level = LEVEL_ALERT
            crossed_watch.append(metric)
        elif crossed == LEVEL_WATCH:
            crossed_watch.append(metric)
            if evaluation.level == LEVEL_OK:
                evaluation.level = LEVEL_WATCH

    # Контур — по всем картам, которые сработали В ЭТОМ цикле, а не по первой
    # попавшейся: одно событие может задеть два контура, и подменять их одним
    # значит терять именно ту информацию, ради которой контуры разведены.
    triggered_contours = sorted({info["contour"] for metric, info in evaluation.charts.items()
                                 if info.get("crossed") and info.get("contour")})
    evaluation.contour = "+".join(triggered_contours) if triggered_contours else None

    # 3. Подтверждение. Ранг определяет, можно ли повысить watch до alert.
    if evaluation.level != LEVEL_OK:
        evaluation.corroboration, evaluation.corroboration_rank = _corroborate(
            cfg, evaluation, crossed_watch)

    # 4. Быстрый маршрут: watch + сильное независимое подтверждение -> alert.
    #    Здесь он почти никогда не срабатывает — сильное подтверждение приходит
    #    позже, отдельным шагом (см. escalate_with_context). Вызов оставлен, чтобы
    #    маршрут был ОДИН: появись сильный источник внутри _corroborate, он не
    #    потерялся бы молча.
    escalate_with_context(cfg, evaluation)

    # 5. Гейт. Уровень alert недоступен, пока окружение черновое или пороги
    #    не откалиброваны в этом же окружении.
    block = cfg.alert_block_reason()
    if evaluation.level == LEVEL_ALERT and block:
        evaluation.level = LEVEL_WATCH
        evaluation.notes.append(f"alert понижен до watch: {block}")

    return evaluation


def escalate_with_context(cfg: Config, evaluation: CycleEvaluation, *,
                          coverage_blocks: bool = False) -> bool:
    """Повысить `watch` до `alert` по СИЛЬНОМУ независимому подтверждению.

    Отдельный шаг, а не часть `evaluate_cycle`, потому что сильное подтверждение
    физически приходит позже самой оценки: обстановка окна (среда MIRI, смены лота)
    собирается только для СРАБОТАВШИХ циклов — то есть уже зная, что цикл сработал.
    Внутри `_corroborate` ранга `strong` не бывает и быть не может: вторая карта той
    же метрики подтверждением не является, а другая метрика того же контура даёт
    только `moderate`.

    Три условия, каждое обязательное и каждое проверяется здесь, а не у вызывающего:

    1. **Ранг strong.** `moderate` до `alert` не поднимает — это неподвижное правило
       (`docs/METHODOLOGY.md` §6).
    2. **Покрытие конверта не просело.** Падение покрытия означает, что под вопросом
       сама опора — состав пациенток или дрейф модели (§4). Пока это не разобрано,
       эскалация запрещена, а `watch` остаётся.
    3. **Гейт alert открыт.** Боевое окружение, выполненная калибровка, совпадение
       окружения калибровки с текущим — `config.alert_block_reason()`.

    Возвращает True, если уровень действительно поднялся. Идемпотентна: повторный
    вызов на уже поднятой тревоге ничего не делает.
    """
    if (evaluation.level != LEVEL_WATCH
            or evaluation.corroboration_rank != RANK_STRONG):
        return False
    if coverage_blocks:
        evaluation.notes.append(
            "эскалация до alert заблокирована: покрытие конверта значимо ниже "
            "опорного — сначала разбирается состав пациенток и дрейф модели (§4)")
        return False
    block = cfg.alert_block_reason()
    if block:
        evaluation.notes.append(f"alert по сильному подтверждению недоступен: {block}")
        return False
    evaluation.level = LEVEL_ALERT
    evaluation.notes.append(
        "повышено до alert по быстрому маршруту: сработала карта и есть "
        "сильное независимое подтверждение")
    return True


# Колонки карточки, которые отложенная доводка имеет право переписать. Список
# закрытый и короткий намеренно: всё остальное в записанной карточке — то, что
# система думала В МОМЕНТ ОЦЕНКИ, и переписывать это значило бы подделывать
# собственную историю.
UPGRADABLE = ("level", "corroboration", "corroboration_rank", "miri_status",
              "upgraded_at", "notes")


def deferred_upgrades(cfg: Config, clinic: Clinic, frame: pd.DataFrame,
                      alerts: pd.DataFrame, *,
                      coverage_blocks: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Второй такт по среде: доводка карточек, выписанных до появления логов.

    Зачем это существует. Обстановка окна физически приходит ПОЗЖЕ оценки: цикл
    закрывается на T+6, а выгрузка логов инкубатора приезжает недельной нарезкой и
    его окно ещё не покрывает. Карточка выписывается с `miri_status=pending`, цикл
    попадает в `already_evaluated` и больше не пересматривается никогда. Следствие
    измерено на этой площадке: маршрут `watch -> alert` по сильному подтверждению
    не сработал ни разу за всю историю (`escalated_by_context: 0`), хотя система
    ровно ради него и строилась. Обещание «обогащение может повысить watch → alert»
    было невыполнимым по устройству, а не по обстоятельствам.

    Что здесь разрешено — и это весь список:

    * поднять `watch` до `alert`, если обстановка окна оказалась необычной на фоне
      когорты и все три гейта эскалации открыты;
    * записать ставший известным `miri_status` — это факт, а не решение.

    Что запрещено и чего здесь нет: создавать тревогу (карточки берутся только из
    уже записанных), снижать уровень, трогать состояние карт (карта секвенциальна,
    второй проход по циклу сдвинул бы её дважды) и переписывать что-либо кроме
    `UPGRADABLE`. Доводка идёт только в инкрементальном прогоне: `--replay`
    пересобирает журнал с нуля, уже зная всю обстановку.

    Возвращает (журнал, список произведённых повышений).
    """
    from . import confounders  # noqa: PLC0415

    if alerts.empty or "miri_status" not in alerts:
        return alerts, []
    waiting = alerts.loc[(alerts["level"] == LEVEL_WATCH)
                         & (alerts["miri_status"] == confounders.STATUS_PENDING)]
    if waiting.empty:
        return alerts, []

    thresholds_env, _ = confounders.load_baseline(cfg, clinic)
    subset = frame.loc[frame["cycle_uid"].isin(set(waiting["cycle_uid"]))]
    if subset.empty:
        return alerts, []
    contexts = confounders.collect(cfg, clinic, subset)

    alerts = alerts.copy()
    for column in ("upgraded_at", "notes"):
        if column not in alerts:
            alerts[column] = ""
    block = cfg.alert_block_reason()
    stamp = datetime.now().isoformat(timespec="seconds")
    upgrades: list[dict] = []

    for index, row in waiting.iterrows():
        context = contexts.get(row["cycle_uid"])
        if context is None or context.miri_status == confounders.STATUS_PENDING:
            continue    # логи всё ещё не покрыли окно — ждём следующей выгрузки
        # Статус пишется в любом случае, в том числе `unavailable`. Иначе карточка
        # с окном, которое выгрузки уже не догонят, осталась бы в `pending` навсегда
        # и проверялась бы на каждом прогоне до конца времён.
        alerts.at[index, "miri_status"] = context.miri_status
        if (context.miri_status != confounders.STATUS_ENRICHED
                or not thresholds_env or not context.is_unusual(thresholds_env)):
            continue

        detail = (f"критических минут {context.critical_minutes:.0f}, "
                  f"смен лота {context.lot_changes}")
        try:
            corroboration = json.loads(row.get("corroboration") or "[]")
        except (TypeError, ValueError):
            corroboration = []
        corroboration.append({"type": "unusual_window", "detail": detail,
                              "found_at": stamp})
        alerts.at[index, "corroboration"] = json.dumps(corroboration,
                                                       ensure_ascii=False)
        alerts.at[index, "corroboration_rank"] = RANK_STRONG
        alerts.at[index, "upgraded_at"] = stamp

        reason = ("эскалация до alert заблокирована: покрытие проспективного "
                  "конверта значимо ниже опорного (§4)" if coverage_blocks
                  else f"alert по сильному подтверждению недоступен: {block}"
                  if block else "")
        previous = str(row.get("notes") or "")
        if reason:
            note = f"обстановка окна разобрана {stamp[:10]}: {detail}; {reason}"
        else:
            alerts.at[index, "level"] = LEVEL_ALERT
            note = (f"повышено до alert отложенной доводкой {stamp[:10]}: карта "
                    f"сработала, обстановка окна необычна ({detail})")
            upgrades.append({"cycle_uid": row["cycle_uid"], "detail": detail,
                             "raised_at": row.get("raised_at")})
        alerts.at[index, "notes"] = f"{previous} | {note}".strip(" |")

    return alerts, upgrades


def _corroborate(cfg: Config, evaluation: CycleEvaluation,
                 crossed: list[str]) -> tuple[list, str]:
    """Источники подтверждения и их ранг.

    Ключевое ограничение: CUSUM и EWMA одной метрики питаются одним рядом и
    подтверждением друг друга не являются. Сюда они не попадают вовсе.

    Второе ограничение, не менее важное: одновременное срабатывание во ВСЕХ
    контурах — не тройное подтверждение проблемы культивирования, а признак
    общего фактора (состав пациенток, дрейф модели, общелабораторное событие).
    Оно понижает, а не повышает уверенность в локализации — см. §4 методологии.
    """
    sources: list[dict] = []
    rank = RANK_NONE
    contours_cfg = cfg.settings["localisation"]["contours"]

    by_contour: dict[str, list[str]] = {}
    for metric in sorted(set(crossed)):
        contour = _contour_of(cfg, metric)
        if contour:
            by_contour.setdefault(contour, []).append(metric)

    # Умеренное подтверждение: другая метрика ТОГО ЖЕ контура.
    for contour, metrics in by_contour.items():
        if len(metrics) > 1:
            sources.append({"type": "other_metric_same_contour", "contour": contour,
                            "detail": ", ".join(metrics)})
            rank = RANK_MODERATE

    # Сработало во всех контурах сразу — общий фактор, а не локализованный.
    if len(by_contour) >= len(contours_cfg):
        sources.append({"type": "all_contours_at_once",
                        "detail": "сработали все контуры — разбирать по §4 "
                                  "(состав пациенток / дрейф модели / общий фактор), "
                                  "а не как локальную проблему культивирования"})
        rank = RANK_NONE
        evaluation.notes.append(
            "сработали все контуры сразу: локализация недостоверна, эскалация не "
            "производится до разбора по §4 методологии")

    return sources, rank


def attach_verdict(evaluation: CycleEvaluation, baselines: dict[str, dict],
                   literature: pd.DataFrame | None) -> None:
    """Дописать вердикт по трём контурам. Меняет смысл тревоги, а не её факт."""
    triggered = {m for m, info in evaluation.charts.items() if info.get("crossed")}
    if not triggered:
        return
    states = verdict_mod.build_states(
        {m: evaluation.metrics[m] for m in triggered if m in evaluation.metrics},
        triggered, baselines, literature)
    evaluation.verdicts = [verdict_mod.read(state) for state in states]
    evaluation.verdict_summary = verdict_mod.summarise(evaluation.verdicts)
    priority = (verdict_mod.ACTION_REVIEW_PROTOCOL, verdict_mod.ACTION_REVIEW_CASEMIX,
                verdict_mod.ACTION_STANDING_GAP, verdict_mod.ACTION_TREND)
    for action in priority:
        if any(v.action == action for v in evaluation.verdicts):
            evaluation.action = action
            break


def apply_evaluations(cfg: Config, clinic: Clinic, frame: pd.DataFrame,
                      baselines: dict[str, dict], thresholds: dict,
                      states: dict[str, ChartState],
                      envelopes: dict[str, dict] | None = None,
                      literature: pd.DataFrame | None = None,
                      already_evaluated: set[str] | None = None,
                      escalation_block: str | None = None) -> list[CycleEvaluation]:
    """Прогнать закрытые циклы в хронологическом порядке.

    Порядок обязателен: карта секвенциальна, и перестановка циклов изменила бы её
    траекторию. Сортировка идёт по дате закрытия, а не по дате пункции.

    `escalation_block` — причина, по которой alert в этом прогоне недоступен помимо
    гейта окружения (просело покрытие конверта, §4 методологии). Карты обновляются
    как обычно; меняется только уровень: alert остаётся watch, причина пишется.
    """
    if "checkpoint" not in frame.columns:
        return []
    closed = frame.loc[frame["checkpoint"] == "closure"].copy()
    if closed.empty:
        # Ранний выход, а не «пустая сортировка»: кадр без единого цикла приходит
        # без колонки closure_due вовсе (её дописывает load_from_teable только
        # непустому кадру), и сортировка по ней упала бы с KeyError на первом же
        # прогоне новой клиники. Нечего оценивать — это результат, а не сбой.
        return []
    if already_evaluated:
        # Цикл оценивается ОДИН раз. Повторная подача сдвинула бы карту повторно
        # тем же свидетельством, а тревога появилась бы заново с новой датой.
        closed = closed.loc[~closed["cycle_uid"].isin(already_evaluated)]
    closed = closed.sort_values(["closure_due", "cycle_uid"], kind="stable")
    envelopes = envelopes or {}
    out = []
    for _, row in closed.iterrows():
        evaluation = evaluate_cycle(cfg, clinic, row, baselines, thresholds, states,
                                    envelope=envelopes.get(row["cycle_uid"]))
        if evaluation.level == LEVEL_ALERT and escalation_block:
            evaluation.level = LEVEL_WATCH
            evaluation.notes.append(f"alert понижен до watch: {escalation_block}")
        attach_verdict(evaluation, baselines, literature)
        out.append(evaluation)
    return out


def baselines_from_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {entry["metric"]: entry for entry in manifest.get("metrics", [])}
