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

        state.record(row.get("closure_due"), runner.statistic)
        evaluation.charts[metric] = {
            "statistic": round(state.statistic, 3),
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
    if (evaluation.level == LEVEL_WATCH
            and evaluation.corroboration_rank == RANK_STRONG):
        evaluation.level = LEVEL_ALERT
        evaluation.notes.append(
            "повышено до alert по быстрому маршруту: сработала карта и есть "
            "сильное независимое подтверждение")

    # 5. Гейт. Уровень alert недоступен, пока окружение черновое или пороги
    #    не откалиброваны в этом же окружении.
    block = cfg.alert_block_reason()
    if evaluation.level == LEVEL_ALERT and block:
        evaluation.level = LEVEL_WATCH
        evaluation.notes.append(f"alert понижен до watch: {block}")

    return evaluation


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
                      already_evaluated: set[str] | None = None) -> list[CycleEvaluation]:
    """Прогнать закрытые циклы в хронологическом порядке.

    Порядок обязателен: карта секвенциальна, и перестановка циклов изменила бы её
    траекторию. Сортировка идёт по дате закрытия, а не по дате пункции.
    """
    closed = frame.loc[frame["checkpoint"] == "closure"].copy()
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
        attach_verdict(evaluation, baselines, literature)
        out.append(evaluation)
    return out


def baselines_from_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {entry["metric"]: entry for entry in manifest.get("metrics", [])}
