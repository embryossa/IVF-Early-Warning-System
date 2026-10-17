"""Медленный контур подтверждения по исходу.

Замыкает конструкцию исходной статьи. У Bormann весь смысл AI-KPI был в том, что он
опережает ongoing pregnancy rate: процесс виден на 3-й день, исход — через 6-8 недель,
и связь между ними (R²=0.9063) оправдывает раннее вмешательство.

У нас обе половины есть, но до сих пор жили порознь:

* **быстрый контур** — доли воронки, оценка на 6-й день (карты, консенсус, литература);
* **медленный контур** — вероятность беременности от Digital Twin (`DIGITAL TWIN`,
  L7 BEFE) против фактического исхода, доступного через недели. PRAI (TabPFN) убран
  решением владельца 2026-10-17: в других клиниках доступен именно DT, и опора должна
  быть одной и той же.

Роль медленного контура — не поднимать тревоги, а **подтверждать или опровергать**
тренды быстрого. Если процесс просел и предсказанный исход поехал следом — тренд
реален. Если процесс просел, а исход держится — просадка в стадии, не влияющей на
результат, и цена вмешательства выше цены наблюдения.

Из Digital Twin здесь нужны только итоговые вероятности, и они уже приходят
в мастер-таблице OPU. Пайплайн DT запускать не требуется.

**Чего этот контур не делает.** Он не поднимает тревог и не участвует в эскалации:
исход зависит от переноса, эндометрия, эмбриона и десятка вещей вне лаборатории.
Приписывать лаборатории изменение частоты беременности значило бы обвинять её в том,
чем она не управляет.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Модельная колонка в мастер-таблице — вероятность беременности на цикл.
DIGITAL_TWIN = "digital_twin"    # L7 BEFE, итоговый ансамбль


@dataclass
class OutcomeWindow:
    """Сопоставление прогноза и факта по окну циклов."""
    label: str
    n_cycles: int
    n_known: int
    predicted_dt: float | None
    observed: float | None
    ci_low: float | None
    ci_high: float | None
    note: str = ""

    def agrees(self, model: float | None) -> bool | None:
        """Попал ли прогноз модели в доверительный интервал факта."""
        if model is None or self.ci_low is None:
            return None
        return bool(self.ci_low <= model <= self.ci_high)


def _wilson(successes: float, total: float, z: float = 1.959963985) -> tuple[float, float]:
    if not total or successes > total or successes < 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(frame: pd.DataFrame, label: str = "когорта") -> OutcomeWindow:
    """Прогноз против факта по набору циклов.

    Знаменатель — только циклы с ИЗВЕСТНЫМ исходом. Считать частоту беременности по
    всем циклам, включая ещё не дошедшие до переноса, значит делить на популяцию,
    которой этот исход ещё не мог случиться.
    """
    outcome = pd.to_numeric(frame.get("transfer_outcome"), errors="coerce")
    known = outcome.isin([0, 1])
    n_known = int(known.sum())

    digital_twin = pd.to_numeric(frame.get(DIGITAL_TWIN), errors="coerce")

    if n_known == 0:
        return OutcomeWindow(label, len(frame), 0, None, None, None, None,
                             note="исход не известен ни у одного цикла окна")

    successes = float(outcome[known].sum())
    observed = successes / n_known
    ci_low, ci_high = _wilson(successes, n_known)
    return OutcomeWindow(
        label=label, n_cycles=len(frame), n_known=n_known,
        predicted_dt=(float(digital_twin[known].mean())
                      if digital_twin[known].notna().any() else None),
        observed=observed, ci_low=ci_low, ci_high=ci_high,
    )


def compare_periods(frame: pd.DataFrame, split: pd.Timestamp,
                    before: str = "до окна", after: str = "опорное окно"
                    ) -> list[OutcomeWindow]:
    """Исход до и после границы опорного окна — тот же разрез, что у процесса."""
    dates = pd.to_datetime(frame.get("date_opu"), errors="coerce")
    return [summarise(frame.loc[dates < split], before),
            summarise(frame.loc[dates >= split], after)]


def confirms_process_trend(windows: list[OutcomeWindow],
                           min_known: int = 20) -> tuple[bool | None, str]:
    """Подтверждает ли исход тренд процесса.

    Возвращает (вердикт, объяснение). ``None`` — данных не хватает, и это самый
    частый честный ответ: исход известен у трети циклов и приходит с задержкой
    в недели. Молчание здесь лучше вывода на пяти наблюдениях.
    """
    usable = [w for w in windows if w.n_known >= min_known]
    if len(usable) < 2:
        have = ", ".join(f"{w.label}: {w.n_known}" for w in windows)
        return None, (f"исходов недостаточно для сравнения (нужно {min_known} на "
                      f"период, есть {have}) — медленный контур ещё не созрел")

    earlier, later = usable[0], usable[-1]
    drop = later.observed - earlier.observed
    overlap = not (later.ci_high < earlier.ci_low or earlier.ci_high < later.ci_low)
    if overlap:
        return False, (f"исход не изменился значимо ({earlier.observed:.0%} -> "
                       f"{later.observed:.0%}, интервалы пересекаются): тренд "
                       f"процесса пока не подтверждается результатом")
    direction = "снизился" if drop < 0 else "вырос"
    return True, (f"исход {direction} значимо ({earlier.observed:.0%} -> "
                  f"{later.observed:.0%}, интервалы не пересекаются): изменение "
                  f"дошло до результата")


def calibration_note(window: OutcomeWindow) -> str:
    """Насколько прогноз DT согласуется с фактом на этой когорте."""
    parts = []
    for name, value in (("DIGITAL TWIN", window.predicted_dt),):
        if value is None:
            continue
        agree = window.agrees(value)
        mark = "в интервале факта" if agree else "вне интервала факта"
        parts.append(f"{name} {value:.0%} — {mark}")
    if not parts:
        return "модельных вероятностей в выгрузке нет"
    return "; ".join(parts)
