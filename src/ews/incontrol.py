"""In-control параметры опорного окна — центр относительного контура EWS.

Опорное окно объявляет владелец (`settings.reference_window`). Задача этого модуля —
не оспорить решение, а зафиксировать его последствия: если метрика внутри окна
значимо отличается от того, что было до него, окно вбирает уже случившийся сдвиг в
норму, и карта на этой метрике перестаёт его видеть. Такая оговорка записывается в
`baseline_caveats` и переносится дальше в манифест калибровки — чтобы «в контроле»
никогда не читалось как «в порядке».
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MetricBaseline:
    metric: str
    numerator: float
    denominator: float
    rate: float
    ci_low: float
    ci_high: float
    direction: str
    competency: float | None
    benchmark: float | None
    status: str
    # Сравнение с периодом ДО опорного окна.
    prior_rate: float | None
    delta_pp: float | None
    p_value: float | None
    shift_absorbed: bool


def _wilson(successes: float, total: float, z: float = 1.959963985) -> tuple[float, float]:
    """95% интервал Уилсона. NaN при numerator > denominator — интервал не определён."""
    if not total or successes > total or successes < 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _fisher(prior_num: float, prior_den: float,
            window_num: float, window_den: float) -> float | None:
    """Точный тест Фишера «окно против того, что было до него»."""
    values = [prior_num, prior_den, window_num, window_den]
    if any(v is None or pd.isna(v) for v in values):
        return None
    table = [[int(prior_num), int(prior_den - prior_num)],
             [int(window_num), int(window_den - window_num)]]
    if min(table[0][1], table[1][1]) < 0 or min(table[0][0], table[1][0]) < 0:
        # numerator > denominator — штатная ситуация для ClR/MIIR (см. правила
        # проекта), но тест на ней не определён.
        return None
    return float(stats.fisher_exact(table).pvalue)


def build_baselines(window_kpi: pd.DataFrame, prior_kpi: pd.DataFrame | None,
                    alpha: float = 0.05) -> list[MetricBaseline]:
    """Свести KPI опорного окна и предшествующего периода в базовые параметры."""
    prior_lookup: dict[str, pd.Series] = {}
    if prior_kpi is not None and not prior_kpi.empty:
        prior_lookup = {row["metric"]: row for _, row in prior_kpi.iterrows()}

    out: list[MetricBaseline] = []
    for _, row in window_kpi.iterrows():
        metric = row["metric"]
        numerator, denominator = row.get("numerator"), row.get("denominator")
        rate = row.get("rate")
        ci_low, ci_high = _wilson(numerator, denominator)

        prior = prior_lookup.get(metric)
        prior_rate = float(prior["rate"]) if prior is not None and pd.notna(prior["rate"]) else None
        p_value = (_fisher(prior["numerator"], prior["denominator"], numerator, denominator)
                   if prior is not None else None)
        delta_pp = ((float(rate) - prior_rate) * 100.0
                    if prior_rate is not None and pd.notna(rate) else None)

        # Сдвиг «поглощён» опорным окном, если он значим И направлен в сторону
        # ухудшения по направлению контроля этой метрики.
        worsened = False
        if delta_pp is not None:
            worsened = delta_pp < 0 if row.get("direction") == "lower" else delta_pp > 0
        shift_absorbed = bool(p_value is not None and p_value < alpha and worsened)

        out.append(MetricBaseline(
            metric=metric,
            numerator=float(numerator) if pd.notna(numerator) else float("nan"),
            denominator=float(denominator) if pd.notna(denominator) else float("nan"),
            rate=float(rate) if pd.notna(rate) else float("nan"),
            ci_low=ci_low, ci_high=ci_high,
            direction=row.get("direction"),
            competency=row.get("competency"), benchmark=row.get("benchmark"),
            status=row.get("status"),
            prior_rate=prior_rate, delta_pp=delta_pp, p_value=p_value,
            shift_absorbed=shift_absorbed,
        ))
    return out


def caveats(baselines: list[MetricBaseline]) -> list[str]:
    """Человекочитаемые оговорки, которые обязаны дожить до манифеста калибровки."""
    notes: list[str] = []
    for base in baselines:
        if base.shift_absorbed:
            notes.append(
                f"{base.metric}: опорное окно вобрало уже случившийся сдвиг "
                f"{base.prior_rate:.3f} → {base.rate:.3f} ({base.delta_pp:+.1f} пп, "
                f"p={base.p_value:.3f}). Карта на этой метрике центрирована на новом "
                f"уровне и прошлый сдвиг не увидит — она сработает только на дальнейшем "
                f"ухудшении. Абсолютный контур Венского консенсуса продолжает показывать "
                f"уровень честно (status={base.status})."
            )
        # Оговорка про абсолютный уровень выдаётся только когда benchmark лежит ВНЕ
        # доверительного интервала. Иначе она срабатывала бы на каждой метрике,
        # чья точечная оценка на волосок ниже порога (BlR 0.590 при CI 0.534-0.643
        # не является «ниже 0.60» ни в каком полезном смысле), и обесценила бы
        # весь список.
        if base.benchmark is None or np.isnan(base.ci_low) or np.isnan(base.ci_high):
            continue
        below = base.direction == "lower" and base.ci_high < base.benchmark
        above = base.direction == "upper" and base.ci_low > base.benchmark
        if below or above:
            notes.append(
                f"{base.metric}: {base.rate:.3f} (95% CI {base.ci_low:.3f}–{base.ci_high:.3f}) — "
                f"benchmark {base.benchmark} лежит вне интервала. Принятие этого уровня за "
                f"норму относительного контура не делает его нормой абсолютного."
            )
    return notes


def as_manifest(baselines: list[MetricBaseline], window_start: str,
                window_end: str | None, n_cycles: int, n_embryos: float,
                declared_by: str) -> dict:
    return {
        "reference_window": {
            "start": window_start,
            "end": window_end or date.today().isoformat(),
            "declared_by": declared_by,
            "n_cycles": n_cycles,
            "n_embryos_2pn": float(n_embryos),
        },
        "metrics": [asdict(b) for b in baselines],
        "baseline_caveats": caveats(baselines),
    }
