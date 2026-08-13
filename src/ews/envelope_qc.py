"""QA-слой конверта: факт против P10–P90.

То единственное, для чего конверт пригоден на этих данных, и то, ради чего его стоит
регистрировать. Разделение принципиальное:

* **доля** — модель не предсказывает (corr 0.03…0.16 на наших циклах), поэтому
  риск-коррекция карт закрыта гейтом;
* **счёт** — предсказывает хорошо (corr 0.86–0.97), поэтому интервал P10–P90 вокруг
  счёта является осмысленной опорой.

Отсюда роль слоя: не тревога и не вход в карту, а независимый вопрос «попал ли факт
этого цикла в то, что для него предсказывали». Ответ нужен двум читателям —
`ews-model-auditor` (не поехала ли модель) и разбору тревоги (был ли цикл сам по себе
необычным).

Покрытие сравнивается с **измеренным** опорным, а не с номинальными 0.80: у стадий оно
разное (0.759 у 2PN, 0.823 у бластоцист), и сравнение с номиналом давало бы постоянный
ложный сигнал на одной стадии и слепоту на другой.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Стадия конверта -> колонка факта в каноническом кадре.
STAGE_ACTUAL = {
    "mii": "mii",
    "pn2": "two_pn",
    "blasts_early": "bl_total",
    "blasts_after_fert": "bl_total",
}

# Ключ стадии в settings.envelope.baseline_coverage.
STAGE_BASELINE_KEY = {
    "mii": "mii", "pn2": "pn2",
    "blasts_early": "blasts", "blasts_after_fert": "blasts",
}

DEGENERATE = "envelope_degenerate"


def build(frame: pd.DataFrame, envelopes: dict[str, dict],
          settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(по циклам, по стадиям). Пустые кадры, если конвертов нет."""
    if not envelopes:
        return pd.DataFrame(), pd.DataFrame()

    envelope_cfg = settings.get("envelope") or {}
    min_width = float(envelope_cfg.get("min_interval_width", 1.0))
    divisor = float(envelope_cfg.get("iqr_to_sigma_divisor", 2.5631))
    baseline = envelope_cfg.get("baseline_coverage") or {}

    rows: list[dict] = []
    for _, cycle in frame.iterrows():
        entry = envelopes.get(cycle.get("cycle_uid"))
        if not entry:
            continue
        for stage, actual_column in STAGE_ACTUAL.items():
            p10, p50, p90 = (entry.get(f"{stage}_p10"), entry.get(f"{stage}_p50"),
                             entry.get(f"{stage}_p90"))
            actual = pd.to_numeric(pd.Series([cycle.get(actual_column)]),
                                   errors="coerce").iloc[0]
            if p10 is None or p90 is None or pd.isna(actual):
                continue

            width = float(p90) - float(p10)
            record = {
                "cycle_uid": cycle["cycle_uid"], "stage": stage,
                "actual": float(actual), "p10": float(p10),
                "p50": float(p50) if p50 is not None else np.nan,
                "p90": float(p90), "width": round(width, 2),
                "mode": entry.get(f"{stage}_p50__mode"),
            }
            if width <= min_width:
                # Полоса шириной <= 1 на счётной величине не может дать 80%
                # покрытия: дискретность счёта превышает разрешение интервала.
                # Это ограничение интервала, а не сигнал лаборатории.
                record.update({"status": DEGENERATE, "in80": None, "z": np.nan})
            else:
                inside = float(p10) <= float(actual) <= float(p90)
                record.update({
                    "status": "in80" if inside else "outside",
                    "in80": bool(inside),
                    "z": round((float(actual) - float(p50)) / (width / divisor), 2)
                    if p50 is not None else np.nan,
                })
            rows.append(record)

    per_cycle = pd.DataFrame(rows)
    if per_cycle.empty:
        return per_cycle, pd.DataFrame()

    usable = per_cycle.loc[per_cycle["status"] != DEGENERATE]
    summary = []
    for stage, group in usable.groupby("stage"):
        observed = float(group["in80"].mean())
        expected = baseline.get(STAGE_BASELINE_KEY.get(stage, ""), None)
        n = len(group)
        # Односторонний тест: проверяется только ПАДЕНИЕ покрытия. Рост означает,
        # что интервалы шире нужного — это потеря чувствительности, а не сбой.
        p_value = (float(stats.binomtest(int(group["in80"].sum()), n,
                                         float(expected), alternative="less").pvalue)
                   if expected and n >= 10 else np.nan)
        summary.append({
            "stage": stage, "n": n,
            "coverage": round(observed, 3),
            "baseline": expected,
            "delta_pp": round((observed - expected) * 100, 1) if expected else None,
            "p_below": round(p_value, 4) if not pd.isna(p_value) else None,
            "mae_p50": round(float((group["actual"] - group["p50"]).abs().mean()), 2),
            "degenerate_share": round(
                float((per_cycle.loc[per_cycle["stage"] == stage, "status"]
                       == DEGENERATE).mean()), 3),
        })
    return per_cycle, pd.DataFrame(summary)


def coverage_alarm(summary: pd.DataFrame, alpha: float = 0.01) -> tuple[bool, str]:
    """Просело ли покрытие значимо — и потому блокируется ли эскалация.

    Падение покрытия сразу по нескольким стадиям — не сигнал культивирования, а
    повод разобрать модель и состав пациенток (docs/METHODOLOGY.md §4). Пока разбор
    не сделан, тревоги остаются на уровне watch.
    """
    if summary.empty or "p_below" not in summary:
        return False, ""
    dropped = summary.loc[summary["p_below"].notna()
                          & (summary["p_below"] < alpha)]
    if dropped.empty:
        return False, ""
    stages = ", ".join(dropped["stage"])
    return True, (f"покрытие конверта значимо ниже опорного на стадиях: {stages}. "
                  f"Это не сигнал культивирования — сначала разбирается состав "
                  f"пациенток и дрейф модели (§4), эскалация до alert заблокирована")
