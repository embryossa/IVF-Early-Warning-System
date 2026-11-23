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

# Стадии после оплодотворения. Цикл без инсеминации (инсеминировано = 0: Cryo, NO)
# на них не оценивается: 2PN и бластоцист там нет не потому, что модель промахнулась,
# а потому что оплодотворения не было. Иначе каждый такой цикл — гарантированный
# промах ниже P10 (то же правило, что у знаменателя FR, cycles.fr_denominator).
POST_FERTILISATION = {"pn2", "blasts_early", "blasts_after_fert"}

# Ключ стадии в settings.envelope.baseline_coverage.
STAGE_BASELINE_KEY = {
    "mii": "mii", "pn2": "pn2",
    "blasts_early": "blasts", "blasts_after_fert": "blasts",
}

DEGENERATE = "envelope_degenerate"

# Режим регистрации конверта. Ранним предупреждением является только первый:
# ретроспективный добор писался моделью, которая обучалась в том числе на этих
# циклах, и его покрытие — свойство добора, а не прогноза.
MODE_PROSPECTIVE = "prospective"

# Минимум наблюдений для теста покрытия. Ниже него биномиальный тест не отличает
# просадку от разброса, и «нет решения» честнее любого решения.
MIN_N_FOR_TEST = 10


def _p_below(successes: int, n: int, expected) -> float:
    """p-value одностороннего теста «покрытие ниже опорного».

    Односторонний намеренно: проверяется только ПАДЕНИЕ покрытия. Рост означает,
    что интервалы шире нужного — это потеря чувствительности, а не сбой.
    """
    if not expected or n < MIN_N_FOR_TEST:
        return np.nan
    return float(stats.binomtest(successes, n, float(expected),
                                 alternative="less").pvalue)


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
        not_inseminated = pd.to_numeric(pd.Series([cycle.get("inseminated")]),
                                        errors="coerce").iloc[0] == 0
        for stage, actual_column in STAGE_ACTUAL.items():
            if stage in POST_FERTILISATION and not_inseminated:
                continue
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
        p_value = _p_below(int(group["in80"].sum()), n, expected)

        # Проспективное подмножество считается ОТДЕЛЬНО, и именно оно решает,
        # блокировать ли эскалацию (решение владельца, 2026-09-10). Пулированное
        # число смешивает две популяции с разным происхождением: измерено на этой
        # когорте — blasts_after_fert даёт 0.699 на ретроспективном доборе (n=73)
        # против 0.833 на проспективных конвертах (n=6), и в пуле выходит 0.709.
        # Запирать alert числом, которое тянет вниз добор, — значит блокировать
        # тревогу данными, которые по определению проекта ранним предупреждением
        # не являются. Пулированное остаётся в отчёте: оно нужно аудиту модели.
        forward = group.loc[group["mode"] == MODE_PROSPECTIVE]
        n_forward = len(forward)
        observed_forward = float(forward["in80"].mean()) if n_forward else np.nan
        p_forward = _p_below(int(forward["in80"].sum()), n_forward, expected)

        summary.append({
            "stage": stage, "n": n,
            "coverage": round(observed, 3),
            "baseline": expected,
            "delta_pp": round((observed - expected) * 100, 1) if expected else None,
            "p_below": round(p_value, 4) if not pd.isna(p_value) else None,
            "n_prospective": n_forward,
            "coverage_prospective": (round(observed_forward, 3)
                                     if n_forward else None),
            "p_below_prospective": (round(p_forward, 4)
                                    if not pd.isna(p_forward) else None),
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

    Судит ТОЛЬКО по проспективным конвертам: см. комментарий в `build`. Пока их
    меньше `MIN_N_FOR_TEST` на стадию, `p_below_prospective` пуст, и гейт молчит —
    не потому что всё хорошо, а потому что решать не на чем. Это состояние обязано
    быть видно в отчёте, поэтому его называет вторая половина возвращаемого текста.
    """
    if summary.empty or "p_below_prospective" not in summary:
        return False, ""
    dropped = summary.loc[summary["p_below_prospective"].notna()
                          & (summary["p_below_prospective"] < alpha)]
    if dropped.empty:
        return False, _coverage_silence_note(summary)
    stages = ", ".join(dropped["stage"])
    return True, (f"покрытие проспективного конверта значимо ниже опорного на "
                  f"стадиях: {stages}. Это не сигнал культивирования — сначала "
                  f"разбирается состав пациенток и дрейф модели (§4), эскалация "
                  f"до alert заблокирована")


def _coverage_silence_note(summary: pd.DataFrame) -> str:
    """Почему гейт молчит: «проверено и чисто» против «проверять пока нечем».

    Разница не косметическая. Первое — свидетельство в пользу конверта, второе —
    его отсутствие, и читатель отчёта обязан различать их без чтения кода.
    """
    untested = summary.loc[summary["p_below_prospective"].isna(), "stage"]
    if untested.empty:
        return ""
    have = int(summary["n_prospective"].fillna(0).max())
    return (f"покрытие по проспективным конвертам ещё не проверялось на стадиях: "
            f"{', '.join(untested)} — их накоплено до {have} при минимуме "
            f"{MIN_N_FOR_TEST}. Эскалация не блокируется, но и подтверждения "
            f"конверту пока нет")
