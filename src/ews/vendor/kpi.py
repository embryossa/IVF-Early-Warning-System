"""Embryology KPI calculations and cohort-level Vienna controls.

Two concepts intentionally stay separate:

* ``Model_KPIScore`` is the exact five-part score used while training the
  Digital Twin.  ``KPIScore`` is retained only as its model-input alias.
* Laboratory KPI are rates with explicit numerators/denominators, direction
  of control, competency/benchmark limits and Wilson confidence intervals.

Missing observations are never converted to zero.  A recorded zero remains a
valid observation.  As agreed for this all-ICSI laboratory, FR uses MII when a
positive MII count is available and otherwise the number inseminated/injected.
Rows with 2PN equal to zero remain excluded from the KPI/model dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KPISpec:
    numerator: str
    denominator: str
    direction: str
    competency: float | None
    benchmark: float | None
    label: str
    indicator_class: str
    source_note: str


# FR uses the owner's accepted local 75% benchmark for this all-ICSI workflow;
# its source is labelled separately from the Vienna ICSI benchmark. DegR and
# 3pNR are upper-control indicators. Reference indicators are not mislabeled KPIs.
KPI_SPECS: dict[str, KPISpec] = {
    "OCCR": KPISpec("occ", "follicles", "lower", 0.70, 0.85,
                    "Oocyte retrieval rate", "reference indicator",
                    "Vienna survey range lower bounds (competency 70–80%; benchmark 85–100%)"),
    "MIIR": KPISpec("mii", "occ", "lower", 0.75, 0.90,
                    "Oocyte maturity rate", "reference indicator",
                    "Vienna survey medians"),
    "FR": KPISpec("two_pn", "_fr_denominator", "lower", 0.65, 0.75,
                  "Fertilization rate (all-ICSI workflow)", "local KPI",
                  "Competency Vienna ICSI 65%; accepted local unified benchmark 75%"),
    "ClR": KPISpec("cleavage_d3", "two_pn", "lower", 0.95, 0.99,
                   "Cleavage rate", "Vienna KPI", "Vienna consensus"),
    "BlR": KPISpec("bl_total", "two_pn", "lower", 0.40, 0.60,
                   "Blastocyst development rate", "Vienna KPI", "Vienna consensus Day 5"),
    "TGBDR": KPISpec("bl_good", "two_pn", "lower", 0.30, 0.40,
                     "Good-quality blastocyst rate", "Vienna PI", "Vienna consensus Day 5"),
    "DegR": KPISpec("degrade", "_degr_denominator", "upper", 0.10, 0.05,
                    "Oocyte degeneration rate", "Vienna KPI", "Vienna ICSI damage rate"),
    "3pNR": KPISpec("three_pn", "occ", "upper", 0.09, 0.045,
                    "3PN rate", "reference indicator",
                    "Vienna Alpha-survey medians; upper control"),
}

# Backwards-compatible constants used by the HTML renderer.  Upper-control
# limits must not be put into LCL_VALUES, otherwise a low DegR/3pNR is falsely
# reported as an alert.
LCL_VALUES = {name: (spec.benchmark if spec.direction == "lower" else None)
              for name, spec in KPI_SPECS.items()}
UCL_VALUES = {name: (spec.benchmark if spec.direction == "upper" else None)
              for name, spec in KPI_SPECS.items()}

# Embryologist and doctor competency tables must never be mixed: embryologists are
# judged on everything downstream of oocyte retrieval, doctors only on retrieval and
# maturity yield. Kept as ordered tuples so report tables have a stable column order.
EMBRYOLOGIST_METRICS: tuple[str, ...] = ("FR", "ClR", "BlR", "TGBDR", "DegR", "3pNR")
DOCTOR_METRICS: tuple[str, ...] = ("OCCR", "MIIR")


KPI_ADJUSTMENTS = {
    25: {"prob": 0.70, "ci": (0.59, 0.79)}, 24: {"prob": 0.65, "ci": (0.56, 0.74)},
    23: {"prob": 0.60, "ci": (0.51, 0.69)}, 22: {"prob": 0.55, "ci": (0.47, 0.63)},
    21: {"prob": 0.50, "ci": (0.42, 0.57)}, 20: {"prob": 0.45, "ci": (0.38, 0.51)},
    19: {"prob": 0.40, "ci": (0.33, 0.46)}, 18: {"prob": 0.35, "ci": (0.28, 0.41)},
    17: {"prob": 0.30, "ci": (0.24, 0.36)}, 16: {"prob": 0.26, "ci": (0.20, 0.32)},
    15: {"prob": 0.22, "ci": (0.16, 0.30)}, 14: {"prob": 0.18, "ci": (0.13, 0.26)},
    13: {"prob": 0.15, "ci": (0.10, 0.22)}, 12: {"prob": 0.13, "ci": (0.08, 0.20)},
    11: {"prob": 0.10, "ci": (0.06, 0.18)}, 10: {"prob": 0.09, "ci": (0.05, 0.15)},
    9: {"prob": 0.07, "ci": (0.04, 0.14)}, 8: {"prob": 0.06, "ci": (0.03, 0.12)},
    7: {"prob": 0.05, "ci": (0.02, 0.10)}, 6: {"prob": 0.04, "ci": (0.016, 0.09)},
    5: {"prob": 0.03, "ci": (0.012, 0.08)},
}


_NUMERIC = [
    "age", "attempt_no", "follicles", "occ", "mii", "inseminated",
    "two_pn", "cleavage_d3", "embryos_d5", "bl_total", "bl_good",
    "three_pn", "degrade", "frozen", "transferred",
]


def _numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric column, or an all-missing aligned series."""
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide only known observations by a strictly positive denominator.

    This preserves the clinically important distinction between a measured
    zero and an absent value.
    """
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    out = pd.Series(np.nan, index=n.index, dtype=float)
    valid = n.notna() & d.notna() & (d > 0)
    out.loc[valid] = n.loc[valid] / d.loc[valid]
    return out


def compute_rates(df: pd.DataFrame, drop_without_two_pn: bool = True) -> pd.DataFrame:
    """Calculate row rates while keeping the accepted ``2PN != 0`` filter.

    ``drop_without_two_pn=False`` keeps rows that never reached the
    fertilisation check. It exists for the DOCTOR competency table only:
    OCCR=OCC/follicles and MIIR=MII/OCC are computed entirely above the 2PN
    stage, so a retrieval-only cycle ("Только пункция" — oocyte banking, donor
    retrieval) is a complete OCCR/MIIR observation even though 2PN is blank
    there by definition. Dropping it would erase a real retrieval by a real
    doctor from the doctor's own table (owner confirmed 2026-07-31).

    Everything else — embryologist metrics, cohort rates, the model dataset —
    keeps the filter, so the owner's standing ``2PN==0`` exclusion is
    untouched.
    """
    d = df.copy()
    two_pn = _numeric_column(d, "two_pn")
    # A recorded 2PN==0 is dropped either way; only the treatment of a BLANK
    # 2PN differs (pandas ``ne`` already yields True for NaN).
    keep = two_pn.notna() & two_pn.ne(0) if drop_without_two_pn else two_pn.ne(0)
    d = d.loc[keep].copy()
    for col in _NUMERIC:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Optional columns are represented as missing series, never manufactured
    # zeros.  A measured MII=0 stays available for MIIR but cannot be an FR
    # denominator, so FR falls back to the injected/inseminated count.
    mii = _numeric_column(d, "mii")
    inseminated = _numeric_column(d, "inseminated")
    fr_den = mii.where(mii.notna() & mii.gt(0), inseminated)
    d["_fr_denominator"] = fr_den
    d["_degr_denominator"] = fr_den

    d["OCCR"] = _safe_div(_numeric_column(d, "occ"), _numeric_column(d, "follicles"))
    d["MIIR"] = _safe_div(mii, _numeric_column(d, "occ"))
    d["FR"] = _safe_div(_numeric_column(d, "two_pn"), fr_den)
    # Training-time feature is deliberately separate from the laboratory FR.
    # The historical all-ICSI model used injected/inseminated as denominator.
    d["Model_FR"] = _safe_div(_numeric_column(d, "two_pn"), inseminated)
    d["ClR"] = _safe_div(_numeric_column(d, "cleavage_d3"), _numeric_column(d, "two_pn"))
    d["BlR"] = _safe_div(_numeric_column(d, "bl_total"), _numeric_column(d, "two_pn"))
    d["TGBDR"] = _safe_div(_numeric_column(d, "bl_good"), _numeric_column(d, "two_pn"))
    d["3pNR"] = _safe_div(_numeric_column(d, "three_pn"), _numeric_column(d, "occ"))
    d["DegR"] = _safe_div(_numeric_column(d, "degrade"), fr_den)
    return d


def add_russian_feature_columns(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    ru = {
        "age": "Возраст", "attempt_no": "№ попытки",
        "follicles": "Количество фолликулов", "occ": "Число ОКК",
        "mii": "MII", "inseminated": "Число инсеминированных", "two_pn": "2 pN",
        "cleavage_d3": "Число дробящихся на 3 день", "bl_total": "Число Bl",
        "bl_good": "Число Bl хор.кач-ва", "embryos_d5": "Число эмбрионов 5 дня",
        "frozen": "Заморожено эмбрионов", "transferred": "Перенесено эмбрионов",
    }
    for canon, rus in ru.items():
        if canon in d.columns and rus not in d.columns:
            d[rus] = d[canon]
    rates = {
        "ClR": "Частота дробления",
        "BlR": "Частота формирования бластоцист",
        "TGBDR": "Частота формирования бластоцист хорошего качества",
        "OCCR": "Частота получения ОКК",
        "MIIR": "Частота зрелости ОКК",
    }
    for eng, rus_name in rates.items():
        if eng in d.columns:
            d[rus_name] = d[eng]
    if "FR" in d.columns:
        d["Частота оплодотворения Vienna"] = d["FR"]
    if "Model_FR" in d.columns:
        d["Частота оплодотворения"] = d["Model_FR"]
    return d


def _kpi_score_row(row: pd.Series) -> int:
    """Exact training-time five-component Model KPIScore formula."""
    score = 0
    score += 1 if row["Возраст"] >= 40 else (5 if row["Возраст"] <= 36 else 3)
    follicles = row["Количество фолликулов"]
    score += 5 if follicles > 15 else (3 if 8 <= follicles <= 15 else 1)
    inseminated = row["Число инсеминированных"]
    score += 1 if inseminated <= 3 else (3 if 4 <= inseminated <= 7 else 5)
    fr = row["Частота оплодотворения"]
    score += 1 if fr < 0.5 else (3 if fr <= 0.65 else 5)
    good_bl = row["Число Bl хор.кач-ва"]
    score += 1 if good_bl == 0 else (3 if good_bl <= 2 else 5)
    return int(score)


def compute_kpi(df: pd.DataFrame) -> pd.DataFrame:
    d = add_russian_feature_columns(compute_rates(df))
    # This fill policy is deliberately confined to the historical model score
    # so its training formula is reproduced exactly.  Laboratory rates above
    # retain missingness.
    for col in ["Возраст", "Количество фолликулов", "Число инсеминированных",
                "Число Bl хор.кач-ва", "Частота оплодотворения"]:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    if d.empty:
        d["Model_KPIScore"] = pd.Series(index=d.index, dtype="int64")
    else:
        d["Model_KPIScore"] = d.apply(_kpi_score_row, axis=1)
    d["KPIScore"] = d["Model_KPIScore"]  # external model contract only
    return d


def wilson_interval(numerator: float, denominator: float,
                    z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(numerator):
        return np.nan, np.nan
    # numerator > denominator is NOT bad data (owner, 2026-07-29): ClR>100% means the
    # fertilisation check missed the pronuclei window on zygotes that went on to
    # cleave, and MII>OCC means warmed (vitrified) oocytes joined the programme after
    # the fresh retrieval.  Both are real observations and stay in the table — only
    # the binomial CI is undefined here, so it alone is suppressed.  See the
    # vienna-kpi skill; do not relabel these rows as a counting/integrity problem.
    if numerator < 0 or numerator > denominator:
        return np.nan, np.nan
    p = numerator / denominator
    denom = 1.0 + z * z / denominator
    centre = (p + z * z / (2.0 * denominator)) / denom
    half = z * sqrt((p * (1.0 - p) + z * z / (4.0 * denominator)) / denominator) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _control_met(rate: float, limit: float | None, direction: str) -> bool | None:
    if limit is None or pd.isna(rate):
        return None
    return bool(rate >= limit) if direction == "lower" else bool(rate <= limit)


def _aggregate_one(frame: pd.DataFrame, metric: str, spec: KPISpec) -> dict:
    num = _numeric_column(frame, spec.numerator)
    den = _numeric_column(frame, spec.denominator)
    valid = num.notna() & den.notna() & den.gt(0)
    numerator = float(num.loc[valid].sum()) if valid.any() else np.nan
    denominator = float(den.loc[valid].sum()) if valid.any() else np.nan
    rate = numerator / denominator if valid.any() and denominator > 0 else np.nan
    ci_low, ci_high = wilson_interval(numerator, denominator)
    competency_met = _control_met(rate, spec.competency, spec.direction)
    benchmark_met = _control_met(rate, spec.benchmark, spec.direction)
    if pd.isna(rate):
        status = "insufficient_data"
    elif benchmark_met:
        status = "benchmark"
    elif competency_met:
        status = "competency"
    else:
        status = "outside_control"
    return {
        "metric": metric,
        "label": spec.label,
        "indicator_class": spec.indicator_class,
        "source_note": spec.source_note,
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate,
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
        "direction": spec.direction,
        "competency": spec.competency,
        "benchmark": spec.benchmark,
        "competency_met": competency_met,
        "benchmark_met": benchmark_met,
        "status": status,
        "n_cycles": int(valid.sum()),
    }


def aggregate_kpis(df: pd.DataFrame,
                   group_cols: str | Sequence[str] | None = None,
                   metrics: Iterable[str] | None = None) -> pd.DataFrame:
    """Aggregate KPI as ``sum(numerator) / sum(denominator)``.

    ``group_cols`` may be a single column or a sequence.  Missing grouping
    values are retained as ``<missing>`` to make data-quality gaps visible.
    ``metrics`` restricts which ``KPI_SPECS`` entries are computed (e.g. the
    embryologist/doctor competency split); ``None`` computes all of them.
    """
    d = df.copy() if "_fr_denominator" in df.columns else compute_rates(df)
    groups = [group_cols] if isinstance(group_cols, str) else list(group_cols or [])
    wanted = KPI_SPECS if metrics is None else {
        name: spec for name, spec in KPI_SPECS.items() if name in set(metrics)
    }
    records: list[dict] = []
    if groups:
        missing = [col for col in groups if col not in d.columns]
        if missing:
            raise KeyError(f"KPI grouping columns missing: {missing}")
        iterator: Iterable[tuple[object, pd.DataFrame]] = d.groupby(
            groups[0] if len(groups) == 1 else groups, dropna=False, sort=True
        )
    else:
        iterator = [((), d)]

    for key, frame in iterator:
        keys = key if isinstance(key, tuple) else (key,)
        prefix = {col: value for col, value in zip(groups, keys)}
        for metric, spec in wanted.items():
            records.append({**prefix, **_aggregate_one(frame, metric, spec)})
    return pd.DataFrame.from_records(records)


def _token_match(series: pd.Series, tokens: Sequence[str]) -> pd.Series:
    """Case-insensitive, stripped exact-token match against a raw text column."""
    normed = series.astype("string").str.strip().str.lower()
    wanted = {str(t).strip().lower() for t in tokens}
    return normed.isin(wanted)


def _open_cycle_mask(df: pd.DataFrame) -> pd.Series:
    """Cycles still running at export time, so their blanks are not observations.

    An explicit closure signal wins when the export carries one: Fertimed's
    "Причина завершения программы ВРТ" (``clin_termination_reason``) is filled
    exactly when the programme ended — "Остановка развития эмбрионов",
    "Отсутствие оплодотворения", "Криоконсервация ооцитов" and so on. A blank
    reason is the only genuinely open cycle.

    Without such a column the older proxy stands: no Bl value yet means
    mid-cultivation. That proxy must not be used where a closure signal exists
    — it silently drops every retrieval-only cycle (no Bl by definition) from
    the doctor's OCCR/MIIR, which are computed from follicles/OCC/MII and never
    touch Bl at all (owner confirmed 2026-07-31).
    """
    if "clin_termination_reason" in df.columns:
        reason = df["clin_termination_reason"]
        return ~(reason.notna() & (reason.astype(str).str.strip() != ""))
    return _numeric_column(df, "bl_total").isna()


def competency_exclusion_mask(df: pd.DataFrame, columns_cfg: dict) -> pd.Series:
    """Rows excluded only from the embryologist/doctor competency tables.

    Mirrors the owner's Vienna-style dashboard: cycles still open at export
    time (see ``_open_cycle_mask``), NO-retrieval cycles, and freeze-all
    (Cryo, no fresh transfer planned) cycles. This is deliberately narrower
    than nothing (review-list/OZEMA are unaffected) and does not
    touch the fixed ``2PN==0`` exclusion already applied upstream.
    """
    cfg = columns_cfg.get("competency_exclusion", {}) if columns_cfg else {}
    open_cycle = _open_cycle_mask(df)
    method = df["fertilization_method"] if "fertilization_method" in df.columns else pd.Series(
        pd.NA, index=df.index, dtype="string")
    no_retrieval = _token_match(method, cfg.get("no_retrieval_tokens", []))
    freeze_all = _token_match(method, cfg.get("freeze_all_tokens", []))
    return (open_cycle | no_retrieval | freeze_all).fillna(False)


def embryologist_only_exclusion_mask(df: pd.DataFrame, columns_cfg: dict) -> pd.Series:
    """Rows that stay in the DOCTOR competency table but leave the embryologist one.

    Fertimed's "Только пункция" (retrieval only — oocyte banking, donor
    retrieval, cancelled fertilisation) is a real retrieval by a real doctor:
    follicles/OCC/MII are recorded and OCCR/MIIR must count them (owner
    confirmed 2026-07-31). Nothing downstream of insemination happened, so the
    row carries no FR/ClR/BlR/TGBDR/DegR/3pNR observation at all and must not
    be read as an embryologist's cycle.

    Config-driven (``competency_exclusion.embryologist_only_treatment_tokens``
    matched against ``treatment_type``); a no-op for a clinic without that
    column/config.
    """
    cfg = columns_cfg.get("competency_exclusion", {}) if columns_cfg else {}
    treatment = df["treatment_type"] if "treatment_type" in df.columns else pd.Series(
        pd.NA, index=df.index, dtype="string")
    return _token_match(treatment, cfg.get("embryologist_only_treatment_tokens", [])).fillna(False)


def funnel_exclusion_mask(df: pd.DataFrame, columns_cfg: dict) -> pd.Series:
    """Rows whose funnel counts (2PN/Bl/cleavage/...) are not self-consistent
    for THIS row and must never enter a KPI sum — e.g. cryo-transfer rows
    whose fertilization-stage counts were carried over from the original
    retrieval batch by the source MIS (one batch feeding several transfer
    attempts), mismatched against a per-transfer blastocyst count (owner
    confirmed 2026-07-18 for Fertimed: exclude ``treatment_type``="Криоперенос"
    rows from KPI aggregation specifically for this reason).

    Excludes only from KPI aggregation (cohort-level AND embryologist/doctor
    competency tables) — never from the broader analysis cohort (DT/TabPFN/
    review-list/OZEMA), which keep judging the actual transfer outcome.
    Config-driven (``funnel_exclusion.stale_funnel_tokens`` matched against
    ``treatment_type``); absent for a clinic without that column/config, this
    is a no-op (e.g. clinic A).
    """
    cfg = columns_cfg.get("funnel_exclusion", {}) if columns_cfg else {}
    treatment = df["treatment_type"] if "treatment_type" in df.columns else pd.Series(
        pd.NA, index=df.index, dtype="string")
    return _token_match(treatment, cfg.get("stale_funnel_tokens", [])).fillna(False)


def surgical_sperm_mask(df: pd.DataFrame, columns_cfg: dict) -> pd.Series:
    """Rows using surgical sperm retrieval (TESE/SMF/...), excluded only from FR/FF."""
    cfg = columns_cfg.get("competency_exclusion", {}) if columns_cfg else {}
    source = df["sperm_source"] if "sperm_source" in df.columns else pd.Series(
        pd.NA, index=df.index, dtype="string")
    return _token_match(source, cfg.get("surgical_sperm_tokens", [])).fillna(False)


def aggregate_kpi_slices(df: pd.DataFrame,
                         time_col: str | None = None,
                         method_col: str | None = None,
                         embryologist_col: str | None = None,
                         doctor_col: str | None = None) -> dict[str, pd.DataFrame]:
    """Build overall and requested time/method/operator cohort slices."""
    d = df.copy() if "_fr_denominator" in df.columns else compute_rates(df)
    result = {"overall": aggregate_kpis(d)}
    requested = {
        "time": time_col,
        "method": method_col,
        "embryologist": embryologist_col,
        "doctor": doctor_col,
    }
    for label, col in requested.items():
        if not col or col not in d.columns:
            continue
        group_col = col
        if label == "time":
            parsed = pd.to_datetime(d[col], errors="coerce", dayfirst=True)
            d = d.assign(_kpi_period=parsed.dt.to_period("M").astype("string"))
            group_col = "_kpi_period"
        result[label] = aggregate_kpis(d, group_col)
    return result


def validate_finite(df: pd.DataFrame, cols=None) -> dict:
    """Report missing/infinite values without treating optional MIIR as invalid."""
    cols = cols or ["OCCR", "FR", "Model_FR", "ClR", "BlR", "TGBDR", "MIIR", "DegR", "3pNR",
                    "Model_KPIScore"]
    report = {}
    for col in cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            report[col] = {
                "nan": int(series.isna().sum()),
                "inf": int(np.isinf(series.to_numpy(dtype=float)).sum()),
                "optional": col == "MIIR",
            }
    return report
