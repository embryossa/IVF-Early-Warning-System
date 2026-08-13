"""
tabpfn3_ivf_regression_report_CPU_16K.py
================================

Stage-structured TabPFN-3 regression report for IVF/ICSI cycles.

The script:
1. Reads the 16K development cohort and the prospective OPU table.
2. Resolves Russian/English IVF column aliases into a canonical schema.
3. Fits separate TabPFN-3 regressors for quantitative stages without using
   information from later stages (prevents target/future-stage leakage).
4. Produces patient-level predictive quantiles (P10/P25/P50/P75/P90),
   biologically coherent count limits, and actual-vs-predicted comparisons.
   Cross-validation is intentionally disabled in the CPU-first version.
5. Writes:
      ivf_tabpfn3_regression_report.html
      opu_tabpfn3_regression_predictions.xlsx
      opu_tabpfn3_regression_predictions.csv

Run in PyCharm or from a terminal:
    python tabpfn3_ivf_regression_report_CPU_16K.py

Recommended environment (Python 3.10+):
    pip install -U "tabpfn>=8.1.0" torch pandas openpyxl scikit-learn plotly

Important:
- TabPFN-3 local weights require accepting the Prior Labs model licence.
- Local TabPFN-3 weights are intended for non-commercial evaluation/research
  unless an appropriate commercial licence has been obtained.
- This report is a research/model-audit output, not independently validated
  clinical decision support.
"""

from __future__ import annotations

# Explicitly permit TabPFN to run a >1000-row context on CPU.
# This must be set before importing tabpfn (and preferably before torch).
import os
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

# Import torch before other scientific packages on Windows.
import torch

import argparse
import hashlib
import html
import json
import math
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


# =============================================================================
# USER CONFIGURATION
# =============================================================================

SCRIPT_BUILD = "CPU_16K_NO_CV_v4_2026-07-16"

DEV_PATH = Path(
    r"C:/Users/User/Desktop/IVF/AI/DNN/обучение и валидация/all_df_with_KPI.xlsx"
)
OPU_PATH = Path("OPU table.xlsx")

OUTPUT_HTML = Path("ivf_tabpfn3_regression_report_CPU_16K.html")
OUTPUT_XLSX = Path("opu_tabpfn3_regression_predictions_CPU_16K.xlsx")
OUTPUT_CSV = Path("opu_tabpfn3_regression_predictions_CPU_16K.csv")
CACHE_DIR = Path("tabpfn3_report_cache")

# None = TabPFN downloads/uses the current default TabPFN-3 checkpoint.
# Or give a full local path to a TabPFN-3 regressor checkpoint.
TABPFN_MODEL_PATH: str | None = None

# CPU-first configuration. The complete development cohort is still used as
# in-context support, but the ensemble is reduced to one estimator and repeated
# out-of-fold fitting is disabled.
DEVICE = "cpu"
CPU_THREADS = min(8, max(1, (os.cpu_count() or 2) - 1))
N_ESTIMATORS = 1
RANDOM_STATE = 42
CV_FOLDS = 5  # retained only for dormant legacy helper functions
RUN_CV = False
USE_CV_CACHE = True
MAX_SCATTER_POINTS = 3000
MIN_TARGET_ROWS = 50
MIN_NONMISSING_FEATURE_ROWS = 20
DEFAULT_ATTEMPT = 1.0

# Patient predictions use these quantiles. P10-P90 is the principal interval;
# P25-P75 is also displayed in the interval chart.
PATIENT_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
CV_QUANTILES = [0.10, 0.50, 0.90]

# A self-contained report is larger but works without an internet connection.
EMBED_PLOTLY_JS = True

# Optional exact overrides. Leave None to use aliases below.
# Canonical keys: patient_id, date, age, attempt, follicles, occ, mii, pn2,
# blasts, good, frozen, amh, bmi.
DEV_COLUMN_OVERRIDES: dict[str, str | None] = {
    "age": "Возраст",
    "attempt": "№ попытки",
    "follicles": "Количество фолликулов",
    "occ": "Число ОКК",
    "mii": "Число инсеминированных",
    "pn2": "2 pN",
    "blasts": "Число Bl",
    "good": "Число Bl хор.кач-ва",
    "frozen": "Заморожено эмбрионов",
    # Uncomment/adapt when present in the 16K file:
    # "amh": "AMH",
    # "bmi": "BMI",
    # "date": "Дата OPU",
    # "patient_id": "Patient ID",
}

OPU_COLUMN_OVERRIDES: dict[str, str | None] = {
    "patient_id": "#",
    "age": "Age",
    "attempt": "Attempt",
    "follicles": "N folicules OPU",
    "occ": "OCC",
    "mii": "MII",
    "pn2": "2pN",
    "blasts": "Bl",
    "good": "Good Bl",
    # Add exact names when present:
    # "frozen": "Frozen",
    # "amh": "AMH",
    # "bmi": "BMI",
    # "date": "OPU date",
}


# =============================================================================
# COLUMN RESOLUTION AND MODEL TASKS
# =============================================================================

COLUMN_ALIASES: dict[str, list[str]] = {
    "patient_id": [
        "#", "Patient", "Patient ID", "Patient_ID", "Пациент", "ID",
        "cycle_uid", "Cycle ID", "Cycle_ID",
    ],
    "date": [
        "OPU date", "OPU Date", "Date OPU", "Дата OPU", "Дата пункции",
        "Puncture date", "Дата",
    ],
    "age": ["Возраст", "Age", "Oocyte age", "Возраст ооцитов"],
    "attempt": ["№ попытки", "Attempt", "# Attempt", "Номер попытки"],
    "follicles": [
        "Количество фолликулов", "N folicules OPU", "N follicles OPU",
        "Follicles", "Число фолликулов",
    ],
    "occ": ["Число ОКК", "OCC", "ОКК", "COC", "Oocytes retrieved"],
    "mii": [
        "Число инсеминированных", "MII", "Число MII", "MII oocytes",
        "Mature oocytes",
    ],
    "pn2": ["2 pN", "2pN", "2PN", "Число 2PN", "Normally fertilized"],
    "blasts": [
        "Число Bl", "Bl", "Blastocysts", "Число бластоцист",
        "Total blastocysts",
    ],
    "good": [
        "Число Bl хор.кач-ва", "Good Bl", "Good blastocysts",
        "Good-quality blastocysts", "Число хороших бластоцист",
    ],
    "frozen": [
        "Заморожено эмбрионов", "Frozen", "Usable blastocysts",
        "Usable Bl", "Frozen embryos", "Cryopreserved embryos",
    ],
    "amh": ["AMH", "АМГ", "Anti-Mullerian hormone", "Антимюллеров гормон"],
    "bmi": ["BMI", "ИМТ", "Body mass index"],
}


@dataclass(frozen=True)
class RegressionTask:
    name: str
    label: str
    target_key: str
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...]
    upper_bound_key: str | None
    color: str
    description: str


TASKS: tuple[RegressionTask, ...] = (
    RegressionTask(
        name="mii",
        label="MII oocytes",
        target_key="mii",
        required_features=("age", "occ"),
        optional_features=("attempt", "follicles", "amh", "bmi"),
        upper_bound_key="occ",
        color="#0d6efd",
        description="MII predicted after oocyte retrieval; OCC is known.",
    ),
    RegressionTask(
        name="pn2",
        label="2PN",
        target_key="pn2",
        required_features=("age", "mii"),
        optional_features=("attempt", "follicles", "occ", "amh", "bmi"),
        upper_bound_key="mii",
        color="#20c997",
        description="Normally fertilized oocytes predicted using the observed MII count.",
    ),
    RegressionTask(
        name="blasts_early",
        label="Blastocysts — early",
        target_key="blasts",
        required_features=("age", "mii"),
        optional_features=("attempt", "follicles", "occ", "amh", "bmi"),
        upper_bound_key="mii",
        color="#8f6fd6",
        description="Early blastocyst forecast without using 2PN.",
    ),
    RegressionTask(
        name="blasts_after_fert",
        label="Blastocysts — after fertilization",
        target_key="blasts",
        required_features=("age", "pn2"),
        optional_features=("attempt", "follicles", "occ", "mii", "amh", "bmi"),
        upper_bound_key="pn2",
        color="#6f42c1",
        description="Stage-conditioned blastocyst forecast using the observed 2PN count.",
    ),
    RegressionTask(
        name="good",
        label="Good-quality blastocysts",
        target_key="good",
        required_features=("age", "blasts"),
        optional_features=(
            "attempt", "follicles", "occ", "mii", "pn2", "amh", "bmi"
        ),
        upper_bound_key="blasts",
        color="#0dcaf0",
        description="Good-quality blastocysts predicted after the total blastocyst count is known.",
    ),
    RegressionTask(
        name="frozen",
        label="Usable / frozen embryos",
        target_key="frozen",
        required_features=("age", "blasts"),
        optional_features=(
            "attempt", "follicles", "occ", "mii", "pn2", "good", "amh", "bmi"
        ),
        upper_bound_key="blasts",
        color="#6610f2",
        description="Usable/frozen embryos; included only when this target exists in the data.",
    ),
)


@dataclass
class TaskResult:
    task: RegressionTask
    features: list[str]
    train_rows: int
    patient_quantiles: np.ndarray  # n_patients x len(PATIENT_QUANTILES)
    patient_pred_mask: np.ndarray
    oof_quantiles: np.ndarray | None  # n_dev x len(CV_QUANTILES)
    metrics: dict[str, float | int | None]
    scatter: dict[str, list[float]]
    monthly: list[dict[str, Any]]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def log(message: str) -> None:
    print(message, flush=True)


def normalized_name(value: Any) -> str:
    text = str(value).strip().lower().replace("ё", "е")
    return "".join(ch for ch in text if ch.isalnum())


def resolve_column(
    df: pd.DataFrame,
    key: str,
    overrides: dict[str, str | None],
) -> str | None:
    override = overrides.get(key)
    if override:
        if override in df.columns:
            return override
        # Try normalized matching for accidental spaces/case differences.
        norm = normalized_name(override)
        matches = [c for c in df.columns if normalized_name(c) == norm]
        if len(matches) == 1:
            return matches[0]
        log(f"[warn] configured column for {key!r} not found: {override!r}")

    aliases = COLUMN_ALIASES.get(key, [])
    by_norm: dict[str, list[str]] = {}
    for col in df.columns:
        by_norm.setdefault(normalized_name(col), []).append(str(col))

    for alias in aliases:
        matches = by_norm.get(normalized_name(alias), [])
        if len(matches) == 1:
            return matches[0]
    return None


def canonicalize(
    df: pd.DataFrame,
    overrides: dict[str, str | None],
    source_name: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    out = pd.DataFrame(index=df.index)
    resolved: dict[str, str] = {}
    for key in COLUMN_ALIASES:
        column = resolve_column(df, key, overrides)
        if column is None:
            continue
        resolved[key] = column
        if key in {"patient_id"}:
            out[key] = df[column].astype("string")
        elif key == "date":
            out[key] = pd.to_datetime(df[column], errors="coerce", dayfirst=True)
        else:
            out[key] = pd.to_numeric(df[column], errors="coerce")

    if "attempt" in out.columns:
        out["attempt"] = out["attempt"].fillna(DEFAULT_ATTEMPT)

    log(f"\n[{source_name}] resolved columns:")
    for key, value in resolved.items():
        log(f"  {key:<12} <- {value}")
    missing = [k for k in COLUMN_ALIASES if k not in resolved]
    if missing:
        log(f"  not found: {', '.join(missing)}")
    return out, resolved


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=path.parent,
        prefix=path.stem + ".tmp.",
        suffix=path.suffix,
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        handle.write(text)
    os.replace(tmp, path)


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}{path.suffix}")
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}{path.suffix}")
    try:
        with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
            for sheet_name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def finite_or_none(value: Any, digits: int | None = None) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if digits is None:
        return number
    return round(number, digits)


def dataframe_has_signal(df: pd.DataFrame, key: str, minimum: int) -> bool:
    return key in df.columns and int(df[key].notna().sum()) >= minimum


def choose_task_features(
    task: RegressionTask,
    dev: pd.DataFrame,
    opu: pd.DataFrame,
) -> list[str] | None:
    for key in task.required_features:
        if not dataframe_has_signal(dev, key, MIN_NONMISSING_FEATURE_ROWS):
            log(f"[skip] {task.name}: required DEV feature missing/empty: {key}")
            return None
        if key not in opu.columns:
            log(f"[skip] {task.name}: required OPU feature column missing: {key}")
            return None
        if int(opu[key].notna().sum()) == 0:
            log(f"[skip] {task.name}: required OPU feature has no observed values: {key}")
            return None

    features = list(task.required_features)
    for key in task.optional_features:
        if (
            dataframe_has_signal(dev, key, MIN_NONMISSING_FEATURE_ROWS)
            and key in opu.columns
            and int(opu[key].notna().sum()) > 0
        ):
            features.append(key)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(features))


def make_regressor():
    from tabpfn import TabPFNRegressor

    kwargs: dict[str, Any] = {
        "device": DEVICE,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "fit_mode": "fit_preprocessors",
        "memory_saving_mode": "auto",
        "n_preprocessing_jobs": 1,
        # Required for a >1000-row context on CPU. The same permission is also
        # set through TABPFN_ALLOW_CPU_LARGE_DATASET=1 before importing TabPFN.
        "ignore_pretraining_limits": True,
    }
    if TABPFN_MODEL_PATH:
        kwargs["model_path"] = TABPFN_MODEL_PATH
    return TabPFNRegressor(**kwargs)


def stack_quantile_output(output: Sequence[np.ndarray], n_rows: int) -> np.ndarray:
    arrays = [np.asarray(item, dtype=float).reshape(-1) for item in output]
    if any(len(item) != n_rows for item in arrays):
        raise RuntimeError("Unexpected TabPFN quantile output shape.")
    return np.column_stack(arrays)


def biologically_clip(
    quantile_matrix: np.ndarray,
    frame: pd.DataFrame,
    upper_bound_key: str | None,
) -> np.ndarray:
    q = np.asarray(quantile_matrix, dtype=float).copy()
    q[~np.isfinite(q)] = np.nan
    q = np.maximum(q, 0.0)

    if upper_bound_key and upper_bound_key in frame.columns:
        upper = pd.to_numeric(frame[upper_bound_key], errors="coerce").to_numpy(float)
        finite_upper = np.isfinite(upper)
        for col in range(q.shape[1]):
            q[finite_upper, col] = np.minimum(q[finite_upper, col], upper[finite_upper])

    # Numerical safety: enforce non-decreasing quantiles row-wise.
    q = np.maximum.accumulate(q, axis=1)
    return q


def predict_quantiles(
    model: Any,
    X: pd.DataFrame,
    quantiles: Sequence[float],
    upper_bound_key: str | None,
    canonical_frame: pd.DataFrame,
) -> np.ndarray:
    raw = model.predict(
        X,
        output_type="quantiles",
        quantiles=list(quantiles),
    )
    stacked = stack_quantile_output(raw, len(X))
    return biologically_clip(stacked, canonical_frame, upper_bound_key)


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_path_for_task(
    dev_path: Path,
    task: RegressionTask,
    features: list[str],
    n_rows: int,
) -> Path:
    try:
        import tabpfn

        version = getattr(tabpfn, "__version__", "unknown")
    except Exception:
        version = "unknown"
    payload = {
        "dev": file_fingerprint(dev_path),
        "task": asdict(task),
        "features": features,
        "rows": n_rows,
        "cv_folds": CV_FOLDS,
        "cv_quantiles": CV_QUANTILES,
        "n_estimators": N_ESTIMATORS,
        "tabpfn_version": version,
        "seed": RANDOM_STATE,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return CACHE_DIR / f"{task.name}_oof_{digest}.npz"


def load_cached_oof(path: Path, n_rows: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            q = np.asarray(payload["quantiles"], dtype=float)
        if q.shape != (n_rows, len(CV_QUANTILES)):
            return None
        log(f"[cache] loaded {path}")
        return q
    except Exception as exc:
        log(f"[warn] failed to read CV cache {path}: {exc}")
        return None


def save_cached_oof(path: Path, quantiles: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, quantiles=quantiles)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def cross_validated_quantiles(
    task: RegressionTask,
    dev: pd.DataFrame,
    features: list[str],
    dev_path: Path,
) -> np.ndarray:
    n_rows = len(dev)
    cache_path: Path | None = None
    if USE_CV_CACHE:
        cache_path = cache_path_for_task(dev_path, task, features, n_rows)
        cached = load_cached_oof(cache_path, n_rows)
        if cached is not None:
            return cached

    target = pd.to_numeric(dev[task.target_key], errors="coerce")
    valid_indices = np.flatnonzero(target.notna().to_numpy())
    if len(valid_indices) < max(MIN_TARGET_ROWS, CV_FOLDS * 2):
        raise RuntimeError(
            f"Not enough target rows for {task.name}: {len(valid_indices)}"
        )

    splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full((n_rows, len(CV_QUANTILES)), np.nan, dtype=float)

    X_all = dev.loc[:, features]
    y_all = target.to_numpy(float)

    for fold, (train_local, valid_local) in enumerate(
        splitter.split(valid_indices), start=1
    ):
        train_idx = valid_indices[train_local]
        valid_idx = valid_indices[valid_local]
        log(
            f"    CV fold {fold}/{CV_FOLDS}: train={len(train_idx)}, "
            f"valid={len(valid_idx)}"
        )
        model = make_regressor()
        model.fit(X_all.iloc[train_idx], y_all[train_idx])
        pred = predict_quantiles(
            model,
            X_all.iloc[valid_idx],
            CV_QUANTILES,
            task.upper_bound_key,
            dev.iloc[valid_idx],
        )
        oof[valid_idx] = pred
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if USE_CV_CACHE and cache_path is not None:
        save_cached_oof(cache_path, oof)
        log(f"[cache] saved {cache_path}")
    return oof


def calculate_metrics(
    y: np.ndarray,
    q: np.ndarray | None,
) -> dict[str, float | int | None]:
    if q is None:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "bias": None,
            "coverage80": None,
            "width80": None,
        }
    mask = np.isfinite(y) & np.isfinite(q[:, 1])
    if not np.any(mask):
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "bias": None,
            "coverage80": None,
            "width80": None,
        }
    yy = y[mask]
    lo, med, hi = q[mask, 0], q[mask, 1], q[mask, 2]
    try:
        r2 = r2_score(yy, med) if len(yy) > 1 and np.nanstd(yy) > 0 else np.nan
    except Exception:
        r2 = np.nan
    return {
        "n": int(len(yy)),
        "mae": finite_or_none(mean_absolute_error(yy, med), 4),
        "rmse": finite_or_none(math.sqrt(mean_squared_error(yy, med)), 4),
        "r2": finite_or_none(r2, 4),
        "bias": finite_or_none(np.mean(med - yy), 4),
        "coverage80": finite_or_none(np.mean((yy >= lo) & (yy <= hi)), 4),
        "width80": finite_or_none(np.mean(hi - lo), 4),
    }


def downsample_scatter(
    y: np.ndarray,
    q: np.ndarray | None,
    max_points: int,
) -> dict[str, list[float]]:
    if q is None:
        return {"actual": [], "p10": [], "p50": [], "p90": []}
    mask = np.isfinite(y) & np.isfinite(q[:, 1])
    idx = np.flatnonzero(mask)
    if len(idx) > max_points:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    return {
        "actual": np.round(y[idx], 4).tolist(),
        "p10": np.round(q[idx, 0], 4).tolist(),
        "p50": np.round(q[idx, 1], 4).tolist(),
        "p90": np.round(q[idx, 2], 4).tolist(),
    }


def monthly_summary(
    dates: pd.Series | None,
    y: np.ndarray,
    q: np.ndarray | None,
) -> list[dict[str, Any]]:
    if dates is None or q is None:
        return []
    date_values = pd.to_datetime(dates, errors="coerce")
    mask = date_values.notna().to_numpy() & np.isfinite(y) & np.isfinite(q[:, 1])
    if not np.any(mask):
        return []

    frame = pd.DataFrame(
        {
            "period": date_values[mask].dt.to_period("M").astype(str),
            "actual": y[mask],
            "pred": q[mask, 1],
        }
    )
    rows: list[dict[str, Any]] = []
    for period, group in frame.groupby("period", sort=True):
        actual = group["actual"].to_numpy(float)
        pred = group["pred"].to_numpy(float)
        if len(actual) >= 2:
            rng = np.random.default_rng(RANDOM_STATE + len(rows))
            boot = np.mean(
                rng.choice(actual, size=(1000, len(actual)), replace=True), axis=1
            )
            lo, hi = np.percentile(boot, [2.5, 97.5])
        else:
            lo = hi = actual[0]
        rows.append(
            {
                "period": str(period),
                "n": int(len(group)),
                "actual": round(float(np.mean(actual)), 4),
                "predicted": round(float(np.mean(pred)), 4),
                "actual_ci_low": round(float(lo), 4),
                "actual_ci_high": round(float(hi), 4),
                "difference": round(float(np.mean(actual) - np.mean(pred)), 4),
            }
        )
    return rows


def patient_identifier(opu: pd.DataFrame, row_number: int) -> str:
    if "patient_id" in opu.columns:
        value = opu.iloc[row_number]["patient_id"]
        if pd.notna(value) and str(value).strip() not in {"", "<NA>", "nan"}:
            return str(value).strip()
    return f"Cycle {row_number + 1}"


def actual_value(frame: pd.DataFrame, row: int, key: str) -> float | None:
    if key not in frame.columns:
        return None
    return finite_or_none(frame.iloc[row][key], 4)


# =============================================================================
# MODEL EXECUTION
# =============================================================================


def run_task(
    task: RegressionTask,
    dev: pd.DataFrame,
    opu: pd.DataFrame,
    dev_path: Path,
    run_cv: bool,
) -> TaskResult | None:
    if task.target_key not in dev.columns:
        log(f"[skip] {task.name}: target {task.target_key!r} not found in DEV")
        return None
    target_rows = int(dev[task.target_key].notna().sum())
    if target_rows < MIN_TARGET_ROWS:
        log(f"[skip] {task.name}: only {target_rows} target rows")
        return None

    features = choose_task_features(task, dev, opu)
    if not features:
        return None

    log("\n" + "=" * 76)
    log(f"[task] {task.label}")
    log(f"       target={task.target_key}; features={features}; rows={target_rows}")

    oof: np.ndarray | None = None
    if run_cv:
        log("[cv] generating out-of-fold predictions ...")
        oof = cross_validated_quantiles(task, dev, features, dev_path)

    y_dev = pd.to_numeric(dev[task.target_key], errors="coerce").to_numpy(float)
    metrics = calculate_metrics(y_dev, oof)
    scatter = downsample_scatter(y_dev, oof, MAX_SCATTER_POINTS)
    monthly = monthly_summary(dev.get("date"), y_dev, oof)

    target_mask = pd.to_numeric(dev[task.target_key], errors="coerce").notna().to_numpy()
    X_train = dev.loc[target_mask, features]
    y_train = y_dev[target_mask]

    log(f"[fit] final model on {len(X_train)} rows ...")
    model = make_regressor()
    model.fit(X_train, y_train)

    X_query = opu.loc[:, features]
    query_mask = X_query.notna().any(axis=1).to_numpy()
    patient_q = np.full((len(opu), len(PATIENT_QUANTILES)), np.nan, dtype=float)
    if np.any(query_mask):
        patient_q[query_mask] = predict_quantiles(
            model,
            X_query.loc[query_mask],
            PATIENT_QUANTILES,
            task.upper_bound_key,
            opu.loc[query_mask],
        )
    log(f"[ok] predicted {int(query_mask.sum())}/{len(opu)} OPU rows")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return TaskResult(
        task=task,
        features=features,
        train_rows=len(X_train),
        patient_quantiles=patient_q,
        patient_pred_mask=query_mask,
        oof_quantiles=oof,
        metrics=metrics,
        scatter=scatter,
        monthly=monthly,
    )


# =============================================================================
# OUTPUT TABLES
# =============================================================================


def append_prediction_columns(
    original_opu: pd.DataFrame,
    results: list[TaskResult],
) -> pd.DataFrame:
    out = original_opu.copy()
    suffixes = ["p10", "p25", "p50", "p75", "p90"]
    for result in results:
        for col_idx, suffix in enumerate(suffixes):
            out[f"TPFN3_{result.task.name}_{suffix}"] = np.round(
                result.patient_quantiles[:, col_idx], 4
            )
        if result.task.target_key in {"mii", "pn2", "blasts", "good", "frozen"}:
            out[f"TPFN3_{result.task.name}_interval80_width"] = np.round(
                result.patient_quantiles[:, 4] - result.patient_quantiles[:, 0], 4
            )
    return out


def metrics_dataframe(results: list[TaskResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row: dict[str, Any] = {
            "validation": "not run (CPU mode)",
            "task": result.task.name,
            "label": result.task.label,
            "target": result.task.target_key,
            "features": ", ".join(result.features),
            "train_rows": result.train_rows,
        }
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# HTML REPORT
# =============================================================================


def build_report_payload(
    dev: pd.DataFrame,
    opu: pd.DataFrame,
    results: list[TaskResult],
    dev_path: Path,
    opu_path: Path,
    run_cv: bool,
) -> dict[str, Any]:
    result_by_name = {result.task.name: result for result in results}
    task_payload: dict[str, Any] = {}
    for result in results:
        task_payload[result.task.name] = {
            "name": result.task.name,
            "label": result.task.label,
            "target_key": result.task.target_key,
            "features": result.features,
            "description": result.task.description,
            "color": result.task.color,
            "train_rows": result.train_rows,
            "metrics": result.metrics,
            "scatter": result.scatter,
            "monthly": result.monthly,
        }

    patients = []
    q_names = ["p10", "p25", "p50", "p75", "p90"]
    for row in range(len(opu)):
        predictions: dict[str, Any] = {}
        for result in results:
            values = result.patient_quantiles[row]
            predictions[result.task.name] = {
                q_names[i]: finite_or_none(values[i], 4) for i in range(len(q_names))
            }
            predictions[result.task.name]["actual"] = actual_value(
                opu, row, result.task.target_key
            )

        actual = {
            key: actual_value(opu, row, key)
            for key in [
                "age", "attempt", "follicles", "occ", "mii", "pn2",
                "blasts", "good", "frozen", "amh", "bmi",
            ]
        }
        patients.append(
            {
                "id": patient_identifier(opu, row),
                "row": row + 1,
                "actual": actual,
                "predictions": predictions,
            }
        )

    return {
        "meta": {
            "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dev_path": str(dev_path),
            "opu_path": str(opu_path),
            "dev_rows": len(dev),
            "opu_rows": len(opu),
            "device": DEVICE,
            "n_estimators": N_ESTIMATORS,
            "cv_folds": CV_FOLDS if run_cv else 0,
            "cv_enabled": bool(run_cv),
            "cpu_threads": CPU_THREADS,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "task_order": [r.task.name for r in results],
        "tasks": task_payload,
        "patients": patients,
        "cascade_order": [
            name
            for name in ["mii", "pn2", "blasts_after_fert", "good", "frozen"]
            if name in result_by_name
        ],
    }


def json_for_html(payload: dict[str, Any]) -> str:
    # Prevent accidental </script> termination.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def plotly_script_tag() -> str:
    if EMBED_PLOTLY_JS:
        try:
            from plotly.offline import get_plotlyjs

            return f"<script>{get_plotlyjs()}</script>"
        except Exception as exc:
            log(f"[warn] could not embed Plotly JS ({exc}); using CDN")
    return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


def build_html(payload: dict[str, Any]) -> str:
    data_json = json_for_html(payload)
    generated = html.escape(payload["meta"]["generated"])
    plotly_tag = plotly_script_tag()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IVF TabPFN-3 Quantitative Report</title>
{plotly_tag}
<style>
:root{{--bg:#f5f7fb;--card:#fff;--line:#dfe5ec;--text:#233044;--muted:#6d7b8d;--blue:#0d6efd;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;font-size:14px}}
.container{{max-width:1450px;margin:0 auto;padding:20px}}
.header{{background:linear-gradient(135deg,#16243a,#263f66);color:#fff;border-radius:16px;padding:22px 26px;box-shadow:0 8px 26px rgba(20,40,70,.18)}}
.header h1{{margin:0 0 7px;font-size:25px}}
.header p{{margin:3px 0;color:#d9e4f5}}
.toolbar{{display:flex;gap:14px;align-items:end;flex-wrap:wrap;margin:18px 0}}
.field{{min-width:280px}}
label{{display:block;font-size:12px;font-weight:700;color:var(--muted);margin-bottom:5px;text-transform:uppercase;letter-spacing:.04em}}
select{{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--text)}}
.section-title{{font-size:18px;font-weight:800;margin:24px 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:14px;box-shadow:0 3px 12px rgba(25,45,75,.05)}}
.kpi{{grid-column:span 2;min-height:112px}}
.kpi .label{{font-size:12px;color:var(--muted);font-weight:700;min-height:31px}}
.kpi .value{{font-size:25px;font-weight:800;margin-top:6px}}
.kpi .sub{{font-size:11px;color:var(--muted);margin-top:4px;line-height:1.35}}
.chart-half{{grid-column:span 6}}
.chart-full{{grid-column:span 12}}
.hint{{font-size:12px;color:var(--muted);margin:3px 0 8px;line-height:1.45}}
.plot{{height:330px}}
.plot-tall{{height:390px}}
.table-wrap{{overflow:auto;max-height:480px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th,td{{padding:8px 9px;border-bottom:1px solid #edf0f4;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
th{{position:sticky;top:0;background:#f8fafc;z-index:1;color:#556274}}
.badge{{display:inline-block;padding:4px 8px;border-radius:20px;background:#e9f2ff;color:#0d6efd;font-size:11px;font-weight:700}}
.warning{{border-left:4px solid #f0ad4e;background:#fff8e8;padding:12px 14px;border-radius:8px;margin-top:14px;color:#725619}}
.disclaimer{{font-size:11px;color:var(--muted);line-height:1.45;margin:20px 2px}}
.meta{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}
.meta span{{background:rgba(255,255,255,.11);padding:5px 8px;border-radius:20px;font-size:11px}}
@media(max-width:1100px){{.kpi{{grid-column:span 4}}.chart-half{{grid-column:span 12}}}}
@media(max-width:650px){{.container{{padding:10px}}.kpi{{grid-column:span 6}}.header h1{{font-size:21px}}}}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>IVF Analytics — TabPFN-3 quantitative prediction</h1>
    <p>Stage-structured, data-driven forecasts learned in-context from the development cohort.</p>
    <p>Generated {generated}</p>
    <div class="meta" id="metaBadges"></div>
  </div>

  <div class="warning">
    CPU mode: each final stage model is fitted once on the full development cohort;
    5-fold cross-validation is disabled. Predictions are stage-specific. The “early
    blastocyst” model does not use 2PN; the “after fertilization” model does. Later-stage
    values are never used to predict an earlier stage. Intervals are model predictive
    quantiles, not externally validated confidence intervals.
  </div>

  <div class="toolbar">
    <div class="field"><label>Patient / cycle</label><select id="patientSelect"></select></div>
    <div class="field" id="validationToolbar"><label>Validation target</label><select id="taskSelect"></select></div>
  </div>

  <div class="section-title">🔮 Per-patient forecast</div>
  <div class="grid" id="patientCards"></div>

  <div class="grid" style="margin-top:12px">
    <div class="card chart-half">
      <b>Forecast vs actual — cascade</b>
      <div class="hint">Median prediction with P10–P90 whiskers; blue bars show the observed patient result.</div>
      <div id="cascadePlot" class="plot"></div>
    </div>
    <div class="card chart-half">
      <b>Expected vs actual — process rates</b>
      <div class="hint">Rates use the observed upstream denominator when available.</div>
      <div id="ratePlot" class="plot"></div>
    </div>
    <div class="card chart-full">
      <b>Predictive intervals by stage</b>
      <div class="hint">Thin line=P10–P90, thick line=P25–P75, circle=P50, diamond=actual.</div>
      <div id="intervalPlot" class="plot-tall"></div>
    </div>
    <div class="card chart-full">
      <b>Stage detail</b>
      <div class="table-wrap"><table id="patientTable"></table></div>
    </div>
  </div>

  <div id="validationBlock">
    <div class="section-title">🤖 Out-of-fold model validation</div>
    <div class="grid" id="metricCards"></div>
    <div class="grid" style="margin-top:12px">
      <div class="card chart-half">
        <b>Predicted vs actual</b>
        <div class="hint">Each point is an out-of-fold historical cycle. Vertical whiskers show P10–P90.</div>
        <div id="scatterPlot" class="plot"></div>
      </div>
      <div class="card chart-half">
        <b>Predicted vs actual by month</b>
        <div class="hint">Displayed only when a development-cohort OPU date column was found.</div>
        <div id="monthlyPlot" class="plot"></div>
      </div>
      <div class="card chart-full">
        <b>All regression tasks</b>
        <div class="table-wrap"><table id="metricsTable"></table></div>
      </div>
    </div>
  </div>

  <div class="disclaimer">
    Research/model-audit report. It does not establish clinical validity, causal effects,
    or treatment recommendations. Quantitative performance must be confirmed prospectively
    and externally, with attention to clinic drift and repeated cycles from the same patient.
  </div>
</div>

<script>
const DATA={data_json};
const $=id=>document.getElementById(id);
const fmt=(x,d=1)=>x===null||x===undefined||!Number.isFinite(Number(x))?'—':Number(x).toFixed(d).replace(/\\.0$/,'');
const pct=x=>x===null||x===undefined||!Number.isFinite(Number(x))?'—':(100*Number(x)).toFixed(1).replace(/\\.0$/,'')+'%';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const transparent={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{family:'Inter,Segoe UI,Arial',size:11,color:'#344054'}}}};

function init(){{
  $('metaBadges').innerHTML=[
    `DEV ${{DATA.meta.dev_rows}} cycles`,`OPU ${{DATA.meta.opu_rows}} cycles`,
    `device ${{DATA.meta.device}}`,`CPU threads ${{DATA.meta.cpu_threads}}`,
    `estimators ${{DATA.meta.n_estimators}}`,
    DATA.meta.cv_enabled?`CV ${{DATA.meta.cv_folds}}-fold`:'CV disabled' 
  ].map(x=>`<span>${{esc(x)}}</span>`).join('');
  $('patientSelect').innerHTML=DATA.patients.map((p,i)=>`<option value="${{i}}">${{esc(p.id)}}</option>`).join('');
  $('taskSelect').innerHTML=DATA.task_order.map(n=>`<option value="${{n}}">${{esc(DATA.tasks[n].label)}}</option>`).join('');
  $('patientSelect').addEventListener('change',renderPatient);
  $('taskSelect').addEventListener('change',renderValidation);
  if(!DATA.meta.cv_enabled){{
    $('validationToolbar').style.display='none';
    $('validationBlock').style.display='none';
  }}else{{
    renderMetricsTable();
    renderValidation();
  }}
  renderPatient();
}}

function predictionCard(task,pred){{
  const t=DATA.tasks[task];
  const actual=pred.actual;
  const residual=(actual!==null&&pred.p50!==null)?actual-pred.p50:null;
  return `<div class="card kpi">
    <div class="label">${{esc(t.label)}}</div>
    <div class="value" style="color:${{t.color}}">${{fmt(pred.p50,1)}}</div>
    <div class="sub">P10–P90 ${{fmt(pred.p10,1)}}–${{fmt(pred.p90,1)}}<br>actual ${{fmt(actual,1)}}${{residual===null?'':` · residual ${{residual>=0?'+':''}}${{fmt(residual,1)}}`}}</div>
  </div>`;
}}

function renderPatient(){{
  const p=DATA.patients[Number($('patientSelect').value)||0];
  $('patientCards').innerHTML=DATA.task_order.map(n=>predictionCard(n,p.predictions[n])).join('');
  renderCascade(p);
  renderRates(p);
  renderIntervals(p);
  renderPatientTable(p);
}}

function renderCascade(p){{
  const names=DATA.cascade_order;
  const labels=names.map(n=>DATA.tasks[n].label);
  const med=names.map(n=>p.predictions[n].p50);
  const lo=names.map(n=>p.predictions[n].p10);
  const hi=names.map(n=>p.predictions[n].p90);
  const actual=names.map(n=>p.predictions[n].actual);
  const errPlus=med.map((x,i)=>x===null||hi[i]===null?0:Math.max(0,hi[i]-x));
  const errMinus=med.map((x,i)=>x===null||lo[i]===null?0:Math.max(0,x-lo[i]));
  const traces=[
    {{type:'bar',name:'TabPFN-3 median',x:labels,y:med,marker:{{color:'#adb5bd'}},error_y:{{type:'data',symmetric:false,array:errPlus,arrayminus:errMinus,color:'#777',thickness:1.2,width:3}}}},
    {{type:'bar',name:'Actual',x:labels,y:actual,marker:{{color:'#0d6efd'}}}}
  ];
  Plotly.react('cascadePlot',traces,{{...transparent,barmode:'group',margin:{{l:44,r:12,t:10,b:70}},legend:{{orientation:'h'}},yaxis:{{title:'count',rangemode:'tozero'}},xaxis:{{tickangle:-18}}}},{{responsive:true,displaylogo:false}});
}}

function safeRate(num,den){{return num!==null&&den!==null&&Number(den)>0?100*Number(num)/Number(den):null;}}
function renderRates(p){{
  const a=p.actual, pr=p.predictions;
  const stages=[];
  function add(label,pred,actual){{if(pred!==null||actual!==null)stages.push({{label,pred,actual}})}}
  add('MII / OCC',pr.mii?safeRate(pr.mii.p50,a.occ):null,safeRate(a.mii,a.occ));
  add('2PN / MII',pr.pn2?safeRate(pr.pn2.p50,a.mii):null,safeRate(a.pn2,a.mii));
  const blastTask=pr.blasts_after_fert?'blasts_after_fert':(pr.blasts_early?'blasts_early':null);
  add('Blast / 2PN',blastTask?safeRate(pr[blastTask].p50,a.pn2):null,safeRate(a.blasts,a.pn2));
  add('Good / blast',pr.good?safeRate(pr.good.p50,a.blasts):null,safeRate(a.good,a.blasts));
  add('Frozen / blast',pr.frozen?safeRate(pr.frozen.p50,a.blasts):null,safeRate(a.frozen,a.blasts));
  Plotly.react('ratePlot',[
    {{type:'bar',name:'TabPFN-3',x:stages.map(x=>x.label),y:stages.map(x=>x.pred),marker:{{color:'#adb5bd'}}}},
    {{type:'bar',name:'Actual',x:stages.map(x=>x.label),y:stages.map(x=>x.actual),marker:{{color:'#20c997'}}}}
  ],{{...transparent,barmode:'group',margin:{{l:44,r:12,t:10,b:70}},legend:{{orientation:'h'}},yaxis:{{title:'rate',ticksuffix:'%',range:[0,110]}},xaxis:{{tickangle:-18}}}},{{responsive:true,displaylogo:false}});
}}

function renderIntervals(p){{
  const tasks=DATA.task_order.filter(n=>p.predictions[n].p50!==null);
  const labels=tasks.map(n=>DATA.tasks[n].label);
  const traces=[];
  tasks.forEach((n,i)=>{{
    const x=p.predictions[n], y=labels[i], c=DATA.tasks[n].color;
    traces.push({{type:'scatter',mode:'lines',x:[x.p10,x.p90],y:[y,y],line:{{color:c,width:3}},showlegend:false,hoverinfo:'skip'}});
    traces.push({{type:'scatter',mode:'lines',x:[x.p25,x.p75],y:[y,y],line:{{color:c,width:9}},showlegend:false,hoverinfo:'skip'}});
    traces.push({{type:'scatter',mode:'markers',x:[x.p50],y:[y],marker:{{color:c,size:10,symbol:'circle'}},name:'P50',showlegend:i===0,hovertemplate:'P50 %{{x:.2f}}<extra></extra>'}});
    if(x.actual!==null)traces.push({{type:'scatter',mode:'markers',x:[x.actual],y:[y],marker:{{color:'#111827',size:10,symbol:'diamond-open',line:{{width:2}}}},name:'Actual',showlegend:i===0,hovertemplate:'Actual %{{x:.2f}}<extra></extra>'}});
  }});
  Plotly.react('intervalPlot',traces,{{...transparent,margin:{{l:190,r:20,t:10,b:45}},legend:{{orientation:'h'}},xaxis:{{title:'count',rangemode:'tozero'}},yaxis:{{categoryorder:'array',categoryarray:[...labels].reverse()}}}},{{responsive:true,displaylogo:false}});
}}

function renderPatientTable(p){{
  const head='<thead><tr><th>Stage</th><th>P10</th><th>P25</th><th>P50</th><th>P75</th><th>P90</th><th>Actual</th><th>Actual−P50</th><th>Inputs</th></tr></thead>';
  const rows=DATA.task_order.map(n=>{{const t=DATA.tasks[n],x=p.predictions[n];const d=(x.actual!==null&&x.p50!==null)?x.actual-x.p50:null;return `<tr><td>${{esc(t.label)}}</td><td>${{fmt(x.p10,2)}}</td><td>${{fmt(x.p25,2)}}</td><td><b>${{fmt(x.p50,2)}}</b></td><td>${{fmt(x.p75,2)}}</td><td>${{fmt(x.p90,2)}}</td><td>${{fmt(x.actual,2)}}</td><td>${{d===null?'—':(d>=0?'+':'')+fmt(d,2)}}</td><td style="text-align:left">${{esc(t.features.join(', '))}}</td></tr>`}}).join('');
  $('patientTable').innerHTML=head+'<tbody>'+rows+'</tbody>';
}}

function renderValidation(){{
  if(!DATA.meta.cv_enabled)return;
  const n=$('taskSelect').value||DATA.task_order[0];
  const t=DATA.tasks[n],m=t.metrics,s=t.scatter;
  const cards=[
    ['N',m.n,'OOF cycles'],['MAE',m.mae,'average absolute miss'],['RMSE',m.rmse,'root mean square error'],
    ['R²',m.r2,'variance explained'],['80% coverage',m.coverage80===null?'—':pct(m.coverage80),'target ≈80%'],['Mean 80% width',m.width80,'P90−P10']
  ];
  $('metricCards').innerHTML=cards.map((x,i)=>`<div class="card kpi"><div class="label">${{esc(x[0])}}</div><div class="value" style="color:${{t.color}}">${{typeof x[1]==='number'?fmt(x[1],i===0?0:3):x[1]}}</div><div class="sub">${{esc(x[2])}}</div></div>`).join('');

  const maxVal=Math.max(1,...s.actual,...s.p90);
  Plotly.react('scatterPlot',[
    {{type:'scatter',mode:'markers',x:s.actual,y:s.p50,marker:{{color:t.color,size:6,opacity:.55}},error_y:{{type:'data',symmetric:false,array:s.p90.map((v,i)=>Math.max(0,v-s.p50[i])),arrayminus:s.p50.map((v,i)=>Math.max(0,v-s.p10[i])),thickness:.6,width:0,color:'rgba(80,80,80,.35)'}},hovertemplate:'actual %{{x:.1f}}<br>P50 %{{y:.1f}}<extra></extra>'}},
    {{type:'scatter',mode:'lines',x:[0,maxVal],y:[0,maxVal],line:{{color:'#98a2b3',dash:'dash'}},hoverinfo:'skip',showlegend:false}}
  ],{{...transparent,margin:{{l:48,r:15,t:10,b:45}},showlegend:false,xaxis:{{title:'actual',rangemode:'tozero'}},yaxis:{{title:'predicted P50',rangemode:'tozero'}}}},{{responsive:true,displaylogo:false}});

  if(t.monthly&&t.monthly.length){{
    const mo=t.monthly;
    Plotly.react('monthlyPlot',[
      {{type:'scatter',mode:'lines',x:mo.map(x=>x.period),y:mo.map(x=>x.actual_ci_high),line:{{width:0}},hoverinfo:'skip',showlegend:false}},
      {{type:'scatter',mode:'lines',x:mo.map(x=>x.period),y:mo.map(x=>x.actual_ci_low),fill:'tonexty',fillcolor:'rgba(32,201,151,.15)',line:{{width:0}},name:'actual 95% CI',hoverinfo:'skip'}},
      {{type:'scatter',mode:'lines+markers',x:mo.map(x=>x.period),y:mo.map(x=>x.predicted),name:'Predicted P50',line:{{color:t.color,width:2,dash:'dot'}}}},
      {{type:'scatter',mode:'lines+markers',x:mo.map(x=>x.period),y:mo.map(x=>x.actual),name:'Actual',line:{{color:'#198754',width:2}},customdata:mo.map(x=>x.n),hovertemplate:'%{{x}}<br>actual %{{y:.2f}}<br>n=%{{customdata}}<extra></extra>'}}
    ],{{...transparent,margin:{{l:48,r:15,t:10,b:55}},legend:{{orientation:'h'}},xaxis:{{tickangle:-35}},yaxis:{{title:'mean count',rangemode:'tozero'}}}},{{responsive:true,displaylogo:false}});
  }}else{{
    Plotly.purge('monthlyPlot');
    $('monthlyPlot').innerHTML='<div style="padding:95px 20px;text-align:center;color:#7b8794">No development-cohort date column was resolved.</div>';
  }}
}}

function renderMetricsTable(){{
  if(!DATA.meta.cv_enabled)return;
  const head='<thead><tr><th>Task</th><th>N</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Bias</th><th>80% coverage</th><th>Mean width</th><th>Features</th></tr></thead>';
  const rows=DATA.task_order.map(n=>{{const t=DATA.tasks[n],m=t.metrics;return `<tr><td>${{esc(t.label)}}</td><td>${{fmt(m.n,0)}}</td><td>${{fmt(m.mae,3)}}</td><td>${{fmt(m.rmse,3)}}</td><td>${{fmt(m.r2,3)}}</td><td>${{fmt(m.bias,3)}}</td><td>${{pct(m.coverage80)}}</td><td>${{fmt(m.width80,3)}}</td><td style="text-align:left">${{esc(t.features.join(', '))}}</td></tr>`}}).join('');
  $('metricsTable').innerHTML=head+'<tbody>'+rows+'</tbody>';
}}

window.addEventListener('load',init);
</script>
</body>
</html>"""


# =============================================================================
# MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a stage-structured IVF TabPFN-3 regression report."
    )
    parser.add_argument("--dev-path", type=Path, default=DEV_PATH)
    parser.add_argument("--opu-path", type=Path, default=OPU_PATH)
    parser.add_argument("--output-html", type=Path, default=OUTPUT_HTML)
    parser.add_argument("--output-xlsx", type=Path, default=OUTPUT_XLSX)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    return parser.parse_args()


def validate_environment(dev_rows: int) -> None:
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this setting only before parallel work has started.
        pass
    log("\nEnvironment:")
    log(f"  Python:      {sys.version.split()[0]}")
    log(f"  Torch:       {torch.__version__}")
    log(f"  Device:      {DEVICE}")
    log(f"  CPU threads: {CPU_THREADS}")
    log(f"  DEV rows:    {dev_rows}")
    log("  CV:          disabled")
    if dev_rows > 1000:
        log(
            "[warn] TabPFN-3 local CPU inference is intended mainly for small "
            "datasets. This script will nevertheless use the full cohort once "
            "per regression target with n_estimators=1."
        )


def main() -> None:
    args = parse_args()
    start = time.time()

    # Distinctive startup banner: confirms that PyCharm is running this exact
    # CPU-only build rather than an older GPU-protected copy.
    log("=" * 76)
    log(f"[build]  {SCRIPT_BUILD}")
    log(f"[script] {Path(__file__).resolve()}")
    log("[mode]   CPU only; >1000 rows explicitly allowed; cross-validation disabled")
    log(f"[env]    TABPFN_ALLOW_CPU_LARGE_DATASET={os.environ.get('TABPFN_ALLOW_CPU_LARGE_DATASET')}")
    log("=" * 76)

    dev_path = args.dev_path.expanduser().resolve()
    opu_path = args.opu_path.expanduser().resolve()
    if not dev_path.exists():
        raise FileNotFoundError(f"Development cohort not found: {dev_path}")
    if not opu_path.exists():
        raise FileNotFoundError(f"OPU table not found: {opu_path}")

    log("Reading Excel files ...")
    dev_original = pd.read_excel(dev_path)
    opu_original = pd.read_excel(opu_path)
    log(f"  DEV rows={len(dev_original)}, columns={len(dev_original.columns)}")
    log(f"  OPU rows={len(opu_original)}, columns={len(opu_original.columns)}")

    if len(opu_original) == 0:
        raise RuntimeError("OPU table contains no rows.")

    validate_environment(len(dev_original))
    dev, _ = canonicalize(dev_original, DEV_COLUMN_OVERRIDES, "DEV")
    opu, _ = canonicalize(opu_original, OPU_COLUMN_OVERRIDES, "OPU")

    if "patient_id" not in opu.columns:
        opu["patient_id"] = pd.Series(
            [f"Cycle {i + 1}" for i in range(len(opu))], dtype="string"
        )

    # Hard-disabled in this build. No command-line flag can enable CV.
    run_cv: bool = False
    if run_cv:
        raise AssertionError("Cross-validation must remain disabled in CPU-only build.")
    results: list[TaskResult] = []
    for task in TASKS:
        result = run_task(task, dev, opu, dev_path, run_cv)
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError(
            "No regression task could be run. Check DEV/OPU column mappings in the "
            "configuration section."
        )

    log("\nAssembling output tables ...")
    predictions = append_prediction_columns(opu_original, results)
    metrics = metrics_dataframe(results)
    atomic_write_csv(args.output_csv, predictions)
    atomic_write_excel(
        args.output_xlsx,
        {"Predictions": predictions, "Model summary": metrics},
    )

    log("Building HTML report ...")
    payload = build_report_payload(dev, opu, results, dev_path, opu_path, run_cv)
    report_html = build_html(payload)
    atomic_write_text(args.output_html, report_html)

    elapsed = time.time() - start
    log("\n" + "=" * 76)
    log(f"[OK] HTML:  {args.output_html.resolve()}")
    log(f"[OK] Excel: {args.output_xlsx.resolve()}")
    log(f"[OK] CSV:   {args.output_csv.resolve()}")
    log(f"Done in {elapsed / 60:.1f} min")
    log(
        "Licence reminder: verify the TabPFN-3 model licence before any "
        "production or commercial clinical use."
    )


if __name__ == "__main__":
    main()
