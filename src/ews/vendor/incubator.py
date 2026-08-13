"""Forensic analysis of MIRI 6391 incubator logs.

The public :func:`analyze_patients` function returns one compact row per
``cycle_uid``.  The event-level table is available in
``result.attrs["events"]`` and can also be produced directly with
:func:`analyze_window_events`.

MIRI 6391 files are semicolon-separated and normally contain one row every
30 seconds.  Events are contiguous *episodes*, not counts of individual log
rows.  Missing stretches of the time series are represented as ``gap``
events and therefore cannot silently turn into reassuring "no alarm" data.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


TEMP_COLS = [f"Temp{i}" for i in range(1, 13)]
ALARM_COLS = (
    [f"Chamber{i}TemperatureAlarm" for i in range(1, 7)]
    + [
        "Co2ConcentrationAlarm",
        "O2ConcentrationAlarm",
        "Co2PressureAlarm",
        "N2PressureAlarm",
    ]
)
TOL = {"temp": 0.5, "co2": 0.3, "o2": 0.5}
DEFAULT_SETPOINTS = {"temp": 37.0, "co2": 6.5, "o2": 5.0}
DEFAULT_SEVERITY = {
    "critical_deviation_multiplier": 2.0,
    "critical_duration_minutes": 15.0,
    "critical_lid_minutes": 5.0,
    "critical_gap_minutes": 10.0,
    "critical_recovery_minutes": 15.0,
    # Below this, a log gap is a lost ability to WRITE logs (e.g. a power outage) —
    # a real adverse event, not "missing data". At/above it, logging or the
    # incubator itself was explicitly stopped. Owner-fixed rule (CLAUDE.md).
    "gap_reason_threshold_minutes": 480.0,
}
EVENT_COLUMNS = [
    "cycle_uid",
    "event_type",
    "sensor",
    "start",
    "end",
    "duration_minutes",
    "max_deviation",
    "severity",
    "recovery_time_minutes",
    "gap_reason",
]

GAP_REASON_POWER_OUTAGE = "power_or_logging_outage"
GAP_REASON_STOPPED = "incubator_or_logging_stopped"


def _gap_reason(duration_minutes: float, severity_cfg: dict) -> str:
    threshold = severity_cfg.get("gap_reason_threshold_minutes",
                                 DEFAULT_SEVERITY["gap_reason_threshold_minutes"])
    return GAP_REASON_STOPPED if duration_minutes >= threshold else GAP_REASON_POWER_OUTAGE
_FILE_DATE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})")


def cultivation_window(date_opu, days: int | float = 6):
    """Return the inclusive cultivation window ``Date OPU .. +days``."""
    start = pd.to_datetime(date_opu)
    return start, start + timedelta(days=days)


def _file_date(name: str):
    m = _FILE_DATE.search(name)
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def find_log_files(
    base_dir: str,
    incubator: str | int = "6391",
    start=None,
    end=None,
) -> list[Path]:
    """Find MIRI CSV files, restricting by filename date when possible."""
    root = Path(base_dir) / str(incubator)
    if not root.exists():
        root = Path(base_dir)
    files = sorted(list(root.glob("**/Data/*.csv")) or list(root.glob("**/*.csv")))
    if start is None or end is None:
        return files

    start_date, end_date = pd.to_datetime(start).date(), pd.to_datetime(end).date()
    selected = []
    for file in files:
        file_date = _file_date(file.name)
        if file_date is None or start_date <= file_date <= end_date:
            selected.append(file)
    return selected


def load_files(
    files: Iterable[str | Path],
    sep: str = ";",
    dt_fmt: str = "%Y.%m.%d %H:%M:%S",
) -> pd.DataFrame:
    """Load known MIRI CSVs and retain a clean chronological time series."""
    frames = []
    for file in files:
        try:
            frame = pd.read_csv(file, sep=sep)
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue
        if "Date" not in frame.columns:
            continue
        frame["Date"] = pd.to_datetime(frame["Date"], format=dt_fmt, errors="coerce")
        frames.append(frame.dropna(subset=["Date"]))
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def _numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _boolean(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index, dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    return numeric.fillna(0).ne(0) | text.isin({"true", "yes", "да", "alarm", "on"})


def _we_rule1(series: pd.Series, k: float = 3.0) -> int:
    """Preserve the previous Western Electric rule-1 diagnostic."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 5:
        return 0
    mean, standard_deviation = values.mean(), values.std(ddof=0)
    if standard_deviation == 0:
        return 0
    return int(((values - mean).abs() > k * standard_deviation).sum())


def _setpoint_series(
    window: pd.DataFrame,
    key: str,
    column: str,
    setpoints: dict,
) -> pd.Series:
    configured = setpoints.get(key)
    if configured is not None:
        return pd.Series(float(configured), index=window.index, dtype=float)
    measured = _numeric(window.get(column), window.index)
    fallback = measured.median() if measured.notna().any() else DEFAULT_SETPOINTS[key]
    return measured.fillna(float(fallback))


def _severity(
    event_type: str,
    duration: float,
    max_deviation: float | None,
    tolerance: float | None,
    severity_cfg: dict,
    recovery: float | None = None,
) -> str:
    if event_type == "gap":
        critical = duration >= severity_cfg["critical_gap_minutes"]
    elif event_type == "lid_open":
        critical = duration >= severity_cfg["critical_lid_minutes"]
        if recovery is not None:
            critical |= recovery >= severity_cfg["critical_recovery_minutes"]
    elif event_type == "alarm":
        critical = duration >= severity_cfg["critical_duration_minutes"]
    else:
        critical = duration >= severity_cfg["critical_duration_minutes"]
        if tolerance and max_deviation is not None:
            critical |= max_deviation >= tolerance * severity_cfg["critical_deviation_multiplier"]
        if recovery is not None:
            critical |= recovery >= severity_cfg["critical_recovery_minutes"]
    return "critical" if critical else "warning"


def _runs(mask: pd.Series, times: pd.Series, break_seconds: float):
    """Yield inclusive positional bounds of true runs, split across log gaps."""
    truth = mask.fillna(False).to_numpy(dtype=bool)
    ts = pd.to_datetime(times).reset_index(drop=True)
    start = None
    for pos, active in enumerate(truth):
        temporal_break = pos > 0 and (ts.iloc[pos] - ts.iloc[pos - 1]).total_seconds() > break_seconds
        if start is not None and (not active or temporal_break):
            yield start, pos - 1
            start = None
        if active and start is None:
            start = pos
    if start is not None:
        yield start, len(truth) - 1


def _episode_rows(
    window: pd.DataFrame,
    mask: pd.Series,
    event_type: str,
    sensor: str,
    cadence_seconds: float,
    gap_seconds: float,
    severity_cfg: dict,
    deviations: pd.Series | None = None,
    tolerance: float | None = None,
    cycle_uid: str | None = None,
) -> list[dict]:
    rows = []
    times = window["Date"].reset_index(drop=True)
    values = deviations.reset_index(drop=True) if deviations is not None else None
    normalized_mask = mask.reset_index(drop=True)
    for first, last in _runs(normalized_mask, times, gap_seconds):
        start, end = times.iloc[first], times.iloc[last]
        duration = max(cadence_seconds, (end - start).total_seconds() + cadence_seconds) / 60.0
        max_dev = None
        if values is not None:
            segment = pd.to_numeric(values.iloc[first : last + 1], errors="coerce")
            if segment.notna().any():
                max_dev = float(segment.abs().max())

        # Recovery is only known if the next sample is timely and clears the condition.
        recovery = None
        next_pos = last + 1
        if next_pos < len(times):
            delta = (times.iloc[next_pos] - end).total_seconds()
            if delta <= gap_seconds and not bool(normalized_mask.iloc[next_pos]):
                recovery = max(0.0, delta / 60.0)

        rows.append(
            {
                "cycle_uid": cycle_uid,
                "event_type": event_type,
                "sensor": sensor,
                "start": start,
                "end": end,
                "duration_minutes": round(duration, 3),
                "max_deviation": None if max_dev is None else round(max_dev, 4),
                "severity": _severity(
                    event_type,
                    duration,
                    max_dev,
                    tolerance,
                    severity_cfg,
                    recovery,
                ),
                "recovery_time_minutes": None if recovery is None else round(recovery, 3),
                "gap_reason": None,
            }
        )
    return rows


def _gap_rows(
    window: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    cadence_seconds: float,
    gap_seconds: float,
    severity_cfg: dict,
    cycle_uid: str | None,
) -> list[dict]:
    if window.empty:
        return []
    times = window["Date"].sort_values().reset_index(drop=True)
    candidates: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []

    leading = (times.iloc[0] - window_start).total_seconds()
    if leading > gap_seconds:
        candidates.append((window_start, times.iloc[0], leading))
    for previous, current in zip(times.iloc[:-1], times.iloc[1:]):
        delta = (current - previous).total_seconds()
        if delta > gap_seconds:
            missing = max(0.0, delta - cadence_seconds)
            candidates.append((previous + timedelta(seconds=cadence_seconds), current, missing))
    trailing_start = times.iloc[-1] + timedelta(seconds=cadence_seconds)
    trailing = (window_end - trailing_start).total_seconds()
    if trailing > gap_seconds:
        candidates.append((trailing_start, window_end, trailing))

    rows = []
    for start, end, seconds in candidates:
        duration = seconds / 60.0
        rows.append(
            {
                "cycle_uid": cycle_uid,
                "event_type": "gap",
                "sensor": "Date",
                "start": start,
                "end": end,
                "duration_minutes": round(duration, 3),
                "max_deviation": None,
                "severity": _severity("gap", duration, None, None, severity_cfg),
                "recovery_time_minutes": None,
                "gap_reason": _gap_reason(duration, severity_cfg),
            }
        )
    return rows


def _complete_gap(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    severity_cfg: dict,
    cycle_uid: str | None,
) -> pd.DataFrame:
    """Represent a completely absent cultivation log as a real gap event."""
    duration = max(0.0, (window_end - window_start).total_seconds() / 60.0)
    if duration == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    row = {
        "cycle_uid": cycle_uid,
        "event_type": "gap",
        "sensor": "Date",
        "start": window_start,
        "end": window_end,
        "duration_minutes": round(duration, 3),
        "max_deviation": None,
        "severity": _severity("gap", duration, None, None, severity_cfg),
        "recovery_time_minutes": None,
        "gap_reason": _gap_reason(duration, severity_cfg),
    }
    return pd.DataFrame([row], columns=EVENT_COLUMNS)


def _lid_recovery(
    events: list[dict],
    window: pd.DataFrame,
    temp_normal: pd.Series,
    gap_seconds: float,
    severity_cfg: dict,
) -> None:
    """Measure temperature recovery from the first closed-lid sample."""
    times = window["Date"].reset_index(drop=True)
    normal = temp_normal.reset_index(drop=True)
    for event in events:
        if event["event_type"] != "lid_open":
            continue
        last_open = pd.Timestamp(event["end"])
        positions = np.flatnonzero((times > last_open).to_numpy())
        if not len(positions):
            continue
        close_pos = int(positions[0])
        if (times.iloc[close_pos] - last_open).total_seconds() > gap_seconds:
            continue
        recovered = None
        for pos in range(close_pos, len(times)):
            if pos > close_pos and (times.iloc[pos] - times.iloc[pos - 1]).total_seconds() > gap_seconds:
                break
            if bool(normal.iloc[pos]):
                recovered = (times.iloc[pos] - times.iloc[close_pos]).total_seconds() / 60.0
                break
        event["recovery_time_minutes"] = None if recovered is None else round(recovered, 3)
        event["severity"] = _severity(
            "lid_open",
            float(event["duration_minutes"]),
            None,
            None,
            severity_cfg,
            recovered,
        )


def _legacy_diagnostics(window: pd.DataFrame, setpoint_values: dict, tolerances: dict) -> dict:
    temp_cols = [column for column in TEMP_COLS if column in window]
    temp_deviation = pd.DataFrame(index=window.index)
    if temp_cols:
        temp_deviation = window[temp_cols].apply(pd.to_numeric, errors="coerce").sub(
            setpoint_values["temp"], axis=0
        ).abs()
    array = temp_deviation.to_numpy(dtype=float)
    temp_max = float(np.nanmax(array)) if array.size and np.isfinite(array).any() else None
    co2 = _numeric(window.get("Co2Concentration"), window.index)
    o2 = _numeric(window.get("O2Concentration"), window.index)
    alarms = {column: int(_boolean(window.get(column), window.index).sum()) for column in ALARM_COLS if column in window}
    western_electric, drift = {}, {}
    for chamber in range(1, 7):
        columns = [
            column
            for column in (f"Temp{2 * chamber - 1}", f"Temp{2 * chamber}")
            if column in window
        ]
        if not columns:
            continue
        chamber_temp = window[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        western_electric[f"chamber{chamber}"] = _we_rule1(chamber_temp)
        values = chamber_temp.to_numpy(dtype=float)
        positions = np.arange(len(chamber_temp))
        valid = np.isfinite(values)
        if valid.sum() >= 3:
            drift[f"chamber{chamber}_slope"] = round(
                float(np.polyfit(positions[valid], values[valid], 1)[0]), 6
            )
    return {
        "excursions": {
            "temp_excursions": int(np.nansum(array > tolerances["temp"])) if array.size else 0,
            "temp_max_dev": None if temp_max is None else round(temp_max, 3),
            "co2_excursions": int(((co2 - setpoint_values["co2"]).abs() > tolerances["co2"]).sum()),
            "o2_excursions": int(((o2 - setpoint_values["o2"]).abs() > tolerances["o2"]).sum()),
        },
        "alarms_activations": {key: value for key, value in alarms.items() if value} or "нет",
        "lid_open_events": 0,  # overwritten from the episode table
        "western_electric_rule1": western_electric,
        "temp_drift_slope": drift,
    }


def analyze_window_events(
    logs: pd.DataFrame,
    date_opu,
    days: int | float = 6,
    setpoints: dict | None = None,
    tolerances: dict | None = None,
    cadence_seconds: float = 30.0,
    gap_factor: float = 3.0,
    severity_thresholds: dict | None = None,
    cycle_uid: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Return a window summary and an event-level table.

    An excursion event begins only when a measured value exceeds its configured
    tolerance.  Consequently, the summary warning flag cannot be raised merely
    because a patient was selected for review.
    """
    start, end = cultivation_window(date_opu, days)
    tolerance_cfg = {**TOL, **(tolerances or {})}
    severity_cfg = {**DEFAULT_SEVERITY, **(severity_thresholds or {})}
    gap_seconds = float(cadence_seconds) * float(gap_factor)

    if logs.empty or "Date" not in logs.columns:
        events = _complete_gap(start, end, severity_cfg, cycle_uid)
        warning_count = int(events["severity"].eq("warning").sum()) if not events.empty else 0
        critical_count = int(events["severity"].eq("critical").sum()) if not events.empty else 0
        summary = {
            "cycle_uid": cycle_uid,
            "window_start": start,
            "window_end": end,
            "window": [str(start), str(end)],
            "n_points": 0,
            "event_count": int(len(events)),
            "warning_count": warning_count,
            "critical_count": critical_count,
            "max_severity": "critical" if critical_count else ("warning" if warning_count else "none"),
            "warning": bool(warning_count or critical_count),
            "gap_count": int(len(events)),
            "max_event_duration_minutes": (
                float(events["duration_minutes"].max()) if not events.empty else 0.0
            ),
            "error": "нет логов",
        }
        return summary, events

    prepared = logs.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    prepared = prepared.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    window = prepared[(prepared["Date"] >= start) & (prepared["Date"] <= end)].reset_index(drop=True)
    if window.empty:
        events = _complete_gap(start, end, severity_cfg, cycle_uid)
        warning_count = int(events["severity"].eq("warning").sum()) if not events.empty else 0
        critical_count = int(events["severity"].eq("critical").sum()) if not events.empty else 0
        summary = {
            "cycle_uid": cycle_uid,
            "window_start": start,
            "window_end": end,
            "window": [str(start), str(end)],
            "n_points": 0,
            "event_count": int(len(events)),
            "warning_count": warning_count,
            "critical_count": critical_count,
            "max_severity": "critical" if critical_count else ("warning" if warning_count else "none"),
            "warning": bool(warning_count or critical_count),
            "gap_count": int(len(events)),
            "max_event_duration_minutes": (
                float(events["duration_minutes"].max()) if not events.empty else 0.0
            ),
            "note": "нет записей лога в окне культивирования",
        }
        return summary, events

    setpoint_cfg = setpoints or {}
    temp_sp = _setpoint_series(window, "temp", "TempSetPoint", setpoint_cfg)
    co2_sp = _setpoint_series(window, "co2", "Co2ConcentrationSetPoint", setpoint_cfg)
    o2_sp = _setpoint_series(window, "o2", "O2ConcentrationSetPoint", setpoint_cfg)
    setpoint_values = {"temp": temp_sp, "co2": co2_sp, "o2": o2_sp}
    rows: list[dict] = []

    temp_normal_parts = []
    for column in [name for name in TEMP_COLS if name in window]:
        deviation = (_numeric(window[column], window.index) - temp_sp).abs()
        valid = deviation.notna()
        active = valid & deviation.gt(tolerance_cfg["temp"])
        temp_normal_parts.append(valid & ~active)
        rows.extend(
            _episode_rows(
                window,
                active,
                "temperature_excursion",
                column,
                cadence_seconds,
                gap_seconds,
                severity_cfg,
                deviation,
                tolerance_cfg["temp"],
                cycle_uid,
            )
        )

    for key, column, sp in (
        ("co2", "Co2Concentration", co2_sp),
        ("o2", "O2Concentration", o2_sp),
    ):
        if column not in window:
            continue
        deviation = (_numeric(window[column], window.index) - sp).abs()
        active = deviation.notna() & deviation.gt(tolerance_cfg[key])
        rows.extend(
            _episode_rows(
                window,
                active,
                f"{key}_excursion",
                column,
                cadence_seconds,
                gap_seconds,
                severity_cfg,
                deviation,
                tolerance_cfg[key],
                cycle_uid,
            )
        )

    for column in [name for name in ALARM_COLS if name in window]:
        active = _boolean(window[column], window.index)
        alarm_rows = _episode_rows(
            window,
            active,
            "alarm",
            column,
            cadence_seconds,
            gap_seconds,
            severity_cfg,
            cycle_uid=cycle_uid,
        )
        # Pressure alarms represent a direct gas-supply failure signal.
        for event in alarm_rows:
            if "PressureAlarm" in column:
                event["severity"] = "critical"
        rows.extend(alarm_rows)

    lid_rows: list[dict] = []
    if "IsLidOpen" in window:
        lid_rows = _episode_rows(
            window,
            _boolean(window["IsLidOpen"], window.index),
            "lid_open",
            "IsLidOpen",
            cadence_seconds,
            gap_seconds,
            severity_cfg,
            cycle_uid=cycle_uid,
        )
        if temp_normal_parts:
            all_temp_normal = pd.concat(temp_normal_parts, axis=1).all(axis=1)
            _lid_recovery(lid_rows, window, all_temp_normal, gap_seconds, severity_cfg)
        rows.extend(lid_rows)

    rows.extend(
        _gap_rows(window, start, end, cadence_seconds, gap_seconds, severity_cfg, cycle_uid)
    )
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if not events.empty:
        events = events.sort_values(["start", "event_type", "sensor"]).reset_index(drop=True)

    warning_count = int(events["severity"].eq("warning").sum()) if not events.empty else 0
    critical_count = int(events["severity"].eq("critical").sum()) if not events.empty else 0
    diagnostic = _legacy_diagnostics(window, setpoint_values, tolerance_cfg)
    diagnostic["lid_open_events"] = int(events["event_type"].eq("lid_open").sum()) if not events.empty else 0
    summary = {
        "cycle_uid": cycle_uid,
        "window_start": start,
        "window_end": end,
        "window": [str(start), str(end)],
        "n_points": int(len(window)),
        "setpoints": {
            "temp": float(temp_sp.median()),
            "co2": float(co2_sp.median()),
            "o2": float(o2_sp.median()),
        },
        **diagnostic,
        "event_count": int(len(events)),
        "warning_count": warning_count,
        "critical_count": critical_count,
        "max_severity": "critical" if critical_count else ("warning" if warning_count else "none"),
        "warning": bool(warning_count or critical_count),
        "gap_count": int(events["event_type"].eq("gap").sum()) if not events.empty else 0,
        "max_event_duration_minutes": (
            float(events["duration_minutes"].max()) if not events.empty else 0.0
        ),
    }
    return summary, events


def analyze_window(
    logs: pd.DataFrame,
    date_opu,
    days: int | float = 6,
    setpoints: dict | None = None,
    tolerances: dict | None = None,
    cadence_seconds: float = 30.0,
    gap_factor: float = 3.0,
    severity_thresholds: dict | None = None,
) -> dict:
    """Backward-compatible summary-only wrapper around event analysis."""
    summary, _events = analyze_window_events(
        logs,
        date_opu,
        days,
        setpoints,
        tolerances,
        cadence_seconds,
        gap_factor,
        severity_thresholds,
    )
    return summary


def analyze_patients(
    patients: pd.DataFrame,
    base_dir: str,
    incubator="6391",
    date_col="Date OPU",
    days: int | float = 6,
    setpoints: dict | None = None,
    sep=";",
    dt_fmt="%Y.%m.%d %H:%M:%S",
    tolerances: dict | None = None,
    cycle_uid_col: str = "cycle_uid",
    cadence_seconds: float = 30.0,
    gap_factor: float = 3.0,
    severity_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Analyze cultivation windows and return summaries keyed by ``cycle_uid``.

    Direct patient IDs are intentionally neither accepted as fallback keys nor
    copied to the result.  The reverse join remains a local orchestrator task.
    Event-level output is attached as ``result.attrs["events"]``.
    """
    if cycle_uid_col not in patients.columns:
        raise KeyError(f"нет обязательной псевдонимизированной колонки {cycle_uid_col!r}")

    summaries = []
    event_frames = []
    for _, patient in patients.iterrows():
        cycle_uid = patient.get(cycle_uid_col)
        if pd.isna(cycle_uid):
            raise ValueError(f"пустое значение в обязательной колонке {cycle_uid_col!r}")
        date_opu = patient.get(date_col)
        if pd.isna(date_opu):
            summary = {
                "cycle_uid": cycle_uid,
                "n_points": 0,
                "event_count": 0,
                "warning_count": 0,
                "critical_count": 0,
                "max_severity": "none",
                "warning": False,
                "gap_count": 0,
                "max_event_duration_minutes": 0.0,
                "error": f"нет {date_col}",
            }
            events = pd.DataFrame(columns=EVENT_COLUMNS)
        else:
            start, end = cultivation_window(date_opu, days)
            files = find_log_files(base_dir, incubator, start, end)
            logs = load_files(files, sep, dt_fmt)
            summary, events = analyze_window_events(
                logs,
                date_opu,
                days=days,
                setpoints=setpoints,
                tolerances=tolerances,
                cadence_seconds=cadence_seconds,
                gap_factor=gap_factor,
                severity_thresholds=severity_thresholds,
                cycle_uid=str(cycle_uid),
            )
        summaries.append(summary)
        if not events.empty:
            event_frames.append(events)

    result = pd.DataFrame(summaries)
    all_events = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )
    result.attrs["events"] = all_events
    return result


def compare_flagged_vs_baseline(summary: pd.DataFrame, flagged_cycle_uids: Iterable[str]) -> dict:
    """Compare warning/critical rate between flagged cycles and the rest of the cohort.

    ``summary`` must come from :func:`analyze_patients` run over the FULL dated
    cohort, not just the flagged subset — otherwise there is no control group to
    compare against. A cycle whose window has no locally synced MIRI log at all
    (``error`` set, e.g. "нет логов") is excluded from both groups' denominators:
    a missing local download is a data-coverage gap, not an observed absence of
    incubator events, and conflating the two would silently inflate whichever
    group happens to have more unsynced weeks. A window that WAS synced but is
    empty inside it keeps counting as a real gap event, per the project's fixed
    MIRI rule that a logging gap is itself an adverse event, not missing data.
    """
    if summary is None or summary.empty or "cycle_uid" not in summary.columns:
        return {"flagged": None, "baseline": None, "note": "нет данных"}
    d = summary.copy()
    flagged_set = {str(x) for x in flagged_cycle_uids}
    is_flagged = d["cycle_uid"].astype(str).isin(flagged_set)
    covered = d["error"].isna() if "error" in d.columns else pd.Series(True, index=d.index)

    def _group(mask: pd.Series) -> dict:
        excluded = int((mask & ~covered).sum())
        g = d.loc[mask & covered]
        if g.empty:
            return {"n": 0, "n_excluded_no_coverage": excluded,
                    "warning_rate": None, "critical_rate": None}
        warning = g["warning"].fillna(False).astype(bool) if "warning" in g else pd.Series(False, index=g.index)
        critical = (pd.to_numeric(g.get("critical_count"), errors="coerce").fillna(0) > 0
                    if "critical_count" in g else pd.Series(False, index=g.index))
        return {
            "n": int(len(g)),
            "n_excluded_no_coverage": excluded,
            "warning_rate": float(warning.mean()),
            "critical_rate": float(critical.mean()),
        }

    return {"flagged": _group(is_flagged), "baseline": _group(~is_flagged)}
