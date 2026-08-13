"""Forensic analysis of consumable lot changes (Airtable) against cultivation windows.

Mirrors ``incubator.py``'s cultivation-window forensics: instead of a sensor CSV, the
"log" here is the per-item lot-swap timeline pulled from Airtable's «Barcode Changes
History» table. A lot change landing inside a patient's cultivation window
(``Date OPU .. +N days``) is surfaced as a candidate confounder for that cycle, never
as a cause — the same framing discipline the MIRI forensic applies to incubator
episodes.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

try:
    import requests
    _HAVE_REQUESTS = True
except Exception:  # pragma: no cover - exercised only when requests is absent
    _HAVE_REQUESTS = False


API_ROOT = "https://api.airtable.com/v0"
EVENT_COLUMNS = [
    "cycle_uid", "item_name", "attribute", "manufacturer", "event_type",
    "lot_before", "lot_after", "change_date", "window_start", "window_end",
]
TIMELINE_COLUMNS = [
    "item_record_id", "item_name", "attribute", "manufacturer",
    "lot", "period_start", "period_end",
]


def cultivation_window(date_opu, days: int | float = 6):
    """Return the inclusive cultivation window ``Date OPU .. +days``."""
    start = pd.to_datetime(date_opu)
    return start, start + timedelta(days=days)


def fetch_table(base_id: str, table_id: str, token: str, page_size: int = 100) -> pd.DataFrame:
    """Fetch every record of one Airtable table via paginated REST calls."""
    if not _HAVE_REQUESTS:
        raise RuntimeError("пакет requests не установлен: добавьте requests>=2.31 в окружение")
    if not token:
        raise ValueError(
            "нет Airtable token: заполните consumable_lots.airtable.token в config/settings.local.yaml"
        )
    headers = {"Authorization": f"Bearer {token}"}
    records: list[dict] = []
    offset = None
    while True:
        params = {"pageSize": page_size}
        if offset:
            params["offset"] = offset
        response = requests.get(f"{API_ROOT}/{base_id}/{table_id}", headers=headers,
                                params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        for record in payload.get("records", []):
            row = {"record_id": record["id"]}
            row.update(record.get("fields", {}))
            records.append(row)
        offset = payload.get("offset")
        if not offset:
            break
    return pd.DataFrame(records)


def fetch_consumable_lots(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull the items table and the Barcode Changes History table for this run."""
    at_cfg = cfg["consumable_lots"]["airtable"]
    token = at_cfg.get("token") or ""
    base_id = at_cfg["base_id"]
    items = fetch_table(base_id, at_cfg["items_table_id"], token)
    changes = fetch_table(base_id, at_cfg["changes_table_id"], token)
    return items, changes


def _first_link_id(value) -> str | None:
    if isinstance(value, list) and value:
        first = value[0]
        return first.get("id") if isinstance(first, dict) else first
    return None


def _select_name(value) -> str | None:
    """Normalize an Airtable select/multi-select/link cell to a plain string."""
    if isinstance(value, list):
        names = [_select_name(v) for v in value]
        joined = ", ".join(n for n in names if n)
        return joined or None
    if isinstance(value, dict):
        return value.get("name")
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    return str(value)


def build_lot_timeline(items: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """One row per item per lot period, derived from Barcode Changes History.

    ``period_end`` is the next recorded change for that item (``NaT`` for the still-
    active lot). Items with no linked change row are absent from the timeline —
    there is nothing dated to correlate against a cultivation window.
    """
    if changes is None or changes.empty:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)

    items_by_id = {}
    if items is not None and not items.empty and "record_id" in items.columns:
        for _, row in items.iterrows():
            items_by_id[row["record_id"]] = row

    link_col = next((c for c in changes.columns if "Link" in c), None)
    if link_col is None:
        raise KeyError("Barcode Changes History: нет колонки-ссылки на расходные материалы")

    rows = []
    for _, row in changes.iterrows():
        item_id = _first_link_id(row.get(link_col))
        change_date = pd.to_datetime(row.get("Change Date"), errors="coerce")
        if item_id is None or pd.isna(change_date):
            continue
        item_row = items_by_id.get(item_id, {})
        rows.append({
            "item_record_id": item_id,
            "item_name": item_row.get("Наименование") or row.get("Record"),
            "attribute": _select_name(item_row.get("атрибут") if hasattr(item_row, "get") else None),
            "manufacturer": _select_name(item_row.get("Производитель") if hasattr(item_row, "get") else None),
            "lot": row.get("All Barcodes") or row.get("Record"),
            "change_date": change_date,
        })
    if not rows:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)

    frame = pd.DataFrame(rows).sort_values(["item_record_id", "change_date"])
    frame["period_start"] = frame["change_date"]
    frame["period_end"] = frame.groupby("item_record_id")["change_date"].shift(-1)
    return frame.drop(columns="change_date").reset_index(drop=True)


def _lot_row(cycle_uid, meta, event_type, lot_before, lot_after, change_date, start, end) -> dict:
    return {
        "cycle_uid": cycle_uid,
        "item_name": meta["item_name"],
        "attribute": meta["attribute"],
        "manufacturer": meta["manufacturer"],
        "event_type": event_type,
        "lot_before": lot_before,
        "lot_after": lot_after,
        "change_date": change_date,
        "window_start": start,
        "window_end": end,
    }


def analyze_window_events(
    timeline: pd.DataFrame,
    date_opu,
    days: int | float = 6,
    cycle_uid: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Which lot(s) were active per item during one cultivation window.

    An item with exactly one lot period overlapping the window is ``stable``. An
    item with more than one overlapping period had its lot swapped mid-window —
    one ``lot_change`` row is emitted per swap that falls after the window opened.
    """
    start, end = cultivation_window(date_opu, days)

    if timeline is None or timeline.empty:
        summary = {
            "cycle_uid": cycle_uid, "window_start": start, "window_end": end,
            "n_items_tracked": 0, "n_items_changed": 0, "changed_items": [],
            "lot_change_flag": False, "note": "нет данных о лотах",
        }
        return summary, pd.DataFrame(columns=EVENT_COLUMNS)

    overlap = timeline[
        (timeline["period_start"] <= end)
        & (timeline["period_end"].isna() | (timeline["period_end"] > start))
    ].sort_values(["item_record_id", "period_start"])

    rows: list[dict] = []
    changed_items: list[str] = []
    for _, group in overlap.groupby("item_record_id", sort=False):
        group = group.reset_index(drop=True)
        meta = group.iloc[0]
        if len(group) == 1:
            lot = group.loc[0, "lot"]
            rows.append(_lot_row(cycle_uid, meta, "stable", lot, lot, None, start, end))
            continue
        changed_items.append(str(meta["item_name"]))
        for pos in range(len(group)):
            change_at = group.loc[pos, "period_start"]
            if change_at <= start:
                continue  # this period was already the active lot when the window opened
            lot_before = group.loc[pos - 1, "lot"] if pos > 0 else None
            lot_after = group.loc[pos, "lot"]
            rows.append(_lot_row(cycle_uid, meta, "lot_change", lot_before, lot_after,
                                 change_at, start, end))

    events = pd.DataFrame(rows, columns=EVENT_COLUMNS) if rows else pd.DataFrame(columns=EVENT_COLUMNS)
    summary = {
        "cycle_uid": cycle_uid, "window_start": start, "window_end": end,
        "n_items_tracked": int(overlap["item_record_id"].nunique()),
        "n_items_changed": len(changed_items), "changed_items": changed_items,
        "lot_change_flag": bool(changed_items),
    }
    return summary, events


def analyze_patients(
    patients: pd.DataFrame,
    timeline: pd.DataFrame,
    date_col: str = "Date OPU",
    days: int | float = 6,
    cycle_uid_col: str = "cycle_uid",
) -> pd.DataFrame:
    """Analyze cultivation windows against the lot timeline, keyed by ``cycle_uid``.

    Direct patient IDs are intentionally neither accepted as a fallback key nor
    copied to the result, mirroring ``incubator.analyze_patients``. Event-level
    output is attached as ``result.attrs["events"]``.
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
                "cycle_uid": cycle_uid, "n_items_tracked": 0, "n_items_changed": 0,
                "changed_items": [], "lot_change_flag": False, "error": f"нет {date_col}",
            }
            events = pd.DataFrame(columns=EVENT_COLUMNS)
        else:
            summary, events = analyze_window_events(
                timeline, date_opu, days=days, cycle_uid=str(cycle_uid))
        summaries.append(summary)
        if not events.empty:
            event_frames.append(events)

    result = pd.DataFrame(summaries)
    all_events = (
        pd.concat(event_frames, ignore_index=True) if event_frames
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )
    result.attrs["events"] = all_events
    return result


def compare_flagged_vs_baseline(summary: pd.DataFrame, flagged_cycle_uids) -> dict:
    """Compare the lot-change rate between flagged cycles and the rest of the cohort.

    ``summary`` must come from :func:`analyze_patients` run over the FULL dated
    cohort, not just the flagged subset, mirroring
    ``incubator.compare_flagged_vs_baseline``. Without this, "a lot changed
    inside the cultivation window" can't be told apart from ordinary cohort-wide
    base-rate noise. A cycle with no resolvable timeline data at all (``error``
    set, e.g. missing Date OPU, or ``note`` "нет данных о лотах" when the whole
    Airtable timeline came back empty) is excluded from both groups'
    denominators rather than silently counted as "no lot change".
    """
    if summary is None or summary.empty or "cycle_uid" not in summary.columns:
        return {"flagged": None, "baseline": None, "note": "нет данных"}
    d = summary.copy()
    flagged_set = {str(x) for x in flagged_cycle_uids}
    is_flagged = d["cycle_uid"].astype(str).isin(flagged_set)
    no_error = d["error"].isna() if "error" in d.columns else pd.Series(True, index=d.index)
    no_missing_note = (d["note"].astype("string") != "нет данных о лотах") | d["note"].isna() \
        if "note" in d.columns else pd.Series(True, index=d.index)
    covered = no_error & no_missing_note

    def _group(mask: pd.Series) -> dict:
        excluded = int((mask & ~covered).sum())
        g = d.loc[mask & covered]
        if g.empty:
            return {"n": 0, "n_excluded_no_coverage": excluded, "lot_change_rate": None}
        flag = g["lot_change_flag"].fillna(False).astype(bool) if "lot_change_flag" in g else pd.Series(False, index=g.index)
        return {"n": int(len(g)), "n_excluded_no_coverage": excluded,
                "lot_change_rate": float(flag.mean())}

    return {"flagged": _group(is_flagged), "baseline": _group(~is_flagged)}
