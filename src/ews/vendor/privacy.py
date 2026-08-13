"""Local pseudonymisation boundary for Claude agent inputs."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ._io import atomic_write_csv, atomic_write_text


PHI_COLUMNS = {
    "patient_name", "patient full name", "patient_id", "id", "_key", "_join",
    "dob", "dob 2", "husband full name", "id 2", "record_id", "full_name",
}
_PHI_NAME_RE = re.compile(r"(^|[_\s])(name|dob|passport|phone|email|address|record[_\s]?id)($|[_\s])", re.I)


def _is_phi_column(name: object) -> bool:
    normalized = str(name).strip().lower()
    if normalized == "cycle_uid":
        return False
    identifier_like = normalized == "id" or normalized.endswith("_id") or normalized.startswith("id_")
    return normalized in PHI_COLUMNS or identifier_like or bool(_PHI_NAME_RE.search(normalized))


def load_or_create_secret(path: str | Path, configured: str | None = None) -> bytes:
    """Load a stable local HMAC key; create it atomically when absent."""
    if configured:
        return configured.encode("utf-8")
    env = os.getenv("EMBRYO_CYCLE_UID_SECRET")
    if env:
        return env.encode("utf-8")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        atomic_write_text(secrets.token_hex(32), p, encoding="ascii")
        try:
            p.chmod(0o600)
        except OSError:
            pass
    return p.read_text(encoding="ascii").strip().encode("ascii")


def make_cycle_uid(value: object, secret: bytes) -> str:
    if pd.isna(value) or not str(value).strip():
        raise ValueError("cycle_uid cannot be made from an empty identifier")
    normalized = str(value).strip().upper()
    digest = hmac.new(secret, normalized.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"CYC-{digest[:16].upper()}"


def add_cycle_uid(df: pd.DataFrame, key_col: str, secret: bytes) -> pd.DataFrame:
    out = df.copy()
    if key_col not in out.columns:
        raise KeyError(f"missing cycle_uid source column: {key_col}")
    out["cycle_uid"] = out[key_col].map(lambda x: make_cycle_uid(x, secret))
    if out["cycle_uid"].duplicated().any():
        # Сообщение локальное (консоль оператора, не агентский пакет), поэтому в
        # нём можно назвать сам ключ: без него владелец не найдёт строку в МИС и
        # не поймёт, что чинить. Дедупликация здесь СОЗНАТЕЛЬНО не делается —
        # какая из совпавших строк верна, решает владелец, а не код.
        duplicated = out.loc[out["cycle_uid"].duplicated(keep=False), key_col]
        examples = sorted({str(v) for v in duplicated})[:5]
        raise ValueError(
            f"cycle_uid is not unique; source identifier is duplicated: "
            f"{len(duplicated)} строк с повторяющимся {key_col} "
            f"(примеры: {', '.join(examples)})"
        )
    return out


def build_crosswalk(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if "cycle_uid" not in df.columns or key_col not in df.columns:
        raise KeyError("crosswalk requires cycle_uid and the local source identifier")
    cols = ["cycle_uid", key_col]
    cols += [c for c in ("Patient Full Name", "patient_name") if c in df.columns]
    return df[cols].drop_duplicates("cycle_uid").copy()


def deidentify_records(df: pd.DataFrame, allowed_columns: list[str]) -> pd.DataFrame:
    """Return only explicitly allowed clinical fields; the sole identifier is cycle_uid."""
    if "cycle_uid" not in df.columns:
        raise ValueError("agent records require cycle_uid")
    unsafe_requested = [c for c in allowed_columns if _is_phi_column(c)]
    if unsafe_requested:
        raise ValueError(f"agent allow-list contains PHI: {sorted(unsafe_requested)}")
    safe = [c for c in allowed_columns if c in df.columns and c != "cycle_uid"]
    return df[["cycle_uid", *safe]].copy()


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(value) for k, value in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(value) for value in v]
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float):
        return None if np.isnan(v) else v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timedelta,)):
        return str(v)
    if isinstance(v, Mapping):
        return {str(k): _jsonable(value) for k, value in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(value) for value in v]
    return v


def _assert_agent_safe(df: pd.DataFrame, cohort: dict) -> None:
    if "cycle_uid" not in df.columns:
        raise ValueError("agent bundle must contain cycle_uid")
    forbidden = [str(c) for c in df.columns if _is_phi_column(c)]
    if forbidden:
        raise ValueError(f"agent bundle contains PHI columns: {sorted(forbidden)}")
    # Cohort metadata may contain counts/settings, never patient identifiers or names.
    bad_meta = [str(k) for k in cohort if _is_phi_column(k)]
    if bad_meta:
        raise ValueError(f"agent cohort metadata contains PHI fields: {sorted(bad_meta)}")


def write_agent_bundle(df: pd.DataFrame, cohort: dict, out_json: str | Path,
                       crosswalk: pd.DataFrame | None = None,
                       crosswalk_path: str | Path | None = None) -> Path:
    """Atomically write a pseudonymised packet and, separately, the local crosswalk."""
    _assert_agent_safe(df, cohort)
    payload = {
        "schema_version": "1.1",
        "cohort": {k: _jsonable(v) for k, v in cohort.items()},
        "patients": [{k: _jsonable(v) for k, v in row.items()}
                     for row in df.to_dict("records")],
    }
    target = atomic_write_text(json.dumps(payload, ensure_ascii=False, indent=2), out_json)
    if (crosswalk is None) != (crosswalk_path is None):
        raise ValueError("crosswalk and crosswalk_path must be supplied together")
    if crosswalk is not None:
        if "cycle_uid" not in crosswalk.columns:
            raise ValueError("local crosswalk requires cycle_uid")
        atomic_write_csv(crosswalk, crosswalk_path)
    return target
