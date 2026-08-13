#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""teable_common.py — shared Teable REST client and clinic registry.

Imported by teable_bootstrap.py (schema creation/upgrade), teable_sync.py
(data upload) and cards_build.py (patient-card viewer). Keeping the client and
the registry in one place means a new clinic is added by editing ONE list in
teable_config.local.json — no base id is ever hard-coded in a .bat or in the
/teable-sync command again.

Dependencies: requests (+ pandas for the callers).
"""

from __future__ import annotations

import json
import os
import sys

try:
    import requests
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("Missing dependency 'requests'. Run: pip install -r requirements.txt")


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "teable_config.local.json")
# Repo root is one level above teable/ — used to resolve relative processed dirs
# so the registry stays portable if the project folder is ever moved.
REPO_ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# Config / clinic registry
# ---------------------------------------------------------------------------

class Clinic:
    """One clinic = one Teable base + one processed folder + one card page."""

    def __init__(self, raw: dict, cfg: dict):
        self.code = raw["code"]
        self.name = raw.get("name", raw["code"])
        self.base_id = raw.get("base_id") or ""
        self.timezone = raw.get("timezone", cfg.get("timezone", "Asia/Tashkent"))
        self.dt_version = raw.get("dt_version", cfg.get("dt_version", "v7.0"))
        self.upload_files = raw.get("upload_files", cfg.get("upload_files", True))
        processed = raw.get("processed_dir") or ""
        self.processed_dir = (processed if os.path.isabs(processed)
                              else os.path.join(REPO_ROOT, processed))

    @property
    def configured(self) -> bool:
        return bool(self.base_id)

    def card_url(self, cards_base_url: str, cycle_uid: str | None = None) -> str:
        page = f"{cards_base_url.rstrip('/')}/{self.code}.html"
        return f"{page}#uid={cycle_uid}" if cycle_uid else page

    def __repr__(self) -> str:
        return f"<Clinic {self.code} base={self.base_id or '—'}>"


def load_config(path: str | None = None) -> dict:
    path = path or DEFAULT_CONFIG
    if not os.path.exists(path):
        sys.exit(f"Config not found: {path}\n"
                 f"Copy teable_config.local.json.example and fill in the token.")
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_path"] = path
    return cfg


def clinics(cfg: dict, only: str | None = None) -> list[Clinic]:
    """Clinic registry from config['clinics']; `only` filters by code (or name)."""
    raw = cfg.get("clinics")
    if not raw:
        sys.exit("No 'clinics' array in the config — see teable_config.local.json.example.")
    out = [Clinic(c, cfg) for c in raw]
    if only:
        key = only.strip().lower()
        out = [c for c in out if c.code.lower() == key or c.name.lower() == key]
        if not out:
            sys.exit(f"Clinic '{only}' not in the registry "
                     f"({', '.join(c.code for c in clinics(cfg))}).")
    return out


def save_base_id(cfg: dict, code: str, base_id: str) -> None:
    """Persist a freshly bootstrapped base id back into the config file."""
    path = cfg["_path"]
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    for entry in raw.get("clinics", []):
        if entry.get("code") == code:
            entry["base_id"] = base_id
            break
    else:
        raise KeyError(f"clinic '{code}' not present in {path}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def cards_root(cfg: dict) -> str:
    """Folder the nginx `cards` container serves (host side)."""
    raw = cfg.get("cards_dir") or "data/processed/_viewer"
    return raw if os.path.isabs(raw) else os.path.join(REPO_ROOT, raw)


def cards_base_url(cfg: dict) -> str:
    return cfg.get("cards_url", "http://127.0.0.1:8788").rstrip("/")


# ---------------------------------------------------------------------------
# Teable REST client
# ---------------------------------------------------------------------------

class Teable:
    def __init__(self, url: str, token: str, dry_run: bool = False):
        self.url = url.rstrip("/")
        self.dry_run = dry_run
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/json"})

    # -- reads ------------------------------------------------------------
    def _get(self, path, **kw):
        r = self.s.get(self.url + path, timeout=60, **kw)
        r.raise_for_status()
        return r.json()

    def spaces(self):
        return self._get("/api/space")

    def bases(self):
        return self._get("/api/base/access/all")

    def tables(self, base_id):
        return self._get(f"/api/base/{base_id}/table")

    def fields(self, table_id):
        return self._get(f"/api/table/{table_id}/field")

    def views(self, table_id):
        return self._get(f"/api/table/{table_id}/view")

    def create_view(self, table_id, body):
        r = self.s.post(f"{self.url}/api/table/{table_id}/view", json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def row_count(self, table_id) -> int:
        return int(self._get(f"/api/table/{table_id}/aggregation/row-count").get("rowCount", 0))

    def all_records(self, table_id, projection: list[str] | None = None):
        """Every record (paged), fieldKeyType=name."""
        out, skip = [], 0
        while True:
            params = {"take": 1000, "skip": skip, "fieldKeyType": "name"}
            if projection:
                params["projection"] = projection
            page = self._get(f"/api/table/{table_id}/record", params=params)
            recs = page.get("records", [])
            out.extend(recs)
            if len(recs) < 1000:
                break
            skip += 1000
        return out

    # -- writes -----------------------------------------------------------
    def create_base(self, space_id, name):
        r = self.s.post(f"{self.url}/api/base",
                        json={"spaceId": space_id, "name": name}, timeout=60)
        r.raise_for_status()
        return r.json()["id"]

    def create_table(self, base_id, body):
        r = self.s.post(f"{self.url}/api/base/{base_id}/table", json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def rename_table(self, base_id, table_id, name):
        r = self.s.put(f"{self.url}/api/base/{base_id}/table/{table_id}/name",
                       json={"name": name}, timeout=60)
        r.raise_for_status()
        return r

    def rename_field(self, table_id, field_id, name):
        r = self.s.patch(f"{self.url}/api/table/{table_id}/field/{field_id}",
                         json={"name": name}, timeout=60)
        r.raise_for_status()
        return r

    def set_column_meta(self, table_id, view_id, entries):
        """entries: [{"fieldId": ..., "columnMeta": {"order": n, ...}}, ...]"""
        r = self.s.put(f"{self.url}/api/table/{table_id}/view/{view_id}/column-meta",
                       json=entries, timeout=60)
        r.raise_for_status()
        return r

    def move_table(self, base_id, table_id, anchor_id, position="after"):
        r = self.s.put(f"{self.url}/api/base/{base_id}/table/{table_id}/order",
                       json={"anchorId": anchor_id, "position": position}, timeout=60)
        r.raise_for_status()
        return r

    def move_view(self, table_id, view_id, anchor_id, position="after"):
        r = self.s.put(f"{self.url}/api/table/{table_id}/view/{view_id}/order",
                       json={"anchorId": anchor_id, "position": position}, timeout=60)
        r.raise_for_status()
        return r

    def create_field(self, table_id, body):
        r = self.s.post(f"{self.url}/api/table/{table_id}/field", json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def create_records(self, table_id, rows: list[dict]):
        """Batch create. `rows` are plain {field name: value} dicts."""
        if self.dry_run or not rows:
            return [{"id": "DRYRUN"} for _ in rows]
        made = []
        for i in range(0, len(rows), 500):
            chunk = rows[i:i + 500]
            r = self.s.post(f"{self.url}/api/table/{table_id}/record",
                            json={"fieldKeyType": "name", "typecast": True,
                                  "records": [{"fields": f} for f in chunk]}, timeout=180)
            r.raise_for_status()
            made.extend(r.json()["records"])
        return made

    def create(self, table_id, fields):
        return self.create_records(table_id, [fields])[0]

    def update(self, table_id, record_id, fields):
        if self.dry_run:
            return {"id": record_id}
        r = self.s.patch(f"{self.url}/api/table/{table_id}/record/{record_id}",
                         json={"fieldKeyType": "name", "typecast": True,
                               "record": {"fields": fields}}, timeout=60)
        r.raise_for_status()
        return r.json()

    def update_records(self, table_id, updates: list[tuple[str, dict]]):
        """Batch update of (record_id, fields) pairs."""
        if self.dry_run or not updates:
            return
        for i in range(0, len(updates), 500):
            chunk = updates[i:i + 500]
            r = self.s.patch(f"{self.url}/api/table/{table_id}/record",
                             json={"fieldKeyType": "name", "typecast": True,
                                   "records": [{"id": rid, "fields": f} for rid, f in chunk]},
                             timeout=180)
            r.raise_for_status()

    def upload_attachment(self, table_id, record_id, field_id, filepath):
        if self.dry_run:
            return {"id": record_id}
        with open(filepath, "rb") as fh:
            r = self.s.post(
                f"{self.url}/api/table/{table_id}/record/{record_id}/{field_id}/uploadAttachment",
                files={"file": (os.path.basename(filepath), fh)}, timeout=300)
        r.raise_for_status()
        return r.json()


def connect(cfg: dict, dry_run: bool = False) -> Teable:
    return Teable(cfg.get("teable_url", "http://127.0.0.1:3000"), cfg["token"],
                  dry_run=dry_run)


def resolve_tables(tb: Teable, base_id: str) -> dict:
    """Short table key ('Runs', 'Cycles', ...) -> table id, for an existing base.

    Table display names carry a Russian suffix after an em dash ("Runs — Прогоны
    анализа"); the short key before it is what the code refers to.
    """
    return {t["name"].split(" —")[0].strip(): t["id"] for t in tb.tables(base_id)}
