"""Оверлей machine.<code>.local.yaml: вторая клиника не трогает id первой."""
from types import SimpleNamespace

import yaml

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import config as config_mod  # noqa: E402
from ews import teable_client  # noqa: E402


def _setup(tmp_path, monkeypatch):
    (tmp_path / "settings.yaml").write_text("thresholds: {}\n", encoding="utf-8")
    (tmp_path / "machine.local.yaml").write_text(
        "environment: production\nembryo_lab_data:\n  clinic: two_line\n"
        "teable:\n  url: 'http://127.0.0.1:3000'\n  base_id: \"BASE_TWO\"   # bootstrap\n"
        "  opu_table_id: \"OPU_TWO\"\n  dt_base_id: \"DT_TWO\"\n"
        "google_sheet:\n  export_path: 'data/local_private/OPU table.csv'\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    return tmp_path


def test_overlay_applies_only_to_its_clinic(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch)
    (d / "machine.persona.local.yaml").write_text(
        "teable:\n  base_id: ''\n  dt_base_id: ''\ngoogle_sheet:\n  export_path: 'data/persona/x.csv'\n",
        encoding="utf-8")
    two = config_mod.load("two_line")
    per = config_mod.load("persona")
    assert two.machine["teable"]["base_id"] == "BASE_TWO"
    assert per.machine["teable"]["base_id"] == ""
    assert per.machine["teable"]["dt_base_id"] == ""
    assert per.machine["teable"]["url"] == "http://127.0.0.1:3000"   # унаследовано
    assert per.machine["google_sheet"]["export_path"] == "data/persona/x.csv"
    assert two.machine["google_sheet"]["export_path"] == "data/local_private/OPU table.csv"


def test_save_ids_writes_overlay_not_shared_file(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch)
    (d / "machine.persona.local.yaml").write_text("teable:\n  base_id: ''\n", encoding="utf-8")
    before = (d / "machine.local.yaml").read_text(encoding="utf-8")
    teable_client.save_ids(SimpleNamespace(clinic_code="persona"), "B_P", "O_P", "S_P")
    assert (d / "machine.local.yaml").read_text(encoding="utf-8") == before
    over = yaml.safe_load((d / "machine.persona.local.yaml").read_text(encoding="utf-8"))
    assert over["teable"] == {"base_id": "B_P", "opu_table_id": "O_P", "sync_table_id": "S_P"}


def test_without_overlay_shared_file_is_updated(tmp_path, monkeypatch):
    d = _setup(tmp_path, monkeypatch)
    teable_client.save_ids(SimpleNamespace(clinic_code="two_line"), "B2", "O2", "S2")
    text = (d / "machine.local.yaml").read_text(encoding="utf-8")
    assert 'base_id: "B2"   # bootstrap' in text and 'sync_table_id: "S2"' in text
