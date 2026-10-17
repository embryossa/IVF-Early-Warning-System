"""Состояние карт до прогона 17.10, пересобранное по данным 14.08 (экспорт 13.08).

    python rebuild_state_aug.py <root> [--write]

Проверка: BlR 0.00/630/0, FR 1.80/787/1, MIIR 0.00/1051/0 (первый прогон 17.10).
"""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(sys.argv[1]); WRITE = "--write" in sys.argv
sys.path.insert(0, str(ROOT / "src"))
from ews import cycles, evaluate, sync_opu  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import embryo_lab_data_on_path, load  # noqa: E402

cfg = load(None); clinic = load_clinic(cfg, None); embryo_lab_data_on_path(cfg)
from ews.vendor.privacy import load_or_create_secret, make_cycle_uid  # noqa: E402
secret = load_or_create_secret(cfg.path(cfg.settings["privacy"]["secret_file"]))
data_dir = clinic.data_dir(cfg)
old = sync_opu.load_sheet_export(ROOT / "data/local_private/OPU_table_export.csv")
records = []
for _, row in old.iterrows():
    key = sync_opu.build_key(row)
    if key:
        records.append({"id": key, "fields": sync_opu.row_to_fields(row, make_cycle_uid(key, secret), cfg.environment)})

class Fake:
    def all_records(self, table_id, projection=None):
        return records

frame = cycles.load_from_teable(cfg, clinic, Fake(), "x", today=datetime(2026, 10, 17, 20, 44))
print("чекпойнты:", frame.data["checkpoint"].value_counts().to_dict())
baselines = evaluate.baselines_from_manifest(data_dir / "baseline" / "incontrol_parameters.json")
states: dict = {}
evs = evaluate.apply_evaluations(cfg, clinic, frame.data, baselines, cfg.settings["thresholds"], states)
got = {m: (round(float(s.statistic), 2), s.observations, s.signals) for m, s in states.items()}
target = {"BlR": (0.0, 630, 0), "FR": (1.8, 787, 1), "MIIR": (0.0, 1051, 0)}
ok = all(got.get(m) == v for m, v in target.items())
print("оценено", len(evs), got, "СОВПАДАЕТ" if ok else "НЕ совпадает")
if ok and WRITE:
    evaluate.save_state(data_dir / "state" / "ews_charts_state.json", states, cfg, clinic, {e.cycle_uid for e in evs})
    print("состояние записано")
