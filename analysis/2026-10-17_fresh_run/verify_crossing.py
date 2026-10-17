"""Сверка: статистика FR в момент пересечения h_alert, p0 опорной линии, логика above_watch."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from ews import charts, cycles, evaluate, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import embryo_lab_data_on_path, load  # noqa: E402

cfg = load(None); clinic = load_clinic(cfg, None); embryo_lab_data_on_path(cfg)
data_dir = clinic.data_dir(cfg)

log = []
original = charts.CusumRunner.step_block


def logged(self, successes, trials, expected, odds_ratio, h_watch, h_alert):
    p = min(max(float(expected), 1e-6), 1 - 1e-6)
    weight = successes * np.log(odds_ratio) - trials * np.log(1.0 - p + odds_ratio * p)
    before = self.statistic
    event = original(self, successes, trials, expected, odds_ratio, h_watch, h_alert)
    log.append({"before": before, "pre_reset": max(0.0, before + weight), "event": event,
                "h_watch": h_watch, "h_alert": h_alert, "trials": trials, "successes": successes,
                "expected": p})
    return event


charts.CusumRunner.step_block = logged
frame = cycles.load_from_teable(cfg, clinic, teable_client.connect(cfg, dry_run=True),
                                cfg.external("teable")["opu_table_id"]).data
baselines = evaluate.baselines_from_manifest(data_dir / "baseline" / "incontrol_parameters.json")
states: dict = {}
evs = evaluate.apply_evaluations(cfg, clinic, frame, baselines, cfg.settings["thresholds"], states)
for item in log:
    if item["event"] == "alert":
        print("пересечение alert:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in item.items()})
print("FR peak в состоянии:", round(states["FR"].peak, 3))
print("p0:", {m: baselines[m] for m in ("FR", "BlR", "MIIR") if m in baselines})
raw = json.loads((data_dir / "baseline" / "incontrol_parameters.json").read_text(encoding="utf-8"))
print("ключи incontrol:", list(raw.keys())[:12])
