"""Предпросмотр панели из сохранённого состояния, без прогона и без записи в data/."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
from ews import cycles, evaluate, panel, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import embryo_lab_data_on_path, load  # noqa: E402

cfg = load(None); clinic = load_clinic(cfg, None); embryo_lab_data_on_path(cfg)
data_dir = clinic.data_dir(cfg)
frame = cycles.load_from_teable(cfg, clinic, teable_client.connect(cfg, dry_run=True),
                                cfg.external("teable")["opu_table_id"])
states, _ = evaluate.load_state(data_dir / "state" / "ews_charts_state.json")
baselines = evaluate.baselines_from_manifest(data_dir / "baseline" / "incontrol_parameters.json")
journal = pd.read_csv(data_dir / "ews_alerts.csv", encoding="utf-8-sig")
panel.render(cfg, clinic, frame, [], states, baselines, cfg.settings["thresholds"], OUT,
             alerts_journal=journal)
print(OUT)
