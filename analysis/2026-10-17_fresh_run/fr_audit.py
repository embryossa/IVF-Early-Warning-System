"""Все тревоги против метода/инсеминации + определение _fr_denominator в embryo.kpi."""
import inspect
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from ews import cycles, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import embryo_lab_data_on_path, load  # noqa: E402

cfg = load(None); cl = load_clinic(cfg, None); embryo_lab_data_on_path(cfg)
from ews.vendor import kpi  # noqa: E402

f = cycles.load_from_teable(cfg, cl, teable_client.connect(cfg, dry_run=True),
                            cfg.external("teable")["opu_table_id"]).data.set_index("cycle_uid")
a = pd.read_csv(ROOT / "data/two_line/ews_alerts.csv", encoding="utf-8-sig")
for col, src in (("method", "fertilization_method"), ("ins", "inseminated"), ("mii", "mii"), ("pn2", "two_pn")):
    a[col] = a["cycle_uid"].map(f[src])
print(a[["alert_id", "raised_at", "level", "chart", "method", "ins", "mii", "pn2"]].to_string())

source = inspect.getsource(kpi)
for m in re.finditer(r"_fr_denominator", source):
    print("=====\n" + source[max(0, m.start() - 400):m.start() + 400])
