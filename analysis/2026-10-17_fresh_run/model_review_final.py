"""(i) опорное покрытие/MAE только по Two-Line из Regression QC; (ii) KS старых и новых против обучающей."""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ews import cycles, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import embryo_lab_data_on_path, load  # noqa: E402
import ews_baseline  # noqa: E402

cfg = load(None); clinic = load_clinic(cfg, None); embryo_lab_data_on_path(cfg)
data_dir = clinic.data_dir(cfg)
client = teable_client.connect(cfg, dry_run=True)

qc = ews_baseline.collect_regression_qc(cfg, client)
own = qc.loc[qc["clinic"] == "Two-Line IVF"]
(by_stage, _), summary = ews_baseline.envelope_calibration(own, float(cfg.settings["envelope"]["iqr_to_sigma_divisor"]))
own_base = by_stage[["stage", "n", "coverage", "mae_p50"]].round(3).to_dict("records")

FEATURES = {"age": "Возраст", "attempt_no": "№ попытки", "follicles": "Количество фолликулов",
            "occ": "Число ОКК", "mii": "Число инсеминированных", "two_pn": "2 pN"}
dev = pd.read_excel(data_dir / "local_private" / "dev_cohort.xlsx", usecols=list(FEATURES.values()))
dev = dev.rename(columns={v: k for k, v in FEATURES.items()}).apply(pd.to_numeric, errors="coerce")
frame = cycles.load_from_teable(cfg, clinic, client, cfg.external("teable")["opu_table_id"]).data
closed = frame.loc[frame["checkpoint"] == "closure"]
cut = pd.Timestamp("2026-08-13")
ks = {}
for name, g in (("old", closed[closed["date_opu"] < cut]), ("new", closed[closed["date_opu"] >= cut])):
    ks[name] = {}
    for f in FEATURES:
        a, b = dev[f].dropna(), pd.to_numeric(g[f], errors="coerce").dropna()
        r = stats.ks_2samp(a, b)
        ks[name][f] = {"n": int(len(b)), "median_cohort": float(a.median()), "median_group": float(b.median()),
                       "ks": round(float(r.statistic), 3), "p": round(float(r.pvalue), 4)}

out = {"i_own_clinic_baseline": {"source": "Regression QC базы Two-Line IVF — Digital Twin, только своя клиника",
                                 "observations": summary["observations_total"], "by_stage": own_base},
       "ii_vs_training_cohort": ks}
print(json.dumps(out, ensure_ascii=False, indent=1))
packet = data_dir.parent / "agent_packets" / "ews_coverage_audit_two_line_20261017.json"
d = json.loads(packet.read_text(encoding="utf-8")); d["final_checks"] = out
packet.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
