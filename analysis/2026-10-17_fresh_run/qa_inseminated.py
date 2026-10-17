"""QA покрытия с и без циклов инсеминировано=0 на стадиях после оплодотворения."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from ews import cycles, envelope, envelope_qc, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import embryo_lab_data_on_path, load  # noqa: E402

cfg = load(None); clinic = load_clinic(cfg, None); embryo_lab_data_on_path(cfg)
frame = cycles.load_from_teable(cfg, clinic, teable_client.connect(cfg, dry_run=True),
                                cfg.external("teable")["opu_table_id"]).data
closed = frame.loc[frame["checkpoint"] == "closure"]
envs = envelope.as_lookup(envelope.load_envelopes(envelope.envelope_path(cfg, clinic)))
print("STAGE_ACTUAL:", envelope_qc.STAGE_ACTUAL)
not_ins = set(closed.loc[closed["inseminated"].eq(0), "cycle_uid"])
cut = pd.Timestamp("2026-08-13")
for name, sub in (("все", closed), ("старые", closed[closed["date_opu"] < cut]), ("новые", closed[closed["date_opu"] >= cut])):
    per, summ = envelope_qc.build(sub, envs, cfg.settings)
    usable = per.loc[per["status"] != envelope_qc.DEGENERATE]
    inside_ni = usable.loc[usable["cycle_uid"].isin(not_ins)]
    per2, summ2 = envelope_qc.build(sub.loc[~sub["cycle_uid"].isin(not_ins)], envs, cfg.settings)
    print(f"\n{name}: циклов без инсеминации в выборке {int(sub['cycle_uid'].isin(not_ins).sum())}")
    for (_, a), (_, b) in zip(summ.iterrows(), summ2.iterrows()):
        ni = inside_ni[inside_ni["stage"] == a["stage"]]
        print(f"  {a['stage']:18s} с ними n={a['n']:3d} {a['coverage']:.3f} p {a['p_below']:.4f}  |  "
              f"без них n={b['n']:3d} {b['coverage']:.3f} p {b['p_below']:.4f}  | у них в QA: {len(ni)}, вне 80%: {int((~ni['in80'].astype(bool)).sum())}")
