"""QA конверта отдельно: старые (до 13.08) и новые закрытые циклы."""
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
closed = frame.loc[frame["checkpoint"] == cycles.CHECKPOINT_CLOSURE]
envs = envelope.as_lookup(envelope.load_envelopes(envelope.envelope_path(cfg, clinic)))
cut = pd.Timestamp("2026-08-13")
for name, sub in (("старые (пункция < 13.08)", closed[closed["date_opu"] < cut]),
                  ("новые (пункция ≥ 13.08)", closed[closed["date_opu"] >= cut])):
    per, summary = envelope_qc.build(sub, envs, cfg.settings)
    dropped, _ = envelope_qc.coverage_alarm(summary)
    print(f"\n{name}: циклов {len(sub)}; значимое падение: {dropped}")
    usable = per.loc[per["status"] != envelope_qc.DEGENERATE]
    for _, r in summary.iterrows():
        u = usable[usable["stage"] == r["stage"]]
        print(f"  {r['stage']:18s} n={r['n']:3d} покрытие {r['coverage']:.3f} (опорное {r['baseline']}) "
              f"p_below {r['p_below']:.4f}  выше P90 {int((u['actual'] > u['p90']).sum())} ниже P10 {int((u['actual'] < u['p10']).sum())}"
              f"  MAE P50 {r['mae_p50']}")
