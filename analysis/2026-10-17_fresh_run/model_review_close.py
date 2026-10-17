"""Закрытие открытых пунктов аудита §4 (a–e): только Python, только агрегаты.

(a) опорный mae_p50 и откуда опорное покрытие;
(b) версии и даты регистрации конвертов по группам;
(c) почему n стадии меньше числа циклов (причины исключения);
(d) тест источника спермы без «пусто» и только среди инсеминированных;
(e) итоги бэкфита: ложные срабатывания против цели.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from ews import cycles, envelope, envelope_qc, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import embryo_lab_data_on_path, load  # noqa: E402

cfg = load(None); clinic = load_clinic(cfg, None); embryo_lab_data_on_path(cfg)
data_dir = clinic.data_dir(cfg)
frame = cycles.load_from_teable(cfg, clinic, teable_client.connect(cfg, dry_run=True),
                                cfg.external("teable")["opu_table_id"]).data
closed = frame.loc[frame["checkpoint"] == "closure"].copy()
cut = pd.Timestamp("2026-08-13")
closed["group"] = np.where(closed["date_opu"] < cut, "old", "new")
out = {}

# (a)
base = pd.read_csv(data_dir / "baseline" / "envelope_coverage_by_stage.csv", encoding="utf-8-sig")
bm = json.loads((data_dir / "baseline" / "baseline_manifest.json").read_text(encoding="utf-8"))
out["a_baseline"] = {
    "source": ("опорное покрытие и mae_p50 измерены 2026-08-13 по Regression QC баз Digital Twin в Teable, "
               f"клиники: {', '.join(bm.get('clinics', []))}; {bm.get('observations_total')} наблюдений; "
               f"однородность: {bm.get('homogeneity')}"),
    "by_stage": base[["stage", "n", "coverage", "mae_p50"]].round(3).to_dict("records"),
    "mapping": "mii↔MII, pn2↔2PN, blasts_early и blasts_after_fert ↔ Бласт",
}

# (b)
env = envelope.load_envelopes(envelope.envelope_path(cfg, clinic))
env = env.merge(closed[["cycle_uid", "group"]], on="cycle_uid", how="inner")
out["b_envelope_versions"] = (env.groupby(["group", "model_version", "registration_mode"])
                              .agg(rows=("cycle_uid", "size"), first=("registered_at", "min"),
                                   last=("registered_at", "max")).reset_index().astype(str).to_dict("records"))

# (c)
envs = envelope.as_lookup(envelope.load_envelopes(envelope.envelope_path(cfg, clinic)))
reasons = []
for grp, sub in closed.groupby("group"):
    for stage, col in envelope_qc.STAGE_ACTUAL.items():
        counts = {"group": grp, "stage": stage, "cycles": int(len(sub))}
        no_env = sub["cycle_uid"].map(lambda u: envs.get(u, {}).get(f"{stage}_p10") is None)
        not_ins = pd.to_numeric(sub["inseminated"], errors="coerce").eq(0) & (stage in envelope_qc.POST_FERTILISATION)
        no_fact = pd.to_numeric(sub[col], errors="coerce").isna()
        counts["no_envelope_for_stage"] = int(no_env.sum())
        counts["not_inseminated_excluded"] = int((~no_env & not_ins).sum())
        counts["fact_missing"] = int((~no_env & ~not_ins & no_fact).sum())
        per, _ = envelope_qc.build(sub, envs, cfg.settings)
        st = per.loc[per["stage"] == stage] if not per.empty else per
        counts["degenerate_interval"] = int((st["status"] == envelope_qc.DEGENERATE).sum()) if len(st) else 0
        counts["usable_n"] = int((st["status"] != envelope_qc.DEGENERATE).sum()) if len(st) else 0
        reasons.append(counts)
out["c_exclusions"] = reasons

# (d)
ins = closed.loc[pd.to_numeric(closed["inseminated"], errors="coerce").gt(0)].copy()
ins["sperm"] = ins["sperm_source"].fillna("").astype(str).str.strip()
known = ins.loc[ins["sperm"] != ""]
table = pd.crosstab(known["group"], known["sperm"])
rng = np.random.default_rng(17)
chi2 = stats.chi2_contingency(table.to_numpy())[0]
labels, values = known["group"].to_numpy(), known["sperm"].to_numpy()
hits = sum(stats.chi2_contingency(pd.crosstab(rng.permutation(labels), values).to_numpy())[0] >= chi2 - 1e-12
           for _ in range(10000))
missing = ins.groupby("group")["sperm"].apply(lambda s: f"{int((s == '').sum())}/{len(s)}").to_dict()
out["d_sperm_source_inseminated_only"] = {
    "table_known": {g: table.loc[g].to_dict() for g in table.index}, "chi2": round(float(chi2), 3),
    "permutation_p": round((hits + 1) / 10001, 4), "missing_among_inseminated": missing,
    "note": "пропуск источника спермы — неизвестно, а не категория; в тест не входит"}

# (e)
cal = json.loads((data_dir / "baseline" / "calibration_manifest.json").read_text(encoding="utf-8"))
out["e_backfit"] = {k: cal.get(k) for k in ("calibrated_at", "calibrated_environment", "targets_per_year",
                                             "system_false_alarms_per_year", "charts_active", "excluded_from_charts")}
out["e_backfit"]["charts"] = cal.get("charts")

print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])
packet = data_dir.parent / "agent_packets" / "ews_coverage_audit_two_line_20261017.json"
d = json.loads(packet.read_text(encoding="utf-8"))
d["open_items_closed"] = out
d.pop("review_completed_at", None)
packet.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("\nпакет дополнен")
