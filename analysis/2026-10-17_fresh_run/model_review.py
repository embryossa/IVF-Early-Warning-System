"""Закрытие разбора §4: OOD против обучающей когорты конверта и χ² по категориям.

    python model_review.py <root>

Печатает и пишет в пакет аудита только счётчики и статистики, без строк когорты.
OOD — Махаланобис по признакам, общим для обучающей когорты TabPFN и наших циклов,
порог — квантиль χ²(df) 0.99 (тот же α 0.01, что у DT 7.1). Ковариация — по самой
когорте; доля OOD в когорте печатается как контроль (ожидаемо около 1%, больше —
признаки не гауссовы, и доли сравниваются между собой, а не с 1%).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from ews import cycles, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import embryo_lab_data_on_path, load  # noqa: E402

cfg = load(None); clinic = load_clinic(cfg, None); embryo_lab_data_on_path(cfg)
data_dir = clinic.data_dir(cfg)

FEATURES = {  # канон -> имя в обучающей когорте (envelope.RATE_MODEL_COLUMNS)
    "age": "Возраст", "attempt_no": "№ попытки", "follicles": "Количество фолликулов",
    "occ": "Число ОКК", "mii": "Число инсеминированных", "two_pn": "2 pN",
}
dev = pd.read_excel(data_dir / "local_private" / "dev_cohort.xlsx", usecols=list(FEATURES.values()))
dev = dev.rename(columns={v: k for k, v in FEATURES.items()}).apply(pd.to_numeric, errors="coerce")

frame = cycles.load_from_teable(cfg, clinic, teable_client.connect(cfg, dry_run=True),
                                cfg.external("teable")["opu_table_id"]).data
closed = frame.loc[frame["checkpoint"] == "closure"].copy()
cut = pd.Timestamp("2026-08-13")
groups = {"old": closed[closed["date_opu"] < cut], "new": closed[closed["date_opu"] >= cut]}

result = {"method": ("Mahalanobis против обучающей когорты TabPFN, порог χ²(df) 0.99; "
                     "признаки, общие для когорты и циклов; MII ~ «Число инсеминированных» когорты"),
          "ood": {}, "categorical": {}}
for label, feats in (("stage_T0", ["age", "attempt_no", "follicles", "occ", "mii"]),
                     ("stage_T1", ["age", "attempt_no", "follicles", "occ", "mii", "two_pn"])):
    ref = dev[feats].dropna()
    mu = ref.mean().to_numpy(); inv = np.linalg.pinv(np.cov(ref.to_numpy(), rowvar=False))
    limit = stats.chi2.ppf(0.99, df=len(feats))

    def share(df):
        x = df[feats].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
        if not len(x):
            return {"n": 0}
        d2 = np.einsum("ij,jk,ik->i", x - mu, inv, x - mu)
        return {"n": int(len(x)), "ood": int((d2 > limit).sum()), "share": round(float((d2 > limit).mean()), 3),
                "median_d2": round(float(np.median(d2)), 2)}

    entry = {"features": feats, "limit_d2": round(float(limit), 2), "dev_cohort": share(ref)}
    for name, g in groups.items():
        entry[name] = share(g)
    if entry["old"].get("n") and entry["new"].get("n"):
        table = [[entry["old"]["ood"], entry["old"]["n"] - entry["old"]["ood"]],
                 [entry["new"]["ood"], entry["new"]["n"] - entry["new"]["ood"]]]
        entry["fisher_old_vs_new_p"] = round(float(stats.fisher_exact(table)[1]), 4)
    result["ood"][label] = entry

rng = np.random.default_rng(17)
for col in ("sperm_source", "fertilization_method"):
    both = pd.concat([groups["old"][col].fillna("пусто").astype(str).str.strip().replace("", "пусто").assign() if False else
                      groups["old"][col].fillna("пусто").astype(str).str.strip().replace("", "пусто"),
                      groups["new"][col].fillna("пусто").astype(str).str.strip().replace("", "пусто")],
                     keys=["old", "new"]).reset_index(level=0).rename(columns={"level_0": "group"})
    table = pd.crosstab(both["group"], both[col])
    chi2 = stats.chi2_contingency(table.to_numpy())[0]
    # перестановочный p: ожидаемые частоты малы, асимптотика χ² ненадёжна
    labels, values = both["group"].to_numpy(), both[col].to_numpy()
    hits = 0
    for _ in range(10000):
        perm = rng.permutation(labels)
        t = pd.crosstab(perm, values).to_numpy()
        hits += stats.chi2_contingency(t)[0] >= chi2 - 1e-12
    result["categorical"][col] = {"table": {g: table.loc[g].to_dict() for g in table.index},
                                  "chi2": round(float(chi2), 3), "permutation_p": round((hits + 1) / 10001, 4)}

print(json.dumps(result, ensure_ascii=False, indent=1, default=int))
packet = data_dir.parents[0] / "agent_packets" / "ews_coverage_audit_two_line_20261017.json"
d = json.loads(packet.read_text(encoding="utf-8"))
d["ood_signal"] = result["ood"]; d["categorical_shift"] = result["categorical"]
d["review_completed_at"] = "2026-10-17"
packet.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=int), encoding="utf-8")
print("пакет аудита дополнен:", packet)
