#!/usr/bin/env python3
"""EWS re-analysis on the center D (Two-Line) OPU table used for the publication run.

Source: embryo-lab-data/data/inbox/publication_60/OPU table for publication.xlsx (126 OPU
cycles, Dec 2025 - Sep 2026). Laboratory population = embryo-lab-data canonical frame minus
kpi.competency_exclusion_mask (no retrieval, freeze-all, open cycles), exactly as
scripts/ews_baseline.py builds it. Uses the EWS chart code unchanged (src/ews/charts.py).

What is computed (nothing is written back to the live EWS state or Teable):
  * baseline rates in the declared reference window (Date OPU >= 2026-05-01) vs before;
  * between-cycle overdispersion (Pearson X2/df, rho) per chart;
  * thresholds by parametric bootstrap (solve_thresholds, beta-binomial dispersion,
    targets 24 watch / 2 alert per year for the system, 1,200 replicates, seed 20260813);
  * detection delay to alert for odds shifts x0.5..x0.8;
  * chronological replay of Bernoulli CUSUM (one block per cycle at culture closure);
  * EWMA (lambda 0.2) with overdispersion-inflated limits, monthly Shewhart p-charts
    (binomial and Laney p' limits);
  * out-of-window fertilization check (cleavage > 2PN) and MII > OCC.
Outputs in this folder carry no identifiers (cycle order and dates of culture closure only).
"""
import json, sys
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
EWS = HERE.parents[1]
ELD = EWS.parent / "embryo-lab-data"
sys.path[:0] = [str(EWS / "src"), str(ELD / "src"), str(ELD / "scripts")]
from ews import charts                                  # noqa: E402
import run_analysis as ra                               # noqa: E402
from embryo import ingest, kpi                          # noqa: E402

SRC = ELD / "data/inbox/publication_60/OPU table for publication.xlsx"
TODAY = pd.Timestamp("2026-09-29")
WINDOW_START = pd.Timestamp("2026-05-01")
CHARTS = {"FR": ("two_pn", "mii"), "BlR": ("bl_total", "two_pn"), "MIIR": ("mii", "occ")}
ABS = {"OCCR": ("occ", "follicles"), "ClR": ("cleavage_d3", "two_pn"), "TGBDR": ("bl_good", "two_pn")}
VIENNA = {"FR": (0.65, 0.75), "BlR": (0.40, 0.60), "MIIR": (0.75, 0.90), "OCCR": (0.70, 0.85), "TGBDR": (0.30, 0.40)}
TARGETS = {"watch": 24.0, "alert": 2.0}
REPL, SEED, OR = 1200, 20260813, 0.5

cfg = ra._settings(ELD / "config")
cc = ingest.load_config(str(ELD / "config" / "columns.yaml"))
_, frame, _ = ra._normalise_source(SRC, cc, cfg)
n_all = len(frame)
excl = kpi.competency_exclusion_mask(frame, cc)
lab = frame.loc[~excl].copy()
for c in {v for pair in list(CHARTS.values()) + list(ABS.values()) for v in pair}:
    lab[c] = pd.to_numeric(lab[c], errors="coerce")
lab["date_opu"] = pd.to_datetime(lab["date_opu"], errors="coerce")
lab["closure"] = lab["date_opu"] + pd.Timedelta(days=6)
lab = lab[lab["closure"] <= TODAY].sort_values("date_opu", kind="stable").reset_index(drop=True)
window = lab[lab["date_opu"] >= WINDOW_START]
prior = lab[lab["date_opu"] < WINDOW_START]

def pairs(df, num, den):
    d = df[[num, den]].dropna()
    return d[(d[den] > 0) & (d[num] <= d[den])]

def wilson(x, n, z=1.959964):
    p = x / n; c = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n); return m - c, m + c

def disp(df, num, den):
    d = pairs(df, num, den); x, n = d[num].to_numpy(float), d[den].to_numpy(float)
    p = x.sum() / n.sum(); chi2 = (((x - n * p) ** 2) / (n * p * (1 - p))).sum()
    ratio = chi2 / max(len(d) - 1, 1); rho = max(0.0, (ratio - 1) / max(n.mean() - 1, 1e-6))
    return dict(pearson_ratio=ratio, rho=min(rho, 0.9), n_cycles=len(d))

days = (lab["closure"].max() - WINDOW_START).days
base, rng = {}, np.random.default_rng(SEED)
for metric, (num, den) in {**CHARTS, **ABS}.items():
    w, p0_ = pairs(window, num, den), pairs(prior, num, den)
    x, n = w[num].sum(), w[den].sum(); xp, np_ = p0_[num].sum(), p0_[den].sum()
    lo, hi = wilson(x, n)
    tab = np.array([[x, n - x], [xp, np_ - xp]])
    pval = stats.chi2_contingency(tab)[1] if min(n, np_) > 0 else np.nan
    raw_all = frame[[num, den]].apply(pd.to_numeric, errors="coerce").dropna()
    base[metric] = dict(numerator=float(x), denominator=float(n), rate=x / n, ci_low=lo, ci_high=hi,
                        prior_rate=(xp / np_) if np_ else None, delta_pp=100 * (x / n - xp / np_) if np_ else None,
                        p_prior_vs_window=pval, cycles_window=len(w), cycles_prior=len(p0_),
                        competency=VIENNA.get(metric, (None, None))[0], benchmark=VIENNA.get(metric, (None, None))[1],
                        numerator_gt_denominator_cycles=int((raw_all[num] > raw_all[den]).sum()),
                        **{f"window_{k}": v for k, v in disp(window, num, den).items()},
                        **{f"all_{k}": v for k, v in disp(lab, num, den).items()})

calib = {}
for metric, (num, den) in CHARTS.items():
    b = base[metric]; p0 = b["rate"]; per_year = b["denominator"] * 365.0 / days
    sizes = pairs(window, num, den)[den].astype(int).to_numpy()
    solved = charts.solve_thresholds(charts.jeffreys_sampler(b["numerator"], b["denominator"]), p0, OR,
                                     TARGETS["watch"] / 3, TARGETS["alert"] / 3, horizon=max(int(round(per_year)), 50),
                                     replicates=REPL, rng=rng, block_sizes=sizes, dispersion=b["window_rho"])
    delays = {}
    for R in (0.5, 0.6, 0.7, 0.8):
        ps = p0 * R / (1 - p0 + R * p0)
        _, med = charts.simulate_detection_delay(ps, p0, OR, solved["h_alert"], horizon=int(per_year * 3),
                                                 replicates=REPL, rng=rng)
        delays[f"x{R}"] = med / (per_year / 12.0)
    calib[metric] = dict(p0=p0, observations_per_year=per_year, h_watch=solved["h_watch"], h_alert=solved["h_alert"],
                         false_watch_per_year=solved["rates"]["watch"], false_alert_per_year=solved["rates"]["alert"],
                         median_delay_months=delays, cycles_in_window=len(sizes))

# replay: one block per cycle, chronological by culture closure
events, trace = [], []
runners = {m: charts.CusumRunner() for m in CHARTS}
for i, row in lab.sort_values("closure", kind="stable").reset_index(drop=True).iterrows():
    rec = {"order": i + 1, "closure": row["closure"].date().isoformat()}
    for metric, (num, den) in CHARTS.items():
        x, n = row[num], row[den]
        if pd.isna(x) or pd.isna(n) or n <= 0 or x > n:
            rec[metric] = np.nan; continue
        c = calib[metric]
        ev = runners[metric].step_block(float(x), float(n), c["p0"], OR, c["h_watch"], c["h_alert"])
        rec[metric] = runners[metric].statistic; rec[f"{metric}_x"] = x; rec[f"{metric}_n"] = n
        if ev:
            events.append({"closure": rec["closure"], "chart": metric, "level": ev,
                           "in_window": row["date_opu"] >= WINDOW_START})
    trace.append(rec)
trace = pd.DataFrame(trace); ev = pd.DataFrame(events)
trace.to_csv(HERE / "cusum_trace.csv", index=False); ev.to_csv(HERE / "cusum_events.csv", index=False)

# EWMA on per-cycle proportions (lambda 0.2), limits inflated for overdispersion
ewma_rows, ewma_out = [], {}
for metric in CHARTS:
    p0, rho, lam = calib[metric]["p0"], base[metric]["window_rho"], 0.2
    d = trace[["order", "closure", f"{metric}_x", f"{metric}_n"]].dropna()
    z, var, below = p0, 0.0, 0
    for _, r in d.iterrows():
        n = r[f"{metric}_n"]; x = r[f"{metric}_x"]
        s2 = p0 * (1 - p0) / n * (1 + (n - 1) * rho)
        z = lam * (x / n) + (1 - lam) * z
        var = lam ** 2 * s2 + (1 - lam) ** 2 * var
        lcl, ucl = p0 - 3 * np.sqrt(var), p0 + 3 * np.sqrt(var)
        below += z < lcl
        ewma_rows.append({"chart": metric, "order": r["order"], "closure": r["closure"], "ewma": z,
                          "lcl": lcl, "ucl": ucl, "center": p0})
    ewma_out[metric] = {"points": len(d), "below_lcl": int(below), "above_ucl": int(sum(
        1 for q in ewma_rows if q["chart"] == metric and q["ewma"] > q["ucl"]))}
pd.DataFrame(ewma_rows).to_csv(HERE / "ewma.csv", index=False)

# monthly Shewhart p-charts (binomial and Laney p')
pc_rows, pc_out = [], {}
lab["month"] = lab["closure"].dt.to_period("M").astype(str)
for metric, (num, den) in {**CHARTS, "OCCR": ABS["OCCR"]}.items():
    p0 = base[metric]["rate"]
    g = pairs(lab.assign(month=lab["month"]), num, den).join(lab["month"]).groupby("month")[[num, den]].sum()
    zi = (g[num] / g[den] - p0) / np.sqrt(p0 * (1 - p0) / g[den])
    mr = np.abs(np.diff(zi.to_numpy())); sigma_z = (mr.mean() / 1.128) if len(mr) else 1.0
    for m, r in g.iterrows():
        se = np.sqrt(p0 * (1 - p0) / r[den])
        pc_rows.append({"chart": metric, "month": m, "x": r[num], "n": r[den], "p": r[num] / r[den], "center": p0,
                        "lcl_binom": p0 - 3 * se, "ucl_binom": p0 + 3 * se,
                        "lcl_laney": p0 - 3 * se * sigma_z, "ucl_laney": p0 + 3 * se * sigma_z})
    q = [r for r in pc_rows if r["chart"] == metric]
    pc_out[metric] = {"months": len(q), "sigma_z": float(sigma_z),
                      "outside_binomial": sum(r["p"] < r["lcl_binom"] or r["p"] > r["ucl_binom"] for r in q),
                      "outside_laney": sum(r["p"] < r["lcl_laney"] or r["p"] > r["ucl_laney"] for r in q)}
pd.DataFrame(pc_rows).to_csv(HERE / "pchart_monthly.csv", index=False)

cl = lab[["cleavage_d3", "two_pn"]].dropna()
summary = {
    "source": SRC.name, "rows": int(n_all), "excluded_competency_mask": int(excl.sum()),
    "lab_population_closed": int(len(lab)), "first_opu": str(lab["date_opu"].min().date()),
    "last_closure": str(lab["closure"].max().date()), "window_start": str(WINDOW_START.date()),
    "window_cycles": int(len(window)), "prior_cycles": int(len(prior)), "window_days": int(days),
    "baselines": base, "calibration": calib,
    "system_false_alarms_per_year": {"watch": sum(c["false_watch_per_year"] for c in calib.values()),
                                     "alert": sum(c["false_alert_per_year"] for c in calib.values())},
    "replay": {"cycles": int(len(trace)), "events_by_level": ev["level"].value_counts().to_dict() if len(ev) else {},
               "events_by_chart": ev["chart"].value_counts().to_dict() if len(ev) else {},
               "watch_in_window": int(((ev["level"] == "watch") & ev["in_window"]).sum()) if len(ev) else 0},
    "ewma": ewma_out, "pchart": pc_out,
    "out_of_window_fertilization_check": {"cycles_with_cleavage_and_2pn": int(len(cl)),
                                          "cleavage_gt_2pn": int((cl["cleavage_d3"] > cl["two_pn"]).sum())},
}
(HERE / "ews_publication_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("rows", "excluded_competency_mask", "lab_population_closed", "first_opu",
                  "last_closure", "window_cycles", "prior_cycles", "system_false_alarms_per_year", "replay",
                  "ewma", "pchart", "out_of_window_fertilization_check")}, indent=1, default=float))
for m, b in base.items():
    print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()})
for m, c in calib.items():
    print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k != "median_delay_months"},
          {k: round(v, 1) for k, v in c["median_delay_months"].items()})
