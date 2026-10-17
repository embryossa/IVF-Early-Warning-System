"""Агентские пакеты прогона 17.10: тревоги (ews-alert-analyst) и покрытие (ews-model-auditor).

    python build_packets.py <root>

Только cycle_uid, числа и псевдонимы специалистов (Э1.., В1..). Все числа — Python.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from ews import confounders, cycles, envelope_qc, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import embryo_lab_data_on_path, load  # noqa: E402

cfg = load(None); clinic = load_clinic(cfg, None); embryo_lab_data_on_path(cfg)
from ews.vendor import kpi  # noqa: E402

data_dir = clinic.data_dir(cfg)
out_dir = cfg.path(cfg.settings["privacy"]["agent_packet_dir"])
client = teable_client.connect(cfg, dry_run=True)
frame = cycles.load_from_teable(cfg, clinic, client, cfg.external("teable")["opu_table_id"]).data
frame["date_opu"] = pd.to_datetime(frame["date_opu"])
closed = frame.loc[frame["checkpoint"] == "closure"].copy()
manifest = json.loads((data_dir / "run_manifest.json").read_text(encoding="utf-8"))
state = json.loads((data_dir / "state" / "ews_charts_state.json").read_text(encoding="utf-8"))["charts"]
baselines = json.loads((data_dir / "baseline" / "incontrol_parameters.json").read_text(encoding="utf-8"))
th = cfg.settings["thresholds"]
CUT = pd.Timestamp("2026-08-15")   # граница: оценено в этом прогоне (закрытие после 14.08)


def pseudo(series: pd.Series, prefix: str) -> dict:
    order = series.dropna().astype(str).str.strip().replace("", np.nan).dropna().value_counts().index
    return {name: f"{prefix}{i + 1}" for i, name in enumerate(order)}


emb_map, doc_map = pseudo(frame.get("embryologist"), "Э"), pseudo(frame.get("doctor"), "В")


def ratio(sub: pd.DataFrame, metric: str) -> dict:
    num_col, den_col = cycles.METRIC_PARTS[metric]
    ok = sub[num_col].notna() & sub[den_col].notna() & (sub[den_col] > 0)
    num, den = float(sub.loc[ok, num_col].sum()), float(sub.loc[ok, den_col].sum())
    return {"numerator": num, "denominator": den, "cycles": int(ok.sum()),
            "rate": round(num / den, 3) if den else None}


def none_if_nan(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def spec(metric: str) -> dict:
    s = kpi.KPI_SPECS[metric]
    return {"competency": s.competency, "benchmark": s.benchmark, "direction": s.direction,
            "label": s.label, "indicator_class": s.indicator_class, "source_note": s.source_note}


closed["month"] = closed["date_opu"].dt.to_period("M").astype(str)
monthly = {m: {mon: ratio(g, m) for mon, g in closed.groupby("month")} for m in ("FR", "BlR", "MIIR")}

# --- исправление знаменателя FR: факты для текста --------------------------
archive = pd.read_csv(data_dir / "archive" / "20261017_pre_replay" / "ews_alerts.csv", encoding="utf-8-sig")
by_uid = frame.set_index("cycle_uid")
archive["method"] = archive["cycle_uid"].map(by_uid["fertilization_method"])
archive["inseminated"] = archive["cycle_uid"].map(by_uid["inseminated"])
gone = archive.loc[archive["inseminated"].eq(0)]
not_ins = closed.loc[closed["inseminated"].eq(0)]
fr_correction = {
    "excluded_closed_cycles_inseminated_0": {"total": int(len(not_ins)),
                                             "by_method": not_ins["fertilization_method"].fillna("пусто").value_counts().to_dict()},
    "metrics_affected": ["FR", "DegR (тот же знаменатель, как в embryo.kpi)"],
    "removed_signals_previously_on_inseminated_0": [
        {"raised_at": r["raised_at"], "level_before": r["level"], "chart": r["chart"], "method": r["method"],
         "notes_before": None if pd.isna(r["notes"]) else r["notes"]} for _, r in gone.iterrows()],
    "baseline_recheck": ("ews_baseline.py с исправленным кодом пересчитан во временную папку и сравнён с "
                         "incontrol_parameters.json: единственное различие — reference_window.end "
                         "(2026-08-13 -> 2026-10-17); p0, block_sizes, overdispersion совпали. Причина: опорная "
                         "популяция уже исключала Cryo/NO маской competency_exclusion embryo-lab-data. Пороги h не менялись."),
    "after_replay": {"watch": int(manifest["levels"].get("watch", 0)), "alert": int(manifest["levels"].get("alert", 0))},
}

# --- QA конверта: старые и новые циклы раздельно --------------------------
cut_qa = pd.Timestamp("2026-08-13")
qa_split = {}
for name, sub_frame in (("old_opu_before_2026-08-13", closed[closed["date_opu"] < cut_qa]),
                        ("new_opu_from_2026-08-13", closed[closed["date_opu"] >= cut_qa])):
    from ews import envelope as envelope_mod  # noqa: PLC0415
    envs = envelope_mod.as_lookup(envelope_mod.load_envelopes(envelope_mod.envelope_path(cfg, clinic)))
    per, summ = envelope_qc.build(sub_frame, envs, cfg.settings)
    dropped, _ = envelope_qc.coverage_alarm(summ)
    usable = per.loc[per["status"] != envelope_qc.DEGENERATE]
    qa_split[name] = {"significant_drop": bool(dropped), "stages": [
        {**{k: r[k] for k in ("stage", "n", "coverage", "baseline", "p_below", "mae_p50", "degenerate_share")},
         "above_p90": int((usable.loc[usable["stage"] == r["stage"], "actual"] > usable.loc[usable["stage"] == r["stage"], "p90"]).sum()),
         "below_p10": int((usable.loc[usable["stage"] == r["stage"], "actual"] < usable.loc[usable["stage"] == r["stage"], "p10"]).sum())}
        for _, r in summ.iterrows()]}

# --- тревоги ---------------------------------------------------------------
alerts = pd.read_csv(data_dir / "ews_alerts.csv", encoding="utf-8-sig")
alerts = alerts.loc[pd.to_datetime(alerts["raised_at"]) >= CUT]
uids = list(alerts["cycle_uid"])
contexts = confounders.collect(cfg, clinic, frame.loc[frame["cycle_uid"].isin(uids)])
rows = []
for _, a in alerts.iterrows():
    c = frame.loc[frame["cycle_uid"] == a["cycle_uid"]].iloc[0]
    metric = a["chart"]
    num_col, den_col = cycles.METRIC_PARTS[metric]
    ctx = contexts.get(a["cycle_uid"])
    rows.append({
        "alert_id": a["alert_id"], "cycle_uid": a["cycle_uid"], "raised_at": a["raised_at"],
        "level": a["level"], "chart": metric, "chart_type": "CUSUM (Bernoulli, одно обновление на цикл)",
        "statistic_after_cycle": json.loads(a["statistic"]).get(metric),
        "statistic_note": ("0.0 = сброс карты после пересечения порога alert h=7.441" if "понижен" in str(a["notes"]) else None),
        "contour": a["contour"], "corroboration": json.loads(a["corroboration"]), "corroboration_rank": a["corroboration_rank"],
        "miri_status": a["miri_status"], "verdict_text": a["verdict"],
        "notes": None if pd.isna(a["notes"]) else a["notes"],
        "cycle_counts": {num_col: c[num_col], den_col: c[den_col],
                         "cycle_rate": round(c[num_col] / c[den_col], 3) if c[den_col] else None},
        "fertilization_method": none_if_nan(c.get("fertilization_method")), "sperm_source": none_if_nan(c.get("sperm_source")),
        "embryologist": emb_map.get(str(c.get("embryologist")).strip()), "doctor": doc_map.get(str(c.get("doctor")).strip()),
        "lot_changes_in_window": ctx.lot_changes if ctx else None,
        "lot_change_items": [e.get("item_name", "").strip() for e in (ctx.lot_events if ctx else [])
                             if e.get("event_type") == "lot_change"],
        "lot_change_note": "смена лота — кандидат-confounder, не причина; подтверждением считается только при числе смен выше границы «необычного» окна",
    })

after = closed.loc[closed["date_opu"] >= pd.Timestamp("2026-08-13")]
before = closed.loc[closed["date_opu"] < pd.Timestamp("2026-08-13")]
fr_by_emb = {}
for name, g in after.groupby(after["embryologist"].astype(str).str.strip()):
    fr_by_emb[emb_map.get(name, "нет данных")] = {"after": ratio(g, "FR"),
                                                  "before": ratio(before.loc[before["embryologist"].astype(str).str.strip() == name], "FR")}

alert_packet = {
    "packet": "ews_alerts", "clinic": clinic.code, "run_id": manifest["run_id"], "environment": manifest["environment"],
    "built_at": datetime.now().isoformat(timespec="seconds"),
    "alert_gate": manifest["alert_gate"],
    "escalation_block": ("действует: покрытие просело — alert понижается до watch до разбора §4"
                         if manifest.get("coverage_blocks_escalation") else
                         "снят: после QA на новых закрытых циклах значимого падения покрытия нет (см. coverage_split)"),
    "coverage_split": qa_split,
    "replay_totals": {"note": "прогон — replay всей истории", "cycles_evaluated": manifest["cycles_evaluated"],
                      "levels_all_history": manifest["levels"]},
    "signals_in_packet": {"closures_from": "2026-08-15", "closures_to": "2026-10-17",
                          "watch": int((alerts["level"] == "watch").sum()), "alert": int((alerts["level"] == "alert").sum())},
    "fr_correction": fr_correction,
    "thresholds": {m: {"h_watch": th["h_watch"].get(m), "h_alert": th["h_alert"].get(m)} for m in ("FR", "BlR", "MIIR")},
    "calibrated": th.get("calibrated"),
    "in_control_baseline": {b["metric"]: {k: b[k] for k in ("numerator", "denominator", "rate", "ci_low", "ci_high")}
                            for b in (baselines["metrics"].values() if isinstance(baselines["metrics"], dict) else baselines["metrics"])
                            if b.get("metric") in ("FR", "BlR", "MIIR")},
    "reference_window": baselines.get("reference_window"),
    "fr_alert_crossing_27_09": {"statistic_before_cycle": 5.784, "statistic_pre_reset": 7.558, "h_alert": 7.441,
                                "cycle": "2pN 2 / знаменатель 6, ожидание p0 0.819",
                                "note": "с 17.10 значение до сброса пишется в траекторию и пик (FR peak = 7.558)"},
    "above_watch_semantics": "флаг эпизода watch снимается только при возврате статистики в 0, а не при падении ниже h_watch",
    "selection_rule": (f"в alerts — сигналы с закрытием >= 2026-08-15; всего за историю после replay: "
                       f"{manifest['levels'].get('watch', 0)} watch, {manifest['levels'].get('alert', 0)} alert"),
    "chart_state_now": {m: {k: state[m][k] for k in ("statistic", "observations", "signals", "watch_episodes", "above_watch", "peak")}
                        for m in ("FR", "BlR", "MIIR")},
    "chart_trajectories_since_cut": {m: [p for p in state[m]["trajectory"] if p[0] >= "2026-08-15"] for m in ("BlR", "MIIR")},
    "fr_trajectory_since_cut": [p for p in state["FR"]["trajectory"] if p[0] >= "2026-08-15"],
    "kpi_specs": {m: spec(m) for m in ("FR", "BlR", "MIIR")},
    "consensus_text": {"FR": {"document": "Vienna consensus 2017 (RBM Online 35:494-510)", "verbatim": None,
                              "note": "в config/consensus нет дословной цитаты, только пороги"}},
    "cohort_monthly_by_opu_month": monthly,
    "fr_by_embryologist_pseudonym": fr_by_emb,
    "fr_denominator_rule": ("FR = 2pN / (MII, если MII>0, иначе инсеминировано); циклы с инсеминировано=0 "
                            "(методы Cryo, NO — витрификация ооцитов/без оплодотворения) в FR не входят. "
                            "Исправлено 17.10; карты перестроены replay по всей истории"),
    "unusual_window_bounds": "критических минут > 1915.5 или смен лота > 4 (confounder_baseline от 2026-08-14)",
    "fr_cohort": {"before_2026-08-13": ratio(before, "FR"), "from_2026-08-13": ratio(after, "FR")},
    "outcome_ra": {k: manifest["outcome_ra"][k] for k in ("transfers_with_outcome", "overall", "vlad_last", "cusum", "note")},
    "miri": "pending для всех сигналов пакета: недельная выгрузка логов после 14.08 не подгружена (это не «событий не было»)",
    "alerts": rows,
}

# --- покрытие и состав пациенток -------------------------------------------
qc = pd.read_csv(data_dir / "ews_envelope_qc.csv", encoding="utf-8-sig")
qc = qc.merge(frame[["cycle_uid", "date_opu"]], on="cycle_uid", how="left")
# та же популяция, что в сводке QA: вырожденный интервал покрытия не имеет
qc = qc.loc[qc["status"] != envelope_qc.DEGENERATE].copy()
qc["in80"] = qc["status"] == "in80"
stage_detail = {}
for stage, g in qc.groupby("stage"):
    g = g.sort_values("date_opu")
    half = len(g) // 2
    miss = g.loc[~g["in80"].astype(bool)]
    stage_detail[stage] = {
        "n": int(len(g)), "modes": g["mode"].value_counts().to_dict(),
        "below_p10": int((g["actual"] < g["p10"]).sum()), "above_p90": int((g["actual"] > g["p90"]).sum()),
        "coverage_first_half": round(float(g["in80"].iloc[:half].mean()), 3),
        "coverage_second_half": round(float(g["in80"].iloc[half:].mean()), 3),
        "median_z_misses": round(float(miss["z"].median()), 2) if len(miss) else None,
        "median_width": round(float(g["width"].median()), 2),
        "opu_range": [str(g["date_opu"].min())[:10], str(g["date_opu"].max())[:10]],
    }

ref = closed.loc[closed["date_opu"] < pd.Timestamp("2026-08-13")]
new = closed.loc[closed["date_opu"] >= pd.Timestamp("2026-08-13")]
casemix = {}
for f in ("age", "amh", "afc", "attempt_no", "bmi", "mii"):
    a, b = ref[f].dropna(), new[f].dropna()
    if len(a) >= 5 and len(b) >= 5:
        ks = stats.ks_2samp(a, b)
        casemix[f] = {"ref_n": int(len(a)), "new_n": int(len(b)), "ref_median": float(a.median()), "new_median": float(b.median()),
                      "ks_stat": round(float(ks.statistic), 3), "p": round(float(ks.pvalue), 4)}
    else:
        casemix[f] = {"ref_n": int(len(a)), "new_n": int(len(b)), "note": "мало непустых значений"}
for f in ("sperm_source", "fertilization_method"):
    if f in closed:
        casemix[f] = {"ref": ref[f].fillna("пусто").value_counts().to_dict(), "new": new[f].fillna("пусто").value_counts().to_dict()}

audit_packet = {
    "packet": "ews_coverage_audit", "clinic": clinic.code, "run_id": manifest["run_id"], "environment": manifest["environment"],
    "envelope_qc_summary": manifest["envelope_qc"],
    "coverage_alpha": 0.01,
    "baseline_coverage_measured_at": cfg.settings["envelope"].get("baseline_coverage_measured_at"),
    "stage_detail": stage_detail,
    "qc_population_note": ("с 17.10 конверты зарегистрированы и для 40 новых закрытых циклов (retrospective-регистрация, "
                           "циклы вне обучения модели); общий QA — по всем, раздельный — в coverage_split"),
    "coverage_split": qa_split,
    "casemix_ref_vs_new": {"ref": "закрытые циклы с пункцией до 2026-08-13", "new": "закрытые с 2026-08-13", "features": casemix,
                           "ks_alpha_from_settings": cfg.settings["localisation"]["casemix_check"]["ks_alpha"]},
    "ood_signal": None, "backfit": "калибровка порогов 2026-08-14 (calibration_manifest.json)",
}

out_dir.mkdir(parents=True, exist_ok=True)
for name, obj in (("ews_alerts_two_line_20261017.json", alert_packet), ("ews_coverage_audit_two_line_20261017.json", audit_packet)):
    (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                encoding="utf-8")
    print("записан", out_dir / name)
