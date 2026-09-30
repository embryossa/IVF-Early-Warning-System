"""Сборка всех расчётов демо-панели и детерминированных наблюдений на русском.

Наблюдения («факты») формулирует код из готовых чисел. Гипотезы («что проверить»)
берутся из ``config/consensus/vienna.yaml`` (on_signal_check) и явно отделены от
фактов. Модель в этом слое не участвует: комментарий ИИ, если он есть, приходит
отдельным файлом и помечается как интерпретация (см. скилл ews-demo).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..vendor import kpi
from . import stats as st
from .ingest import DemoFrame

ROOT = Path(__file__).resolve().parents[3]
# Эпизод CUSUM короче этого числа циклов не сравнивается тестом: разница долей в
# одном-двух циклах — это одна-две пациентки, а не процесс.
MIN_SEGMENT_CYCLES = 3
MONTHS_RU = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def pct(v: float, digits: int = 1) -> str:
    return "н/д" if v is None or not np.isfinite(v) else f"{100 * v:.{digits}f}%"


def pval(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "н/д"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def month_ru(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{MONTHS_RU[ts.month - 1]} {ts.year}"


def _chart_metrics() -> list[str]:
    path = ROOT / "config" / "clinics" / "_template" / "clinic.yaml"
    with path.open(encoding="utf-8") as fh:
        clinic = yaml.safe_load(fh) or {}
    return list((clinic.get("charts") or {}).get("cusum") or ["FR", "BlR", "MIIR"])


def _vienna_checks() -> dict:
    with (ROOT / "config" / "consensus" / "vienna.yaml").open(encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("indicators", {})


def build(demo: DemoFrame, *, rolling_window: int = 20, ref_start: str | None = None,
          ref_end: str | None = None, replicates: int = 300, seed: int = 20261001) -> dict:
    settings = st.load_settings()
    rng = np.random.default_rng(seed)
    frame = demo.frame
    rates = st.cycle_rates(frame)
    has_dates = "date_opu" in rates.columns and rates["date_opu"].notna().any()

    res: dict = {"source_name": demo.source_name, "generated_at": date.today().isoformat(),
                 "environment": "draft", "mode": "demo", "columns": demo.columns.as_dict(),
                 "exclusions": demo.exclusions, "notes": list(demo.notes),
                 "settings_used": {"rolling_window_cycles": rolling_window,
                                   "min_den_display": st.MIN_DEN_DISPLAY,
                                   "min_cycles_test": st.MIN_CYCLES_TEST,
                                   "replicates": replicates, "seed": seed,
                                   "ewma_lambda": settings["charts"]["ewma"]["lambda"],
                                   "ewma_L": 3.0,
                                   "targets_per_year": settings["thresholds"]["target_false_alarms_per_year"]}}
    if has_dates:
        res["period"] = {"start": rates["date_opu"].min().date().isoformat(),
                         "end": rates["date_opu"].max().date().isoformat()}
    res["n_cycles"] = int(len(frame))
    res["n_2pn"] = float(pd.to_numeric(frame.get("two_pn"), errors="coerce").sum()) \
        if "two_pn" in frame else float("nan")

    # 2PN == 0 — стоящее исключение владельца (kpi.compute_rates). Печатаем, сколько их
    # и каким был бы FR с ними: исключение должно быть видно, а не молча менять цифру.
    if "two_pn" in frame.columns:
        zero = pd.to_numeric(frame["two_pn"], errors="coerce").eq(0)
        res["two_pn_zero_cycles"] = int(zero.sum())

    summary = st.kpi_summary(rates)
    res["kpi"] = summary
    if res.get("two_pn_zero_cycles"):
        fr_rows = st.metric_rows(rates, "FR")
        z = frame[pd.to_numeric(frame["two_pn"], errors="coerce").eq(0)]
        den = z["mii"].where(z.get("mii", pd.Series(dtype=float)) > 0, z.get("inseminated")) \
            if "mii" in z else z.get("inseminated")
        if den is not None:
            den = pd.to_numeric(den, errors="coerce").fillna(0)
            res["fr_with_zero_2pn"] = float(fr_rows["x"].sum() / (fr_rows["n"].sum() + den.sum()))

    # --- распределения по циклам -------------------------------------------
    dist = {}
    for metric in st.DISPLAY_ORDER:
        r = st.metric_rows(rates, metric)
        dist[metric] = r[r["n"] >= st.MIN_DEN_DISPLAY]
    res["distributions"] = dist

    # --- базовые доли и дисперсия для карт ----------------------------------
    ref_mask = pd.Series(True, index=rates.index)
    if has_dates and ref_start:
        ref_mask &= rates["date_opu"] >= pd.Timestamp(ref_start)
    if has_dates and ref_end:
        ref_mask &= rates["date_opu"] <= pd.Timestamp(ref_end)
    res["reference"] = {"start": ref_start, "end": ref_end,
                        "declared": bool(ref_start or ref_end)}

    series = {}
    chart_metrics = [m for m in _chart_metrics() if m in st.DISPLAY_ORDER]
    available_charts = []
    for metric in st.DISPLAY_ORDER:
        rows = st.metric_rows(rates, metric)
        if rows.empty or not has_dates:
            continue
        ref_rows = rows[rows.index.isin(rates.index[ref_mask])] if res["reference"]["declared"] else rows
        ref_rows = ref_rows[ref_rows["x"] <= ref_rows["n"]]
        if ref_rows.empty:
            continue
        p0 = ref_rows["x"].sum() / ref_rows["n"].sum()
        disp = st.dispersion(ref_rows["x"], ref_rows["n"])
        pchart, pinfo = st.laney_pchart(rows, p0)
        entry = {"p0": p0, "phi": disp["phi"], "rho": disp["rho"],
                 "rolling": st.rolling_pooled(rows, rolling_window, disp["phi"]),
                 "monthly": st.monthly(rows, disp["phi"]),
                 "pchart": pchart, "pchart_info": pinfo,
                 "ewma": st.ewma(rows, p0, disp["rho"], settings["charts"]["ewma"]["lambda"], 3.0),
                 "direction": st.spec(metric).direction}
        mrows = rows.dropna(subset=["date_opu"]).assign(month=lambda d: d["date_opu"].dt.to_period("M").astype(str))
        entry["month_test"] = st.compare_groups(mrows, "month", min_cycles=3)
        series[metric] = entry
        if metric in chart_metrics:
            available_charts.append(metric)

    cusum = {}
    for metric in available_charts:
        e = series[metric]
        cusum[metric] = st.cusum_replay(st.metric_rows(rates, metric), e["p0"], e["rho"],
                                        e["direction"], settings, len(available_charts), rng,
                                        replicates=replicates)
    res["series"], res["cusum"] = series, cusum

    # --- эпизоды CUSUM: сегмент против предшествующего периода --------------
    episodes = []
    for metric, c in cusum.items():
        if not c.get("available") or c["events"].empty:
            continue
        rows = st.metric_rows(rates, metric).dropna(subset=["date_opu"])
        rows = rows[rows["x"] <= rows["n"]]
        for _, ev in c["events"].iterrows():
            seg = rows[(rows["date_opu"] >= ev["segment_start"]) & (rows["date_opu"] <= ev["date"])]
            before = rows[rows["date_opu"] < ev["segment_start"]]
            base_label = "предшествующий период"
            if len(before) < 5:
                before = rows[~rows.index.isin(seg.index)]
                base_label = "остальные циклы"
            if seg.empty or before.empty:
                continue
            cmp = st.two_period(seg["x"].sum(), seg["n"].sum(), before["x"].sum(), before["n"].sum(),
                                series[metric]["phi"])
            if len(seg) < MIN_SEGMENT_CYCLES:
                # Эпизод из 1–2 циклов: сравнение на уровне эмбрионов здесь меряет
                # одну-две пациентки, а не процесс. Доли показываем, p — нет.
                cmp.update({"p": float("nan"), "lo_pp": float("nan"), "hi_pp": float("nan"),
                            "test": f"меньше {MIN_SEGMENT_CYCLES} циклов — тест не выполнялся"})
            episodes.append({"metric": metric, "signal_date": ev["date"], "segment_start": ev["segment_start"],
                             "alert_threshold_crossed": bool(ev["alert_threshold_crossed"]),
                             "segment_cycles": int(len(seg)), "baseline": base_label,
                             "baseline_cycles": int(len(before)), **cmp})
    res["episodes"] = pd.DataFrame(episodes)

    # --- теплокарта «месяц против остального периода» ------------------------
    heat = []
    for metric, e in series.items():
        rows = st.metric_rows(rates, metric).dropna(subset=["date_opu"])
        rows = rows[rows["x"] <= rows["n"]].assign(month=lambda d: d["date_opu"].dt.to_period("M").astype(str))
        cells = []
        for month, grp in rows.groupby("month"):
            rest = rows[rows["month"] != month]
            if len(grp) < 3 or rest.empty:
                continue
            cmp = st.two_period(grp["x"].sum(), grp["n"].sum(), rest["x"].sum(), rest["n"].sum(), e["phi"])
            cells.append({"metric": metric, "month": month, "cycles": int(len(grp)), **cmp})
        for cell, q in zip(cells, st.bh([c["p"] for c in cells])):
            cell["q"] = q
            sign = 1 if e["direction"] == "lower" else -1
            cell["worse_pp"] = -sign * cell["diff_pp"]       # >0 — хуже остального периода
        heat.extend(cells)
    res["heat"] = pd.DataFrame(heat)

    # --- эмбриологи ----------------------------------------------------------
    emb = {}
    if "embryologist" in rates.columns and rates["embryologist"].notna().any():
        for metric in kpi.EMBRYOLOGIST_METRICS:
            rows = st.metric_rows(rates, metric)
            if rows.empty:
                continue
            emb[metric] = {"test": st.compare_groups(rows, "embryologist"),
                           "funnel": _funnel(rows, "embryologist")}
        age = rates.dropna(subset=["embryologist"])
        if "age" in age.columns and age["age"].notna().sum() > 10:
            samples = [g["age"].dropna().to_numpy() for _, g in age.groupby("embryologist")
                       if g["age"].notna().sum() >= st.MIN_CYCLES_TEST]
            if len(samples) >= 2 and np.unique(np.concatenate(samples)).size > 1:
                from scipy import stats as sps
                h = sps.kruskal(*samples)
                emb["_casemix_age"] = {"test": "Краскел–Уоллис", "p": float(h.pvalue),
                                       "medians": {k: float(g["age"].median()) for k, g in age.groupby("embryologist")
                                                   if g["age"].notna().any()}}
        if "method" in age.columns and age["method"].notna().any():
            ct = pd.crosstab(age["embryologist"], age["method"])
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                from scipy import stats as sps
                emb["_casemix_method"] = {"test": "χ² сопряжённости", "p": float(sps.chi2_contingency(ct)[1])}
    res["embryologists"] = emb

    # --- оплодотворение по методу -------------------------------------------
    fr_method, fr_cycles = st.fr_by_method(rates, settings)
    res["fr_method"], res["fr_cycles"] = fr_method, fr_cycles
    if not fr_cycles.empty and fr_cycles["method"].nunique() >= 2:
        res["fr_method_test"] = st.compare_groups(fr_cycles, "method")
        disp_rows = fr_cycles[fr_cycles["n"] >= st.MIN_DEN_DISPLAY]
        a = disp_rows.loc[disp_rows["method"] == "ICSI", "rate"]
        b = disp_rows.loc[disp_rows["method"] == "IVF", "rate"]
        if len(a) >= 5 and len(b) >= 5:
            from scipy import stats as sps
            ks = sps.ks_2samp(a, b)
            res["fr_ks"] = {"D": float(ks.statistic), "p": float(ks.pvalue), "n_icsi": int(len(a)),
                            "n_ivf": int(len(b))}

    # --- TGBDR ------------------------------------------------------------
    res["correlations"] = st.tgbdr_correlations(rates)
    res["tgbdr_age"] = st.tgbdr_by_age(rates)

    res["facts"], res["checks"] = _facts(res)
    return res


def _funnel(rows: pd.DataFrame, group_col: str) -> dict:
    """Funnel plot Спигелхальтера: доля группы против объёма, пределы вокруг общей доли."""
    r = rows.dropna(subset=[group_col])
    r = r[r["x"] <= r["n"]]
    if r.empty:
        return {}
    p = r["x"].sum() / r["n"].sum()
    phi = st.dispersion(r["x"], r["n"])["phi"]
    g = r.groupby(group_col).agg(x=("x", "sum"), n=("n", "sum"), cycles=("x", "size")).reset_index()
    g["rate"] = g["x"] / g["n"]
    n_grid = np.linspace(max(5, g["n"].min() * 0.5), g["n"].max() * 1.15, 120)
    se = np.sqrt(phi * p * (1 - p) / n_grid)
    return {"p": p, "phi": phi, "groups": g, "n": n_grid,
            "l95": p - 1.96 * se, "u95": p + 1.96 * se, "l998": p - 3.09 * se, "u998": p + 3.09 * se}


def _facts(res: dict) -> tuple[list[str], list[dict]]:
    facts: list[str] = []
    ex = res["exclusions"]
    period = res.get("period")
    if period:
        n = res["n_cycles"]
        facts.append(f"Период: {period['start']} — {period['end']}. В анализ {plural(n, 'вошёл', 'вошли', 'вошли')} "
                     f"{n} {plural(n, 'цикл', 'цикла', 'циклов')} из {ex['rows_total']} строк выгрузки"
                     + (" (остальные — переносы без своей воронки, незакрытые циклы, строки итогов; "
                        "см. «Данные и методы»)." if n < ex["rows_total"] else "."))
    summary: pd.DataFrame = res["kpi"]
    ok = summary[summary["status"].isin(["benchmark", "competency", "outside_control"])]
    groups = {s: ok[ok["status"] == s] for s in ("benchmark", "competency", "outside_control")}
    for status, label in (("benchmark", "На уровне бенчмарка Венского консенсуса"),
                          ("competency", "Между компетентностью и бенчмарком"),
                          ("outside_control", "За порогом компетентности")):
        g = groups[status]
        if not g.empty:
            items = ", ".join(f"{r.metric} {pct(r.rate)}" for r in g.itertuples())
            facts.append(f"{label}: {items}.")
    for r in groups["outside_control"].itertuples():
        comp = r.competency
        lo, hi = r.ci_low_phi, r.ci_high_phi
        overlaps = (hi >= comp) if r.direction == "lower" else (lo <= comp)
        if overlaps:
            facts.append(f"{r.metric}: 95% ДИ ({pct(lo)}–{pct(hi)}) перекрывает порог компетентности "
                         f"{pct(comp, 0)} — отставание по этой выборке статистически не подтверждено.")
    data_notes: list[str] = []
    for r in summary.itertuples():
        if getattr(r, "cycles_over_one", 0):
            if r.metric == "ClR":
                data_notes.append(f"ClR > 100% в {int(r.cycles_over_one)} {plural(int(r.cycles_over_one), 'цикле', 'циклах', 'циклах')} — наблюдение о тайминге проверки "
                             "оплодотворения; ДИ для таких строк не считается.")
            elif r.metric == "MIIR":
                data_notes.append(f"MII > ОКК в {int(r.cycles_over_one)} {plural(int(r.cycles_over_one), 'цикле', 'циклах', 'циклах')} — вероятно, добавлены размороженные "
                             "ооциты; это наблюдение, а не ошибка данных.")
    if res.get("two_pn_zero_cycles"):
        extra = ""
        if "fr_with_zero_2pn" in res:
            extra = f" С ними FR составил бы {pct(res['fr_with_zero_2pn'])}."
        data_notes.append(f"Циклы с 2PN = 0 ({res['two_pn_zero_cycles']}) исключены из KPI по правилу "
                     f"проекта.{extra}")

    n_watch = sum(len(c["events"]) for c in res["cusum"].values() if c.get("available"))
    crossed = sum(int(c["events"]["alert_threshold_crossed"].sum()) for c in res["cusum"].values()
                  if c.get("available") and not c["events"].empty)
    if res["cusum"]:
        parts = []
        for m, c in res["cusum"].items():
            if not c.get("available"):
                continue
            months = sorted({month_ru(d) for d in c["events"]["date"]}, key=lambda t: (t[-4:], MONTHS_RU.index(t[:3])))                 if not c["events"].empty else []
            parts.append(f"{m} — {len(c['events'])}" + (f" ({', '.join(months)})" if months else ""))
        expected = ""
        if res.get("period"):
            span = (pd.Timestamp(res["period"]["end"]) - pd.Timestamp(res["period"]["start"])).days
            per_year = res["settings_used"]["targets_per_year"].get("watch", 24)
            expected = (f" По построению порогов даже в стабильной лаборатории ожидается около "
                        f"{per_year * span / 365:.0f} ложных watch за такой период — watch это строка на "
                        "панели, а не тревога.")
        facts.append(f"CUSUM: {n_watch} {plural(n_watch, 'эпизод', 'эпизода', 'эпизодов')} watch "
                     f"({'; '.join(parts)}); alert в демо-режиме не поднимается."
                     + expected
                     + (f" Порог alert пересекался {crossed} раз(а) — уровень понижен до watch." if crossed else ""))
    emb = res["embryologists"]
    for metric, e in emb.items():
        if metric.startswith("_"):
            continue
        t = e["test"]
        if t.get("test") and np.isfinite(t.get("p", np.nan)) and t["p"] < 0.05:
            pairs = [pw for pw in t.get("pairwise", []) if pw["p_holm"] < 0.05]
            pair_txt = "; ".join(f"{pw['a']} − {pw['b']}: {pw['diff_pp']:+.1f} п.п" for pw in pairs[:3])
            facts.append(f"{metric} различается между эмбриологами ({t['test']}, {pval(t['p'])})."
                         + (f" Значимые пары (Холм): {pair_txt}." if pair_txt else ""))
    if "_casemix_age" in emb and emb["_casemix_age"]["p"] < 0.05:
        facts.append(f"Возраст пациенток различается между эмбриологами ({pval(emb['_casemix_age']['p'])}) — "
                     "различия KPI между ними частично могут объясняться составом пациенток.")

    fm = res.get("fr_method")
    if isinstance(fm, pd.DataFrame) and not fm.empty:
        items = ", ".join(f"{r.method} {pct(r.rate)} ({r.cycles} {plural(r.cycles, 'цикл', 'цикла', 'циклов')})"
                          for r in fm.itertuples())
        facts.append(f"Оплодотворение по методу: {items}.")

    # Эпизоды — после эмбриологов и методов: на одну метрику один, последний значимый.
    # Полный список — в таблице вкладки «Контрольные карты».
    ep = res["episodes"]
    if not ep.empty:
        sig = ep[(ep["segment_cycles"] >= MIN_SEGMENT_CYCLES) & (ep["p"] < 0.05)]
        latest = sig.sort_values("signal_date").groupby("metric").tail(1)
        for r in latest.sort_values("signal_date", ascending=False).itertuples():
            n_sig = int((sig["metric"] == r.metric).sum())
            more = f" Всего подтверждённых эпизодов {r.metric}: {n_sig}." if n_sig > 1 else ""
            facts.append(f"{r.metric}, последний эпизод {pd.Timestamp(r.segment_start):%d.%m} — "
                         f"{pd.Timestamp(r.signal_date):%d.%m.%Y} ({r.segment_cycles} "
                         f"{plural(r.segment_cycles, 'цикл', 'цикла', 'циклов')}): {pct(r.rate_a)} против "
                         f"{pct(r.rate_b)} ({r.baseline}), разность {r.diff_pp:+.1f} п.п. "
                         f"[{r.lo_pp:+.1f}; {r.hi_pp:+.1f}], {pval(r.p)} (ретроспективно, справочно).{more}")
        short = int((ep["segment_cycles"] < MIN_SEGMENT_CYCLES).sum())
        if short:
            facts.append(f"{short} из {len(ep)} эпизодов накопились за 1–2 цикла — это сигнал об отдельных "
                         "пациентках, а не о процессе; тест для них не выполнялся.")

    corr = res["correlations"]
    if isinstance(corr, pd.DataFrame) and not corr.empty:
        top = corr[(corr["target"] == "bl_good") & (corr["q"] < 0.05)].sort_values("rho", key=abs, ascending=False)
        if not top.empty:
            items = ", ".join(f"{r.feature_ru} (ρ = {r.rho:+.2f})" for r in top.head(4).itertuples())
            facts.append(f"С числом бластоцист хорошего качества на цикл сильнее всего связаны: {items} "
                         "(Спирмен, BH-FDR < 0.05; связь, не причина — число почти механически растёт "
                         "с числом ооцитов).")
        rate = corr[(corr["target"] == "TGBDR") & (corr["q"] < 0.05) & ~corr["coupled"].astype(bool)]
        if rate.empty:
            facts.append("С долей TGBDR (качество, очищенное от числа ооцитов) ни один признак не связан "
                         "значимо после поправки на множественность.")
        else:
            items = ", ".join(f"{r.feature_ru} (ρ = {r.rho:+.2f})" for r in
                              rate.sort_values("rho", key=abs, ascending=False).head(3).itertuples())
            facts.append(f"С долей TGBDR значимо связаны: {items}.")

    facts.extend(data_notes)

    # --- что проверить: только тексты консенсуса, только для метрик с поводом ---
    vienna = _vienna_checks()
    reasons: dict[str, list[str]] = {}
    for r in groups["outside_control"].itertuples():
        reasons.setdefault(r.metric, []).append("за порогом компетентности")
    # Текущий эпизод watch: последний сигнал карты — в пределах окна подтверждения
    # (settings.corroboration.window_days) от конца периода. Старые эпизоды — история.
    end = pd.Timestamp(res["period"]["end"]) if res.get("period") else None
    window_days = st.load_settings().get("corroboration", {}).get("window_days", 30)
    for metric, c in res["cusum"].items():
        if end is None or not c.get("available") or c["events"].empty:
            continue
        last = c["events"]["date"].max()
        if (end - last).days <= window_days:
            reasons.setdefault(metric, []).append(f"текущий эпизод CUSUM watch ({last:%d.%m.%Y})")
    checks = []
    for metric, why in reasons.items():
        info = vienna.get(metric, {})
        checks.append({"metric": metric, "why": "; ".join(why), "contour": info.get("contour", ""),
                       "items": info.get("on_signal_check", [])})
    return facts, checks
