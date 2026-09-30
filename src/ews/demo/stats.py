"""Статистика демо-панели: всё считает Python, модель только объясняет.

Правила проекта, которые здесь держатся явно:
  * когортная доля — только Σ числитель / Σ знаменатель, никогда не среднее долей;
  * пороги — только из ``KPI_SPECS`` и ``config/consensus/*.yaml``;
  * ``числитель > знаменатель`` (ClR>100%, MII>OCC) — наблюдение, не порча данных:
    доля остаётся, биномиальный ДИ не считается, в биномиальные карты строка не идёт;
  * CUSUM и EWMA одной метрики — одно измерение, не два подтверждения;
  * демо-прогон не поднимает ``alert``: пересечение порога alert понижается до watch.

Между-цикловая сверхбиномиальная вариация у данных ЭКО велика (X²/df 1.8–2.8 в
публикации), поэтому все ДИ и тесты для объединённых долей несут поправку на
дисперсию φ (квазибиномиальная модель), а не голую биномиальную ошибку.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .. import charts
from ..vendor import kpi
from .ingest import fr_by_method_denominator

ROOT = Path(__file__).resolve().parents[3]
Z95 = 1.959963984540054

# Минимальный знаменатель, при котором ДОЛЯ ЦИКЛА показывается на распределениях
# (violin, ECDF, корреляции). Доля 1/1 или 0/2 — вырожденная точка, она рисует
# «пики» в 0 и 1, которых у процесса нет. На сводные Σ/Σ это не влияет.
MIN_DEN_DISPLAY = 3
# Минимум циклов в группе, чтобы группа участвовала в тесте различий.
MIN_CYCLES_TEST = 5

METRIC_RU = {
    "OCCR": "Выход ОКК (OCCR)", "MIIR": "Зрелость ооцитов (MIIR)",
    "FR": "Оплодотворение (FR)", "ClR": "Дробление (ClR)",
    "BlR": "Бластоцисты (BlR)", "TGBDR": "Бластоцисты хорошего качества (TGBDR)",
    "DegR": "Дегенерация ооцитов (DegR)", "3pNR": "3PN (3pNR)",
}
METRIC_SHORT = {"OCCR": "ОКК / фолликулы", "MIIR": "MII / ОКК", "FR": "2PN / MII (инъецир.)",
                "ClR": "дробящиеся D3 / 2PN", "BlR": "бластоцисты / 2PN",
                "TGBDR": "хорошие бластоцисты / 2PN", "DegR": "дегенерировавшие / MII",
                "3pNR": "3PN / ОКК"}
DISPLAY_ORDER = ("OCCR", "MIIR", "FR", "ClR", "BlR", "TGBDR", "DegR", "3pNR")


def load_settings() -> dict:
    with (ROOT / "config" / "settings.yaml").open(encoding="utf-8") as fh:
        settings = yaml.safe_load(fh) or {}
    with (ROOT / "config" / "consensus" / "vienna_fertilization.yaml").open(encoding="utf-8") as fh:
        settings["_fr_by_method"] = (yaml.safe_load(fh) or {}).get("fertilization_by_method", {})
    return settings


# --------------------------------------------------------------------------
# базовые оценки
# --------------------------------------------------------------------------
def wilson(x: float, n: float, z: float = Z95) -> tuple[float, float]:
    """Wilson; при x>n (наблюдение ClR>100%) интервал не определён -> NaN."""
    return kpi.wilson_interval(x, n, z)


def dispersion(x: np.ndarray, n: np.ndarray) -> dict:
    """Pearson X²/df и внутриклассовая ρ между циклами (как в публикации)."""
    x, n = np.asarray(x, float), np.asarray(n, float)
    ok = (n > 0) & (x <= n)
    x, n = x[ok], n[ok]
    if len(x) < 3 or n.sum() == 0:
        return {"phi": 1.0, "rho": 0.0, "n_cycles": int(len(x))}
    p = x.sum() / n.sum()
    if p <= 0 or p >= 1:
        return {"phi": 1.0, "rho": 0.0, "n_cycles": int(len(x))}
    chi2 = (((x - n * p) ** 2) / (n * p * (1 - p))).sum()
    ratio = chi2 / max(len(x) - 1, 1)
    rho = max(0.0, (ratio - 1) / max(n.mean() - 1, 1e-6))
    return {"phi": float(max(ratio, 1.0)), "phi_raw": float(ratio), "rho": float(min(rho, 0.9)),
            "n_cycles": int(len(x))}


def wilson_phi(x: float, n: float, phi: float) -> tuple[float, float]:
    """Wilson по эффективному объёму n/φ — ДИ с поправкой на сверхдисперсию."""
    if not np.isfinite(n) or n <= 0 or x > n:
        return np.nan, np.nan
    n_eff = n / max(phi, 1.0)
    return wilson(x / n * n_eff, n_eff)


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out.tolist()


# --------------------------------------------------------------------------
# построчные наблюдения метрики
# --------------------------------------------------------------------------
@dataclass
class MetricSpec:
    key: str
    numerator: str
    denominator: str
    direction: str
    competency: float | None
    benchmark: float | None
    label: str
    source_note: str
    indicator_class: str


def spec(metric: str) -> MetricSpec:
    s = kpi.KPI_SPECS[metric]
    return MetricSpec(metric, s.numerator, s.denominator, s.direction, s.competency, s.benchmark,
                      s.label, s.source_note, s.indicator_class)


def cycle_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Построчные доли по kpi.compute_rates без фильтра 2PN (он применяется по метрике)."""
    rates = kpi.compute_rates(frame, drop_without_two_pn=False)
    rates["_fr_method_den"] = fr_by_method_denominator(rates)
    return rates


def metric_rows(rates: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Наблюдения метрики: одна строка — цикл, x/n/rate + контекст.

    Эмбриологические метрики — только циклы с 2PN>0 (стоящее правило владельца,
    kpi.compute_rates); FR и DegR — без хирургической спермы; врачебные (OCCR/MIIR)
    — все циклы пункции, включая «только пункцию».
    """
    s = spec(metric)
    d = rates
    if metric in kpi.EMBRYOLOGIST_METRICS:
        if "two_pn" not in d.columns:
            return pd.DataFrame(columns=["cycle_uid", "x", "n", "rate"])
        two_pn = pd.to_numeric(d["two_pn"], errors="coerce")
        d = d[two_pn.notna() & two_pn.ne(0)]
        if "retrieval_only" in d.columns:
            d = d[~d["retrieval_only"].astype(bool)]
    if metric in ("FR", "DegR") and "surgical_sperm" in d.columns:
        d = d[~d["surgical_sperm"].astype(bool)]
    if s.numerator not in d.columns or s.denominator not in d.columns:
        return pd.DataFrame(columns=["cycle_uid", "x", "n", "rate"])
    x = pd.to_numeric(d[s.numerator], errors="coerce")
    n = pd.to_numeric(d[s.denominator], errors="coerce")
    ok = x.notna() & n.notna() & (n > 0)
    keep = [c for c in ("cycle_uid", "date_opu", "embryologist", "doctor", "method", "age")
            if c in d.columns]
    out = d.loc[ok, keep].copy()
    out["x"], out["n"] = x[ok].astype(float), n[ok].astype(float)
    out["rate"] = out["x"] / out["n"]
    out["over_one"] = out["x"] > out["n"]
    return out


def pooled(rows: pd.DataFrame, phi: float = 1.0) -> dict:
    if rows.empty:
        return {"x": np.nan, "n": np.nan, "rate": np.nan, "lo": np.nan, "hi": np.nan, "cycles": 0}
    x, n = float(rows["x"].sum()), float(rows["n"].sum())
    lo, hi = wilson_phi(x, n, phi) if phi > 1 else wilson(x, n)
    return {"x": x, "n": n, "rate": x / n if n > 0 else np.nan, "lo": lo, "hi": hi,
            "cycles": int(len(rows))}


def status_of(rate: float, s: MetricSpec | dict) -> str:
    comp = s.competency if isinstance(s, MetricSpec) else s.get("competency")
    bench = s.benchmark if isinstance(s, MetricSpec) else s.get("benchmark")
    direction = s.direction if isinstance(s, MetricSpec) else s.get("direction", "lower")
    if rate is None or not np.isfinite(rate):
        return "insufficient_data"
    if bench is None and comp is None:
        return "no_threshold"
    ok = (lambda lim: rate >= lim) if direction == "lower" else (lambda lim: rate <= lim)
    if bench is not None and ok(bench):
        return "benchmark"
    if comp is not None and ok(comp):
        return "competency"
    return "outside_control"


STATUS_RU = {"benchmark": "уровень бенчмарка", "competency": "уровень компетентности",
             "outside_control": "ниже компетентности", "insufficient_data": "мало данных",
             "no_threshold": "порог не задан"}


# --------------------------------------------------------------------------
# сводка KPI
# --------------------------------------------------------------------------
def kpi_summary(rates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in DISPLAY_ORDER:
        s = spec(metric)
        r = metric_rows(rates, metric)
        if r.empty:
            rows.append({"metric": metric, "label_ru": METRIC_RU[metric], "cycles": 0,
                         "status": "insufficient_data"})
            continue
        disp = dispersion(r["x"], r["n"])
        p = pooled(r)
        lo_phi, hi_phi = wilson_phi(p["x"], p["n"], disp["phi"])
        rows.append({
            "metric": metric, "label_ru": METRIC_RU[metric], "definition": METRIC_SHORT[metric],
            "numerator": p["x"], "denominator": p["n"], "rate": p["rate"],
            "ci_low": p["lo"], "ci_high": p["hi"], "ci_low_phi": lo_phi, "ci_high_phi": hi_phi,
            "phi": disp["phi"], "rho": disp["rho"], "cycles": p["cycles"],
            "cycles_over_one": int(r["over_one"].sum()),
            "direction": s.direction, "competency": s.competency, "benchmark": s.benchmark,
            "status": status_of(p["rate"], s), "indicator_class": s.indicator_class,
            "source_note": s.source_note,
        })
    return pd.DataFrame(rows)


def fr_by_method(rates: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FR по методу с венскими порогами метода; + построчные доли для violin/ECDF."""
    d = rates.copy()
    if "two_pn" not in d.columns or "method" not in d.columns:
        return pd.DataFrame(), pd.DataFrame()
    two_pn = pd.to_numeric(d["two_pn"], errors="coerce")
    d = d[two_pn.notna() & two_pn.ne(0)]
    if "surgical_sperm" in d.columns:
        d = d[~d["surgical_sperm"].astype(bool)]
    if "retrieval_only" in d.columns:
        d = d[~d["retrieval_only"].astype(bool)]
    d = d[d["method"].isin(["ICSI", "IVF", "IVF+ICSI"]) & (d["_fr_method_den"] > 0)]
    per_cycle = pd.DataFrame({"cycle_uid": d.get("cycle_uid"), "method": d["method"],
                              "x": pd.to_numeric(d["two_pn"], errors="coerce"),
                              "n": d["_fr_method_den"], "embryologist": d.get("embryologist"),
                              "date_opu": d.get("date_opu")})
    per_cycle["rate"] = per_cycle["x"] / per_cycle["n"]
    thresholds = settings.get("_fr_by_method", {})
    rows = []
    for method, grp in per_cycle.groupby("method"):
        disp = dispersion(grp["x"], grp["n"])
        p = pooled(grp)
        lo, hi = wilson_phi(p["x"], p["n"], disp["phi"])
        th = thresholds.get(method, {})
        rows.append({"method": method, "cycles": p["cycles"], "numerator": p["x"],
                     "denominator": p["n"], "rate": p["rate"], "ci_low_phi": lo, "ci_high_phi": hi,
                     "phi": disp["phi"], "competency": th.get("competency"),
                     "benchmark": th.get("benchmark"), "source_note": th.get("source_note", ""),
                     "status": status_of(p["rate"], {"competency": th.get("competency"),
                                                     "benchmark": th.get("benchmark"),
                                                     "direction": "lower"})})
    return pd.DataFrame(rows), per_cycle


# --------------------------------------------------------------------------
# сравнение групп: подбор метода
# --------------------------------------------------------------------------
def compare_groups(rows: pd.DataFrame, group_col: str, *, min_cycles: int = MIN_CYCLES_TEST) -> dict:
    """Различие долей между группами (эмбриологи, месяцы, методы).

    Выбор метода — по данным, и он печатается вместе с результатом:
      * единица — эмбрион, но эмбрионы одного цикла не независимы (φ>1), поэтому
        основной тест — квазибиномиальный: X² однородности k×2, делённый на φ,
        сравнивается с F(k−1, N−k). При φ≤1 это обычный χ²;
      * две группы, ожидаемые частоты <5 и нет сверхдисперсии — точный тест Фишера;
      * проверка устойчивости на уровне циклов — Краскел–Уоллис (k≥3) или
        Манн–Уитни (k=2) по долям циклов со знаменателем ≥ MIN_DEN_DISPLAY;
      * попарные различия — только если омнибус значим, с поправкой Холма.
    """
    r = rows.dropna(subset=[group_col])
    r = r[r["x"] <= r["n"]]
    counts = r.groupby(group_col).size()
    small = counts[counts < min_cycles].index.tolist()
    r = r[~r[group_col].isin(small)]
    groups = sorted(r[group_col].unique().tolist(), key=str)
    result: dict = {"group_col": group_col, "groups": groups, "excluded_small": small,
                    "k": len(groups)}
    if len(groups) < 2:
        result.update({"test": None, "p": np.nan, "why": "меньше двух групп с ≥"
                       f"{min_cycles} циклами — сравнивать нечего"})
        return result

    g = r.groupby(group_col).agg(x=("x", "sum"), n=("n", "sum"), cycles=("x", "size"))
    p_all = g["x"].sum() / g["n"].sum()
    # φ внутри групп: Pearson по циклам вокруг доли СВОЕЙ группы
    p_grp = r[group_col].map(g["x"] / g["n"])
    denom = r["n"] * p_grp * (1 - p_grp)
    valid = denom > 0
    pearson = (((r["x"] - r["n"] * p_grp) ** 2)[valid] / denom[valid]).sum()
    df_res = max(int(valid.sum()) - len(groups), 1)
    phi = max(pearson / df_res, 1.0)

    table = np.column_stack([g["x"], g["n"] - g["x"]])
    expected = np.outer(table.sum(1), table.sum(0)) / table.sum()
    chi2 = float((((table - expected) ** 2) / np.where(expected > 0, expected, 1)).sum())
    k = len(groups)
    if k == 2 and (expected < 5).any() and phi <= 1.1:
        _, p = stats.fisher_exact(table)
        test = "точный тест Фишера"
        why = "две группы, ожидаемые частоты <5"
        stat_value = np.nan
    elif phi > 1.0:
        f_stat = chi2 / (k - 1) / phi
        p = float(stats.f.sf(f_stat, k - 1, df_res))
        test = "квазибиномиальный F-тест (χ² однородности / φ)"
        why = (f"эмбрионы одного цикла зависимы: сверхдисперсия φ = {phi:.2f}, "
               "обычный χ² занизил бы p в ~φ раз")
        stat_value = f_stat
    else:
        p = float(stats.chi2.sf(chi2, k - 1))
        test = "χ² однородности долей"
        why = "сверхдисперсии нет (φ ≤ 1), биномиальная модель адекватна"
        stat_value = chi2

    # проверка устойчивости на уровне циклов
    rr = r[r["n"] >= MIN_DEN_DISPLAY]
    samples = [rr.loc[rr[group_col] == grp, "rate"].to_numpy() for grp in groups]
    samples_ok = [s for s in samples if len(s) >= 3]
    if samples_ok and np.unique(np.concatenate(samples_ok)).size < 2:
        samples_ok = []                        # все доли одинаковы (напр. ClR = 100%) — ранги пусты
    if len(samples_ok) == 2:
        u = stats.mannwhitneyu(*samples_ok, alternative="two-sided")
        n1, n2 = map(len, samples_ok)
        effect = 1 - 2 * u.statistic / (n1 * n2)
        robust = {"test": "Манн–Уитни (доли циклов)", "p": float(u.pvalue),
                  "effect": float(effect), "effect_name": "ранго-бисериальная r"}
    elif len(samples_ok) >= 3:
        h = stats.kruskal(*samples_ok)
        n_tot = sum(map(len, samples_ok))
        eps2 = float(h.statistic / ((n_tot ** 2 - 1) / (n_tot + 1)))
        robust = {"test": "Краскел–Уоллис (доли циклов)", "p": float(h.pvalue),
                  "effect": eps2, "effect_name": "ε²"}
    else:
        robust = {"test": None, "p": np.nan, "why": "доли циклов не различаются или мало циклов"}

    pairwise = []
    if k >= 2 and p < 0.05:
        pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]]
        raw = []
        for a, b in pairs:
            x1, n1 = g.loc[a, "x"], g.loc[a, "n"]
            x2, n2 = g.loc[b, "x"], g.loc[b, "n"]
            p1, p2 = x1 / n1, x2 / n2
            pp = (x1 + x2) / (n1 + n2)
            se0 = sqrt(phi * pp * (1 - pp) * (1 / n1 + 1 / n2)) or 1e-9
            se1 = sqrt(phi * (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
            zval = (p1 - p2) / se0
            raw.append(2 * stats.norm.sf(abs(zval)))
            pairwise.append({"a": a, "b": b, "diff_pp": 100 * (p1 - p2),
                             "lo_pp": 100 * (p1 - p2 - Z95 * se1), "hi_pp": 100 * (p1 - p2 + Z95 * se1)})
        for item, adj in zip(pairwise, holm(raw)):
            item["p_holm"] = adj

    per_group = [{"group": grp, "cycles": int(g.loc[grp, "cycles"]), "x": float(g.loc[grp, "x"]),
                  "n": float(g.loc[grp, "n"]), "rate": float(g.loc[grp, "x"] / g.loc[grp, "n"]),
                  **dict(zip(("lo", "hi"), wilson_phi(g.loc[grp, "x"], g.loc[grp, "n"], phi)))}
                 for grp in groups]
    result.update({"test": test, "why": why, "statistic": stat_value, "p": float(p), "phi": float(phi),
                   "pooled": float(p_all), "robust": robust, "pairwise": pairwise,
                   "per_group": per_group})
    return result


def two_period(x1: float, n1: float, x2: float, n2: float, phi: float) -> dict:
    """Период против сравнения: разность в п.п. с ДИ и p.

    Точный тест Фишера считает эмбрионы независимыми; при заметной сверхдисперсии
    (φ > 1.1) это завышало бы значимость в ~φ раз, поэтому тогда используется
    z-тест с дисперсией, умноженной на φ, даже при малых частотах (с пометкой
    «приближённо»). Фишер — только когда φ ≈ 1 и частоты малы.
    """
    p1, p2 = x1 / n1, x2 / n2
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    expected = np.outer(table.sum(1), table.sum(0)) / table.sum()
    se1 = sqrt(phi * (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
    small = bool((expected < 5).any())
    if small and phi <= 1.1:
        _, p = stats.fisher_exact(table)
        test = "точный тест Фишера"
    else:
        pp = (x1 + x2) / (n1 + n2)
        se0 = sqrt(phi * pp * (1 - pp) * (1 / n1 + 1 / n2)) or 1e-9
        p = float(2 * stats.norm.sf(abs((p1 - p2) / se0)))
        test = "z-тест долей с поправкой на φ" if phi > 1 else "z-тест долей"
        if small:
            test += " (приближённо: малые частоты)"
    return {"rate_a": p1, "rate_b": p2, "diff_pp": 100 * (p1 - p2),
            "lo_pp": 100 * (p1 - p2 - Z95 * se1), "hi_pp": 100 * (p1 - p2 + Z95 * se1),
            "p": float(p), "test": test}


# --------------------------------------------------------------------------
# временные ряды
# --------------------------------------------------------------------------
def rolling_pooled(rows: pd.DataFrame, window: int, phi: float) -> pd.DataFrame:
    """Скользящая доля последних ``window`` циклов (Σx/Σn) и ДИ с поправкой на φ."""
    r = rows.dropna(subset=["date_opu"]).sort_values("date_opu", kind="stable")
    r = r[r["x"] <= r["n"]]
    if len(r) < window:
        return pd.DataFrame()
    xs = r["x"].rolling(window).sum()
    ns = r["n"].rolling(window).sum()
    out = pd.DataFrame({"date": r["date_opu"], "rate": xs / ns, "x": xs, "n": ns}).dropna()
    ci = [wilson_phi(a, b, phi) for a, b in zip(out["x"], out["n"])]
    out["lo"], out["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return out


def monthly(rows: pd.DataFrame, phi: float) -> pd.DataFrame:
    r = rows.dropna(subset=["date_opu"])
    r = r[r["x"] <= r["n"]]
    if r.empty:
        return pd.DataFrame()
    g = r.groupby(r["date_opu"].dt.to_period("M")).agg(x=("x", "sum"), n=("n", "sum"),
                                                        cycles=("x", "size"))
    g["rate"] = g["x"] / g["n"]
    ci = [wilson_phi(a, b, phi) for a, b in zip(g["x"], g["n"])]
    g["lo"], g["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    g.index = g.index.astype(str)
    return g.reset_index(names="month")


def laney_pchart(rows: pd.DataFrame, p0: float) -> tuple[pd.DataFrame, dict]:
    """Месячная p-карта Шухарта: биномиальные 3σ и Laney p′ (как Figure 7C)."""
    g = monthly(rows, 1.0)
    if g.empty or not (0 < p0 < 1):
        return pd.DataFrame(), {}
    se = np.sqrt(p0 * (1 - p0) / g["n"])
    zi = (g["rate"] - p0) / se
    mr = np.abs(np.diff(zi.to_numpy()))
    sigma_z = float(mr.mean() / 1.128) if len(mr) else 1.0
    g["center"] = p0
    g["lcl_binom"], g["ucl_binom"] = p0 - 3 * se, p0 + 3 * se
    g["lcl_laney"], g["ucl_laney"] = p0 - 3 * se * sigma_z, p0 + 3 * se * sigma_z
    g["out_binom"] = (g["rate"] < g["lcl_binom"]) | (g["rate"] > g["ucl_binom"])
    g["out_laney"] = (g["rate"] < g["lcl_laney"]) | (g["rate"] > g["ucl_laney"])
    return g, {"sigma_z": sigma_z, "months": int(len(g)),
               "outside_binomial": int(g["out_binom"].sum()), "outside_laney": int(g["out_laney"].sum())}


def ewma(rows: pd.DataFrame, p0: float, rho: float, lam: float = 0.2, L: float = 3.0) -> pd.DataFrame:
    """EWMA по долям циклов, пределы расширены на сверхдисперсию (как Figure 7B)."""
    r = rows.dropna(subset=["date_opu"]).sort_values("date_opu", kind="stable")
    r = r[r["x"] <= r["n"]]
    z, var, out = p0, 0.0, []
    for _, row in r.iterrows():
        n = row["n"]
        s2 = p0 * (1 - p0) / n * (1 + (n - 1) * rho)
        z = lam * (row["x"] / n) + (1 - lam) * z
        var = lam ** 2 * s2 + (1 - lam) ** 2 * var
        out.append({"date": row["date_opu"], "ewma": z, "lcl": p0 - L * sqrt(var),
                    "ucl": p0 + L * sqrt(var), "center": p0})
    return pd.DataFrame(out)


def cusum_replay(rows: pd.DataFrame, p0: float, rho: float, direction: str, settings: dict,
                 n_charts: int, rng: np.random.Generator, replicates: int = 300) -> dict:
    """Демо-калибровка порогов и хронологический прогон бернуллиевского CUSUM.

    Пороги подбираются тем же кодом, что в бэкфите (charts.solve_thresholds) — под
    совместную цель ложных сигналов из settings.yaml, делённую на число карт. Но
    калибруются на ТОЙ ЖЕ выгрузке, которую затем оценивают: это демонстрация
    метода, а не боевая калибровка. Поэтому пересечение h_alert понижается до
    watch (гейт alert закрыт, как при calibrated=false).
    """
    cfg = settings.get("charts", {}).get("cusum", {})
    odds = cfg.get("odds_ratio_deterioration", 0.5) if direction == "lower" \
        else cfg.get("odds_ratio_upper_control", 2.0)
    targets = settings.get("thresholds", {}).get("target_false_alarms_per_year", {})
    r = rows.dropna(subset=["date_opu"]).sort_values("date_opu", kind="stable")
    r = r[r["x"] <= r["n"]]
    if len(r) < 20 or not (0 < p0 < 1):
        return {"available": False, "reason": "меньше 20 циклов с датой — карта не строится"}
    span_days = max((r["date_opu"].max() - r["date_opu"].min()).days, 1)
    if span_days < 60:
        return {"available": False, "reason": "период короче 60 дней — годовую частоту ложных "
                "сигналов не на чем оценить"}
    per_year = r["n"].sum() * 365.0 / span_days
    solved = charts.solve_thresholds(
        charts.jeffreys_sampler(p0 * r["n"].sum(), r["n"].sum()), p0, odds,
        targets.get("watch", 24) / n_charts, targets.get("alert", 2) / n_charts,
        horizon=max(int(round(per_year)), 50), replicates=replicates, rng=rng,
        block_sizes=r["n"].astype(int).to_numpy(), dispersion=rho)
    runner = charts.CusumRunner()
    trace, events = [], []
    last_zero_date = r["date_opu"].iloc[0]
    for _, row in r.iterrows():
        before_zero = runner.statistic == 0.0
        if before_zero:
            last_zero_date = row["date_opu"]
        event = runner.step_block(row["x"], row["n"], p0, odds, solved["h_watch"], solved["h_alert"])
        trace.append({"date": row["date_opu"], "statistic": runner.statistic
                      if event != "alert" else solved["h_alert"]})
        if event:
            events.append({"date": row["date_opu"], "level": "watch",
                           "alert_threshold_crossed": event == "alert",
                           "segment_start": last_zero_date})
        if runner.statistic == 0.0:
            last_zero_date = row["date_opu"]
    return {"available": True, "odds_ratio": odds, "h_watch": solved["h_watch"],
            "h_alert": solved["h_alert"], "sim_rates": solved["rates"], "per_year": per_year,
            "trace": pd.DataFrame(trace), "events": pd.DataFrame(events), "p0": p0}


# --------------------------------------------------------------------------
# корреляции с выходом бластоцист хорошего качества
# --------------------------------------------------------------------------
FEATURES_RU = {"age": "Возраст", "amh": "АМГ", "afc": "КАФ", "bmi": "ИМТ", "fsh": "ФСГ",
               "attempt_no": "№ попытки", "follicles": "Фолликулы", "occ": "ОКК", "mii": "MII",
               "two_pn": "2PN", "MIIR": "Зрелость (MIIR)", "FR": "Оплодотворение (FR)",
               "ClR": "Дробление (ClR)"}
# Признаки, математически сопряжённые с целевой долей (общий знаменатель 2PN).
COUPLED = {"TGBDR": {"two_pn", "ClR"}, "bl_good": set()}


def spearman_ci(x: pd.Series, y: pd.Series) -> dict:
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    if n < 10 or x[ok].nunique() < 3 or y[ok].nunique() < 3:
        return {"rho": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": n}
    rho, p = stats.spearmanr(x[ok], y[ok])
    se = sqrt((1 + rho ** 2 / 2) / (n - 3))          # Bonett & Wright (2000)
    zr = np.arctanh(np.clip(rho, -0.999, 0.999))
    return {"rho": float(rho), "lo": float(np.tanh(zr - Z95 * se)), "hi": float(np.tanh(zr + Z95 * se)),
            "p": float(p), "n": n}


def tgbdr_correlations(rates: pd.DataFrame) -> pd.DataFrame:
    d = rates.copy()
    if "two_pn" not in d.columns or "bl_good" not in d.columns:
        return pd.DataFrame()
    two_pn = pd.to_numeric(d["two_pn"], errors="coerce")
    d = d[two_pn.notna() & (two_pn > 0) & d["bl_good"].notna()]
    targets = {"TGBDR": d["TGBDR"].where(two_pn.reindex(d.index) >= MIN_DEN_DISPLAY),
               "bl_good": pd.to_numeric(d["bl_good"], errors="coerce")}
    rows = []
    for tkey, target in targets.items():
        for feat, ru in FEATURES_RU.items():
            if feat not in d.columns:
                continue
            values = pd.to_numeric(d[feat], errors="coerce")
            if feat in ("MIIR", "FR", "ClR"):
                values = values.where(values <= 1.0)          # ClR>100%: наблюдение, не в ранги
            res = spearman_ci(values, target)
            if np.isnan(res["rho"]):
                continue
            rows.append({"target": tkey, "feature": feat, "feature_ru": ru,
                         "coupled": feat in COUPLED.get(tkey, set()), **res})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = np.nan
        for tkey, grp in out.groupby("target"):
            out.loc[grp.index, "q"] = bh(grp["p"].tolist())
    return out


def tgbdr_by_age(rates: pd.DataFrame) -> pd.DataFrame:
    r = metric_rows(rates, "TGBDR")
    if "age" not in r.columns or r["age"].isna().all():
        return pd.DataFrame()
    bins = [0, 34.999, 37.999, 40.999, 200]
    labels = ["<35", "35–37", "38–40", ">40"]
    r = r[r["x"] <= r["n"]].assign(age_group=pd.cut(r["age"], bins=bins, labels=labels))
    phi = dispersion(r["x"], r["n"])["phi"]
    out = []
    for grp, g in r.groupby("age_group", observed=True):
        p = pooled(g, phi)
        out.append({"age_group": str(grp), **p})
    return pd.DataFrame(out)
