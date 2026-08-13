"""Опорные измерения EWS: базовые KPI лаборатории и калибровка конверта.

    python scripts/ews_baseline.py [--clinic-base "Two-Line IVF"] [--out data/baseline]

Два независимых измерения, которые НЕЛЬЗЯ смешивать:

1. **Базовые KPI лаборатории** — считаются ТОЛЬКО по своей клинике. BlR соседней
   клиники не является нашим BlR ни в каком приближении.
2. **Калибровка конверта** (покрытие P10-P90, разброс остатков) — свойство МОДЕЛИ,
   а не лаборатории, поэтому измеряется по всем доступным клиникам сразу, с клиникой
   как страта. Однородность проверяется, а не предполагается.

Оба результата — вход фазы 1 (подбор порогов), а не сами пороги.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import teable_client  # noqa: E402
from ews.clinic import Clinic, load_clinic, to_canonical  # noqa: E402
from ews.config import ConfigError, embryo_lab_data_on_path, load  # noqa: E402

# Стадии в таблице Regression QC базы Digital Twin.
STAGES = ("MII", "2PN", "Бласт", "Хор.бласт", "Витрифицировано")

# Ниже этой ширины интервал P10-P90 непригоден как опорный: на счётной величине
# полоса шириной <= 1 физически не может дать 80% покрытия — дискретность счёта
# превышает разрешение интервала. Такие наблюдения помечаются вырожденными и в
# слой остатков не идут. Это ограничение интервала, НЕ сигнал лаборатории.
MIN_INTERVAL_WIDTH = 1.0



# ---------------------------------------------------------------------------
# 1. Базовые KPI лаборатории
# ---------------------------------------------------------------------------

def load_lab_population(cfg, clinic: Clinic,
                        export_path: Path) -> tuple[pd.DataFrame, int]:
    """Широкая lab-популяция минус исключения таблиц компетентности.

    Исключаются: цикл ещё идёт на момент выгрузки, NO-retrieval, freeze-all —
    ровно то же, что делает /run-analysis соседнего проекта. Колонки резолвятся
    через профиль клиники, а не по фиксированным именам: на другой лаборатории
    выгрузка называет их иначе.
    """
    import yaml
    from ews.vendor import kpi  # noqa: PLC0415

    frame = pd.read_csv(export_path, dtype=str, keep_default_na=False)
    filled = pd.Series(False, index=frame.index)
    for canonical in ("patient_id", "patient_name"):
        column = clinic.source_column(canonical)
        if column and column in frame.columns:
            filled |= frame[column].astype(str).str.strip() != ""
    frame = frame.loc[filled].copy()

    data, missing = to_canonical(clinic, frame)
    if missing:
        print(f"[warn] колонок нет в выгрузке: {', '.join(missing)} — "
              f"метрики, которым они нужны, будут недоступны")

    project_dir = Path(cfg.external("embryo_lab_data")["project_dir"])
    with (project_dir / "config" / "columns.yaml").open(encoding="utf-8") as fh:
        columns_cfg = yaml.safe_load(fh)

    # cycle_uid считается тем же составным ключом и тем же HMAC, что при
    # синхронизации в Teable — иначе конверт не сойдётся с циклом.
    from ews.vendor.privacy import load_or_create_secret, make_cycle_uid  # noqa: PLC0415
    from ews.clinic import cycle_key  # noqa: PLC0415

    secret = load_or_create_secret(cfg.path(cfg.settings["privacy"]["secret_file"]))
    keys = data.apply(lambda row: cycle_key(clinic, row), axis=1)
    data["cycle_uid"] = [make_cycle_uid(k, secret) if k else None for k in keys]

    excluded = kpi.competency_exclusion_mask(data, columns_cfg)
    return data.loc[~excluded].copy(), int(excluded.sum())


def kpi_table(population: pd.DataFrame) -> pd.DataFrame:
    from ews.vendor import kpi  # noqa: PLC0415

    if population.empty:
        return pd.DataFrame()
    embryologist = kpi.aggregate_kpis(population, metrics=kpi.EMBRYOLOGIST_METRICS)
    # Врачебная таблица сохраняет строки с ПУСТЫМ 2PN: OCCR/MIIR считаются выше
    # стадии оплодотворения, и пункция без инсеминации — полное их наблюдение.
    doctor = kpi.aggregate_kpis(
        kpi.compute_rates(population, drop_without_two_pn=False),
        metrics=kpi.DOCTOR_METRICS,
    )
    return pd.concat([doctor, embryologist], ignore_index=True)


def lab_kpi_baseline(cfg, clinic: Clinic, export_path: Path) -> pd.DataFrame:
    population, excluded_n = load_lab_population(cfg, clinic, export_path)
    out = kpi_table(population)
    out.insert(0, "population_n", len(population))
    out.insert(0, "excluded_n", excluded_n)
    return out


def write_incontrol(cfg, clinic: Clinic, export_path: Path, out_dir: Path) -> None:
    """In-control параметры опорного окна + оговорки о поглощённых сдвигах."""
    from ews.incontrol import as_manifest, build_baselines, caveats  # noqa: PLC0415

    window_cfg = clinic.reference_window
    start = window_cfg["start"]
    population, _ = load_lab_population(cfg, clinic, export_path)
    start_ts, end_ts = clinic.window_bounds()

    in_window = (population["date_opu"] >= start_ts) & (population["date_opu"] <= end_ts)
    window = population.loc[in_window]
    prior = population.loc[population["date_opu"] < start_ts]

    baselines = build_baselines(kpi_table(window),
                               kpi_table(prior) if not prior.empty else None)
    manifest = as_manifest(
        baselines, window_start=start, window_end=window_cfg.get("end"),
        n_cycles=len(window), n_embryos=float(window["two_pn"].sum()),
        declared_by=window_cfg.get("declared_by", "owner"),
    )
    manifest["clinic"] = {"code": clinic.code, "name": clinic.name}
    manifest["declared_shifts"] = clinic.declared_shifts
    # Эмпирическое распределение размеров цикла на метрику. Нужно калибровке:
    # карта двигается поцикловыми блоками, и от их размеров заметно зависит
    # частота срабатываний. Считать по средней величине нельзя — распределение
    # скошено, а редкие крупные циклы дают самые большие скачки статистики.
    from ews.cycles import METRIC_PARTS  # noqa: PLC0415

    block_sizes: dict[str, list[int]] = {}
    for metric, (_, denominator_col) in METRIC_PARTS.items():
        if denominator_col not in window.columns:
            continue
        values = pd.to_numeric(window[denominator_col], errors="coerce").dropna()
        sizes = [int(v) for v in values if v > 0]
        if sizes:
            block_sizes[metric] = sizes
    manifest["block_sizes"] = block_sizes

    # Сверхбиномиальная вариация между циклами (разнородность пациенток). Без неё
    # калибровка предсказывает втрое меньше срабатываний, чем даёт реальность.
    dispersion: dict[str, dict] = {}
    for metric, (numerator_col, denominator_col) in METRIC_PARTS.items():
        if numerator_col not in window.columns or denominator_col not in window.columns:
            continue
        pair = window[[numerator_col, denominator_col]].dropna()
        pair = pair[(pair[denominator_col] > 0)
                    & (pair[numerator_col] <= pair[denominator_col])]
        if len(pair) < 5:
            continue
        x = pair[numerator_col].to_numpy(dtype=float)
        n = pair[denominator_col].to_numpy(dtype=float)
        p_hat = x.sum() / n.sum()
        if not 0 < p_hat < 1:
            continue
        chi2 = (((x - n * p_hat) ** 2) / (n * p_hat * (1 - p_hat))).sum()
        ratio = chi2 / max(len(pair) - 1, 1)
        # Var = n p (1-p) [1 + (n-1) rho]  ->  rho из отношения Пирсона.
        rho = max(0.0, (ratio - 1.0) / max(n.mean() - 1.0, 1e-6))
        dispersion[metric] = {"pearson_ratio": round(float(ratio), 3),
                              "rho": round(float(min(rho, 0.9)), 4),
                              "n_cycles": int(len(pair))}
    manifest["overdispersion"] = dispersion

    # Та же вариация, но с риск-коррекцией по конверту: p_i своя у каждого цикла
    # вместо общей p0. Это прямое измерение выигрыша — насколько разнородности
    # объясняется составом пациенток, а не остаётся неучтённым шумом. Пороги можно
    # опускать ровно настолько, насколько падает rho.
    from ews import envelope as envelope_mod  # noqa: PLC0415
    from ews.evaluate import ENVELOPE_STAGE  # noqa: PLC0415

    lookup = envelope_mod.as_lookup(
        envelope_mod.load_envelopes(envelope_mod.envelope_path(cfg, clinic)))
    adjusted: dict[str, dict] = {}
    if lookup and "cycle_uid" in window.columns:
        for metric, stage in ENVELOPE_STAGE.items():
            numerator_col, denominator_col = METRIC_PARTS[metric]
            rows_x, rows_n, rows_p = [], [], []
            for _, row in window.iterrows():
                entry = lookup.get(str(row.get("cycle_uid", "")))
                if not entry:
                    continue
                p50 = entry.get(f"{stage}_p50")
                x, n = row.get(numerator_col), row.get(denominator_col)
                if p50 is None or pd.isna(x) or pd.isna(n) or n <= 0 or x > n:
                    continue
                p_i = min(max(float(p50) / float(n), 0.01), 0.99)
                rows_x.append(float(x)); rows_n.append(float(n)); rows_p.append(p_i)
            if len(rows_x) < 5:
                continue
            x = np.asarray(rows_x); n = np.asarray(rows_n); p = np.asarray(rows_p)
            chi2 = (((x - n * p) ** 2) / (n * p * (1 - p))).sum()
            ratio = chi2 / max(len(x) - 1, 1)
            rho = max(0.0, (ratio - 1.0) / max(n.mean() - 1.0, 1e-6))
            adjusted[metric] = {"pearson_ratio": round(float(ratio), 3),
                                "rho": round(float(min(rho, 0.9)), 4),
                                "n_cycles": int(len(x))}
    manifest["overdispersion_risk_adjusted"] = adjusted

    # Третий контур: литературный приор механистического каскада Digital Twin.
    # Он не двигается вместе с нами — в отличие от собственной базовой линии — и
    # при этом учитывает пациентку, в отличие от когортного Венского порога.
    from ews import prior as prior_mod  # noqa: PLC0415

    gaps = prior_mod.gap_table(window, amh_constant=clinic.amh_constant)
    manifest["literature_prior"] = gaps.to_dict("records") if not gaps.empty else []
    if not gaps.empty:
        print("\nЛИТЕРАТУРНЫЙ ПРИОР (каскад L1, не зависит от наших данных):")
        show = gaps[["metric", "n_cycles", "observed", "literature", "gap_pp", "corr_rate"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        for _, row in gaps.iterrows():
            print(f"  {row['metric']:6s} {row['source']} — {row['note']}")
        usable = prior_mod.usable_for_risk_adjustment(gaps)
        fit = [m for m, ok in usable.items() if ok]
        print(f"  годится как риск-коррекция: {', '.join(fit) if fit else 'ни одна метрика'}"
              f" (нужна corr доли >= 0.3)")
    if adjusted:
        print("\nРАЗНОРОДНОСТЬ ДО И ПОСЛЕ РИСК-КОРРЕКЦИИ (X2/df, rho):")
        for metric, after in adjusted.items():
            before = dispersion.get(metric, {})
            print(f"  {metric:6s} без конверта {before.get('pearson_ratio', float('nan')):5.2f} "
                  f"/ rho {before.get('rho', float('nan')):.3f}   ->   "
                  f"с конвертом {after['pearson_ratio']:5.2f} / rho {after['rho']:.3f}"
                  f"   ({after['n_cycles']} циклов)")
    (out_dir / "incontrol_parameters.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nIN-CONTROL ПАРАМЕТРЫ ОПОРНОГО ОКНА "
          f"({start} — {window_cfg.get('end') or 'сейчас'}; "
          f"{len(window)} циклов, {int(window['two_pn'].sum())} зигот 2PN; "
          f"до окна {len(prior)} циклов):")
    table = pd.DataFrame([
        {"metric": b.metric, "p0": b.rate, "n": b.denominator,
         "ci_low": b.ci_low, "ci_high": b.ci_high, "было_до": b.prior_rate,
         "Δ_пп": b.delta_pp, "p": b.p_value, "status": b.status,
         "сдвиг_поглощён": "ДА" if b.shift_absorbed else ""}
        for b in baselines
    ])
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    notes = caveats(baselines)
    if notes:
        print("\nОГОВОРКИ БАЗОВОЙ ЛИНИИ (переносятся в манифест калибровки):")
        for note in notes:
            print(f"  · {note}")


# ---------------------------------------------------------------------------
# 2. Калибровка конверта
# ---------------------------------------------------------------------------

def collect_regression_qc(cfg, client) -> pd.DataFrame:
    rows: list[dict] = []
    for base in client.bases():
        name = base["name"]
        if not name.endswith("Digital Twin"):
            continue
        tables = {t["name"].split(" —")[0].strip(): t["id"] for t in client.tables(base["id"])}
        if "Regression QC" not in tables:
            continue
        for record in client.all_records(tables["Regression QC"]):
            fields = record["fields"]
            for stage in STAGES:
                fact = fields.get(f"{stage} факт")
                p10 = fields.get(f"{stage} P10")
                p50 = fields.get(f"{stage} P50")
                p90 = fields.get(f"{stage} P90")
                if fact is None or p10 is None or p90 is None:
                    continue
                rows.append({
                    "clinic": name.split(" —")[0].strip(), "stage": stage,
                    "fact": float(fact), "p10": float(p10),
                    "p50": float(p50) if p50 is not None else np.nan,
                    "p90": float(p90),
                })
    return pd.DataFrame(rows)


def envelope_calibration(frame: pd.DataFrame, divisor: float) -> tuple[pd.DataFrame, dict]:
    data = frame.copy()
    data["width"] = data["p90"] - data["p10"]
    data["in80"] = (data["fact"] >= data["p10"]) & (data["fact"] <= data["p90"])
    data["degenerate"] = data["width"] <= MIN_INTERVAL_WIDTH
    usable = data.loc[~data["degenerate"]].copy()
    usable["z"] = (usable["fact"] - usable["p50"]) / (usable["width"] / divisor)

    by_stage = usable.groupby("stage").agg(
        n=("in80", "size"), coverage=("in80", "mean"),
        z_mean=("z", "mean"), z_sd=("z", "std"), width_median=("width", "median"),
    )
    by_stage["mae_p50"] = usable.groupby("stage").apply(
        lambda x: (x["fact"] - x["p50"]).abs().mean(), include_groups=False)
    by_stage = by_stage.reindex([s for s in STAGES if s in by_stage.index]).reset_index()

    by_clinic = usable.groupby("clinic").agg(
        n=("in80", "size"), coverage=("in80", "mean"),
        z_mean=("z", "mean"), z_sd=("z", "std"),
    ).reset_index()

    summary = {
        "observations_total": int(len(data)),
        "degenerate_n": int(data["degenerate"].sum()),
        "degenerate_share": float(data["degenerate"].mean()),
        "degenerate_coverage": float(data.loc[data["degenerate"], "in80"].mean())
        if data["degenerate"].any() else None,
        "usable_n": int(len(usable)),
        "coverage_all_observations": float(data["in80"].mean()),
        "coverage_usable": float(usable["in80"].mean()),
        "min_interval_width": MIN_INTERVAL_WIDTH,
        "clinics": sorted(data["clinic"].unique().tolist()),
    }
    return (by_stage, by_clinic), summary


def homogeneity_note(by_clinic: pd.DataFrame) -> str:
    """Можно ли объединять клиники в одну оценку разброса остатков."""
    if len(by_clinic) < 2:
        return "одна клиника — вопрос об объединении не стоит"
    spread = by_clinic["z_sd"].max() / max(by_clinic["z_sd"].min(), 1e-9)
    if spread > 2.0:
        return (f"разброс sd(z) между клиниками {spread:.1f}x — объединять НЕЛЬЗЯ, "
                f"калибровать по своей клинике")
    return (f"разброс sd(z) между клиниками {spread:.1f}x — объединение допустимо "
            f"как оценка масштаба остатка модели")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", help="код профиля из config/clinics/")
    parser.add_argument("--out", help="куда писать; по умолчанию data/<clinic>/baseline")
    args = parser.parse_args()

    try:
        cfg = load(args.clinic)
        clinic = load_clinic(cfg, args.clinic)
        embryo_lab_data_on_path(cfg)
    except ConfigError as exc:
        print(f"[!] {exc}")
        return 2

    out_dir = cfg.path(args.out) if args.out else clinic.data_dir(cfg) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    divisor = float(cfg.settings["envelope"]["iqr_to_sigma_divisor"])

    print(f"клиника: {clinic.name} ({clinic.code})  окружение: {cfg.environment}\n")

    # 1. KPI своей лаборатории + in-control параметры опорного окна
    export_path = cfg.path(cfg.external("google_sheet")["export_path"])
    if export_path.exists():
        kpi_frame = lab_kpi_baseline(cfg, clinic, export_path)
        kpi_frame.to_csv(out_dir / "lab_kpi_baseline.csv", index=False, encoding="utf-8-sig")
        show = [c for c in ("metric", "numerator", "denominator", "rate", "ci_low",
                            "ci_high", "direction", "competency", "benchmark",
                            "status") if c in kpi_frame.columns]
        print(f"БАЗОВЫЕ KPI ЛАБОРАТОРИИ, вся история (n={kpi_frame['population_n'].iloc[0]} "
              f"циклов, исключено {kpi_frame['excluded_n'].iloc[0]}):")
        print(kpi_frame[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        write_incontrol(cfg, clinic, export_path, out_dir)
    else:
        print(f"[warn] нет экспорта листа {export_path} — базовые KPI пропущены")

    # 2. Калибровка конверта
    if not cfg.gate("teable"):
        print("\n[warn] enabled.teable=false — калибровка конверта пропущена")
        return 0
    client = teable_client.connect(cfg)
    qc = collect_regression_qc(cfg, client)
    if qc.empty:
        print("\n[warn] в локальном Teable нет строк Regression QC")
        return 0

    (by_stage, by_clinic), summary = envelope_calibration(qc, divisor)
    by_stage.to_csv(out_dir / "envelope_coverage_by_stage.csv", index=False, encoding="utf-8-sig")
    by_clinic.to_csv(out_dir / "envelope_coverage_by_clinic.csv", index=False, encoding="utf-8-sig")
    summary["homogeneity"] = homogeneity_note(by_clinic)
    (out_dir / "baseline_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nКАЛИБРОВКА КОНВЕРТА ({summary['observations_total']} наблюдений, "
          f"клиники: {', '.join(summary['clinics'])})")
    print(f"  вырожденных (ширина <= {MIN_INTERVAL_WIDTH}): {summary['degenerate_n']} "
          f"({summary['degenerate_share']*100:.1f}%), покрытие в них "
          f"{summary['degenerate_coverage']:.3f} — исключены из слоя остатков")
    print(f"  покрытие по всем наблюдениям: {summary['coverage_all_observations']:.3f}")
    print(f"  покрытие по пригодным:        {summary['coverage_usable']:.3f}")
    print("\n  по стадиям:")
    print(by_stage.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n  по клиникам:")
    print(by_clinic.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  однородность: {summary['homogeneity']}")
    print(f"\nартефакты: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
