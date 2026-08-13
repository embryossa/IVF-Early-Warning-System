"""Фаза 1: подбор порогов карт под целевую частоту ложных тревог.

    python scripts/ews_backfit.py [--replicates 4000] [--write]

Обратная задача: цели заданы сверху (docs/METHODOLOGY.md §5), `h` вычисляется под них
симуляцией in-control процесса, а не берётся из литературы.

Что учитывается и почему:

* **Неопределённость базовой доли.** Опорное окно даёт 312 зигот; центр BlR известен с
  точностью около ±5 пп. В каждом прогоне симуляции истинная доля разыгрывается из
  апостериорного Beta, а карта продолжает считать базовой точечную оценку. Без этого
  порог оказался бы увереннее данных под ним.
* **Совместность целей.** «2 ложных alert в год» — на систему целиком, поэтому на карту
  приходится 2/m. Добавление карты ужесточает пороги всем.
* **Цена порога.** Для каждого найденного `h` считается ARL1 — через сколько эмбрионов
  карта поймает настоящий сдвиг. Порог без этой цифры бессмысленен.

Упрощение, которое надо знать: симуляция ведётся при однородной базовой доле, тогда как
боевая карта риск-скорректирована (p_i своя у каждого эмбриона). Риск-коррекция снижает
дисперсию статистики, поэтому найденный здесь `h` слегка консервативен — ложных тревог
будет не больше целевого, а чуть меньше.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import charts  # noqa: E402
from ews.clinic import Clinic, load_clinic  # noqa: E402
from ews.config import ConfigError, load  # noqa: E402

# Горизонт симуляции как множитель целевого ARL: слишком короткий горизонт цензурирует
# длинные серии и занижает ARL, из-за чего h выходит слишком низким.
HORIZON_FACTOR = 6


def _metric_baseline(manifest: dict, metric: str) -> dict | None:
    for entry in manifest.get("metrics", []):
        if entry["metric"] == metric:
            return entry
    return None


def _volume_per_year(manifest: dict, metric_baseline: dict) -> float:
    """Наблюдений в год по этой метрике, исходя из объёма опорного окна."""
    window = manifest["reference_window"]
    from datetime import date

    start = date.fromisoformat(window["start"])
    end = date.fromisoformat(window["end"])
    days = max((end - start).days, 1)
    return float(metric_baseline["denominator"]) * 365.0 / days


def calibrate(cfg, clinic: Clinic, replicates: int, seed: int = 20260813) -> dict:
    baseline_path = clinic.data_dir(cfg) / "baseline" / "incontrol_parameters.json"
    if not baseline_path.exists():
        raise ConfigError(
            f"нет {baseline_path}. Сначала выполните "
            f"scripts/ews_baseline.py --clinic {clinic.code}."
        )
    manifest = json.loads(baseline_path.read_text(encoding="utf-8"))

    # Набор карт — свойство ЛАБОРАТОРИИ (её объёма), не метода: профиль клиники.
    chart_metrics = clinic.cusum_charts
    m = len(chart_metrics)

    targets = cfg.settings["thresholds"]["target_false_alarms_per_year"]
    cusum_cfg = cfg.settings["charts"]["cusum"]
    rng = np.random.default_rng(seed)

    results: list[dict] = []
    for metric in chart_metrics:
        base = _metric_baseline(manifest, metric)
        if base is None:
            results.append({"metric": metric, "status": "нет в опорном окне"})
            continue

        successes, total = base["numerator"], base["denominator"]
        p0 = base["rate"]
        if not np.isfinite(p0) or total <= 0 or successes > total:
            # numerator > denominator — штатная ситуация (ClR, MIIR), но карта на
            # такой метрике не строится: базовая доля не определена.
            results.append({"metric": metric, "status": "базовая доля не определена"})
            continue

        odds_ratio = (cusum_cfg["odds_ratio_deterioration"]
                      if base["direction"] == "lower"
                      else cusum_cfg["odds_ratio_upper_control"])
        # У верхнего контроля «успех» — нежелательное событие.
        p_chart = p0 if base["direction"] == "lower" else p0
        per_year = _volume_per_year(manifest, base)

        entry: dict = {
            "metric": metric, "status": "ok", "p0": p0,
            "numerator": successes, "denominator": total,
            "direction": base["direction"], "odds_ratio": odds_ratio,
            "observations_per_year": round(per_year, 1),
            "charts_active": m,
        }

        sampler = charts.jeffreys_sampler(successes, total)
        # Горизонт симуляции = наблюдений за год: цель задана в тревогах на год.
        horizon = max(int(round(per_year)), 50)
        # Настоящий сдвиг, ровно тот, на который карта настроена (шансы × odds_ratio).
        shifted = p0 * odds_ratio / (1 - p0 + odds_ratio * p0)

        # Пара порогов подбирается СОВМЕСТНО и тем же счётчиком событий, который
        # работает в бою (charts.CusumRunner). Раздельный подбор моделировал бы
        # процесс, которого не существует: watch статистику не сбрасывает, а alert
        # сбрасывает, и это влияет на то, сколько эпизодов watch успеет случиться.
        sizes = manifest.get("block_sizes", {}).get(metric)
        if not sizes:
            entry["status"] = "нет распределения размеров цикла"
            results.append(entry)
            continue
        solved = charts.solve_thresholds(
            sampler, p_chart, odds_ratio,
            target_watch=float(targets["watch"]) / m,
            target_alert=float(targets["alert"]) / m,
            horizon=horizon, replicates=replicates, rng=rng,
            block_sizes=np.asarray(sizes),
            dispersion=float(manifest.get("overdispersion", {})
                             .get(metric, {}).get("rho", 0.0)),
        )
        overdispersion = manifest.get("overdispersion", {}).get(metric, {})
        entry["overdispersion_rho"] = overdispersion.get("rho", 0.0)
        entry["overdispersion_pearson_ratio"] = overdispersion.get("pearson_ratio")
        entry["median_cycle_size"] = int(np.median(sizes))
        entry["cycles_in_window"] = len(sizes)
        for level in ("watch", "alert"):
            h = solved[f"h_{level}"]
            delay_mean, delay_median = charts.simulate_detection_delay(
                shifted, p_chart, odds_ratio, h,
                horizon=horizon, replicates=replicates, rng=rng,
            )
            entry[level] = {
                "h": round(h, 3),
                "target_false_alarms_per_year": round(float(targets[level]) / m, 3),
                "achieved_false_alarms_per_year": round(solved["rates"].get(level, 0.0), 3),
                "detection_delay_observations_mean": round(delay_mean, 1),
                "detection_delay_observations_median": round(delay_median, 1),
                "detection_delay_months_median": round(
                    delay_median / (per_year / 12.0), 1),
                "p_shifted": round(shifted, 3),
            }
        entry["quiet_years_share"] = round(solved["rates"].get("quiet_years", 0.0), 3)
        results.append(entry)

    ok = [r for r in results if r.get("status") == "ok"]
    return {
        "calibrated_at": None,           # проставляется при записи
        "calibrated_environment": cfg.environment,
        "clinic": {"code": clinic.code, "name": clinic.name},
        "reference_window": manifest["reference_window"],
        "charts_active": m,
        "targets_per_year": dict(targets),
        "replicates": replicates,
        "seed": seed,
        "simplification": "однородная базовая доля; риск-коррекция снизит дисперсию, "
                          "поэтому найденный h слегка консервативен",
        "charts": results,
        "baseline_caveats": manifest.get("baseline_caveats", []),
        "declared_shifts": manifest.get("declared_shifts", {}),
        "excluded_from_charts": clinic.subjective_denominator,
        "system_false_alarms_per_year": {
            level: round(sum(r[level]["achieved_false_alarms_per_year"] for r in ok), 2)
            for level in ("watch", "alert")
        } if ok else {},
    }


def write_back(cfg, clinic: Clinic, report: dict) -> None:
    """Записать пороги в config/thresholds.local.yaml и манифест калибровки.

    Пороги НЕ пишутся в settings.yaml по двум причинам, обеим существенным:
    машинная перезапись стёрла бы комментарии, в которых живёт обоснование
    параметров; и файл в .gitignore физически не может уехать в лабораторию вместе
    с репозиторием — а пороги чернового стенда там непригодны (docs/PORTING.md §4).

    `calibrated_environment` пишется рядом с флагом намеренно: гейт alert сверяет
    его с текущим окружением сам, вместо того чтобы полагаться на память
    переносящего.
    """
    from datetime import date

    import yaml

    report["calibrated_at"] = date.today().isoformat()
    out_dir = clinic.data_dir(cfg) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [c for c in report["charts"] if c.get("status") == "ok"]
    payload = {
        "calibrated": bool(ok),
        "calibrated_at": report["calibrated_at"],
        "calibrated_environment": report["calibrated_environment"],
        "calibration_window": (f"{report['reference_window']['start']}.."
                               f"{report['reference_window']['end']}"),
        "clinic": clinic.code,
        "charts_active": report["charts_active"],
        "h_watch": {c["metric"]: c["watch"]["h"] for c in ok},
        "h_alert": {c["metric"]: c["alert"]["h"] for c in ok},
        "measured_false_alarm_rate": report["system_false_alarms_per_year"],
    }
    header = (
        "# config/thresholds.local.yaml — СГЕНЕРИРОВАНО scripts/ews_backfit.py.\n"
        "# Руками не править: любое изменение будет затёрто следующим бэкфитом.\n"
        "# В .gitignore намеренно — пороги подогнаны под объём и разброс ЭТОЙ\n"
        "# лаборатории и на другой машине непригодны (docs/PORTING.md §4).\n"
        "# Полный отчёт калибровки: data/baseline/calibration_manifest.json\n\n"
    )
    path = clinic.thresholds_path()
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(header + yaml.safe_dump(payload, allow_unicode=True,
                                           sort_keys=False, width=100),
                   encoding="utf-8", newline="\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", help="код профиля из config/clinics/")
    parser.add_argument("--replicates", type=int, default=4000)
    parser.add_argument("--write", action="store_true",
                        help="записать найденные h в settings.yaml и манифест")
    args = parser.parse_args()

    try:
        cfg = load(args.clinic)
        clinic = load_clinic(cfg, args.clinic)
        report = calibrate(cfg, clinic, args.replicates)
    except ConfigError as exc:
        print(f"[!] {exc}")
        return 2

    window = report["reference_window"]
    print(f"клиника: {clinic.name} ({clinic.code})  "
          f"окружение: {report['calibrated_environment']}")
    print(f"опорное окно: {window['start']}..{window['end']} — "
          f"{window['n_cycles']} циклов, {window['n_embryos_2pn']:.0f} зигот")
    print(f"активных карт: {report['charts_active']}  "
          f"цели/год: watch <= {report['targets_per_year']['watch']}, "
          f"alert <= {report['targets_per_year']['alert']}\n")

    header = (f"{'метрика':7s} {'p0':>6s} {'n окна':>7s} {'набл/год':>9s} | "
              f"{'h_watch':>8s} {'ложн/год':>9s} {'задержка':>9s} | "
              f"{'h_alert':>8s} {'ложн/год':>9s} {'задержка':>9s}")
    print(header)
    print("-" * len(header))
    for chart in report["charts"]:
        if chart.get("status") != "ok":
            print(f"{chart['metric']:7s} — {chart.get('status')}")
            continue
        w, a = chart["watch"], chart["alert"]
        print(f"{chart['metric']:7s} {chart['p0']:6.3f} {chart['denominator']:7.0f} "
              f"{chart['observations_per_year']:9.0f} | "
              f"{w['h']:8.2f} {w['achieved_false_alarms_per_year']:9.2f} "
              f"{w['detection_delay_months_median']:6.1f} мес | "
              f"{a['h']:8.2f} {a['achieved_false_alarms_per_year']:9.2f} "
              f"{a['detection_delay_months_median']:6.1f} мес")
    print("\nзадержка = медиана до сигнала при настоящем сдвиге "
          f"(шансы × {report['charts'][0].get('odds_ratio', 0.5)}), в месяцах работы")

    system = report["system_false_alarms_per_year"]
    print(f"\nпо системе целиком: ложных watch/год {system.get('watch')}, "
          f"alert/год {system.get('alert')}")
    print(f"упрощение: {report['simplification']}")

    if report["baseline_caveats"]:
        print("\nоговорки базовой линии (перенесены в манифест):")
        for note in report["baseline_caveats"]:
            print(f"  · {note}")

    if args.write:
        write_back(cfg, clinic, report)
        print(f"\nзаписано: {clinic.thresholds_path().name}, "
              f"{(clinic.data_dir(cfg) / 'baseline' / 'calibration_manifest.json').name}")
    else:
        print("\n(ничего не записано — добавьте --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
