"""Недельный проход: обогащение средой и пересчёт границ «необычного».

    python scripts/ews_enrich_miri.py --clinic <code> [--dry-run]

Запускается раз в неделю, после появления свежей выгрузки логов инкубатора. Делает
две вещи, обе — по всей когорте, а не по сработавшим циклам:

1. **Разбирает окна культивирования** и записывает эпизоды среды и события лотов.
2. **Пересчитывает границы «необычного»** (`state/confounder_baseline.json`) —
   верхний дециль нагрузки по когорте. Суточный прогон берёт их готовыми и потому
   собирает confounders только для сработавших циклов, за секунды вместо минут.

Границы обязаны быть относительными. Измерено на 49 циклах: критический эпизод есть
у 100% из них — за шесть суток на шести камерах открытие крышки это рутинная работа
эмбриолога, а экскурсия O2 после него — физика. Абсолютный порог «есть критический
эпизод» подтверждал бы каждую тревогу, то есть не подтверждал бы ничего. Различает
нагрузку только её объём: критические минуты, медиана 103 против верхнего дециля 1916.

**Чего проход не делает.** Он не создаёт тревог и не снимает их. Он может только
дописать обстановку к уже существующей тревоге и — при появлении сильного
подтверждения — повысить `watch` до `alert`. Дата возникновения тревоги при этом
сохраняется: система обязана помнить, когда она узнала, а не когда ей объяснили.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import confounders, cycles, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import ConfigError, load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", help="код профиля из config/clinics/")
    parser.add_argument("--dry-run", action="store_true",
                        help="посчитать и показать, ничего не записывая")
    args = parser.parse_args()

    try:
        cfg = load(args.clinic)
        clinic = load_clinic(cfg, args.clinic)
    except ConfigError as exc:
        print(f"[!] {exc}")
        return 2

    print(f"клиника: {clinic.name} ({clinic.code})  окружение: {cfg.environment}")
    if not clinic.source_enabled("miri") and not clinic.source_enabled("lots"):
        print("оба источника выключены в профиле клиники — делать нечего")
        return 0

    client = teable_client.connect(cfg)
    table_id = str(cfg.external("teable").get("opu_table_id") or "").strip()
    if not table_id:
        print("[!] teable.opu_table_id пуст — выполните scripts/ews_bootstrap_opu.py")
        return 2

    frame = cycles.load_from_teable(cfg, clinic, client, table_id)
    closed = frame.data.loc[frame.data["checkpoint"] == cycles.CHECKPOINT_CLOSURE]
    print(f"закрытых циклов: {len(closed)}")
    if closed.empty:
        return 0

    contexts = confounders.collect(cfg, clinic, closed)

    statuses: dict[str, int] = {}
    for context in contexts.values():
        statuses[context.miri_status] = statuses.get(context.miri_status, 0) + 1
    print(f"статус среды: {statuses}")

    enriched = [c for c in contexts.values() if c.miri_events]
    if enriched:
        minutes = pd.Series([c.critical_minutes for c in enriched])
        print(f"нагрузка средой, критических минут: медиана {minutes.median():.0f}, "
              f"верхний дециль {minutes.quantile(0.9):.0f}, максимум {minutes.max():.0f}")

    thresholds = confounders.cohort_thresholds(contexts)
    if not thresholds:
        print("[warn] данных мало для границ (нужно >=10 циклов с разобранной "
              "обстановкой) — подтверждение по среде останется недоступным")
    else:
        unusual = sum(1 for c in contexts.values() if c.is_unusual(thresholds))
        print(f"границы «необычного»: "
              f"{ {k: round(v, 1) for k, v in thresholds.items()} }")
        print(f"циклов с необычной обстановкой: {unusual} из {len(contexts)} "
              f"({unusual / len(contexts):.0%})")

    if args.dry_run:
        print("\n(dry-run: ничего не записано)")
        return 0

    if thresholds:
        confounders.save_baseline(cfg, clinic, thresholds, len(contexts))
        print(f"границы записаны: "
              f"{confounders.baseline_path(cfg, clinic).relative_to(cfg.data_dir.parent)}")

    # Обстановка по циклам — отдельным артефактом, чтобы её можно было читать
    # независимо от тревог и сверять с журналом.
    rows = []
    for uid, context in contexts.items():
        rows.append({
            "cycle_uid": uid,
            "miri_status": context.miri_status,
            "episodes": len(context.miri_events),
            "critical_minutes": round(context.critical_minutes, 1),
            "lot_changes": context.lot_changes,
            "unusual": bool(thresholds and context.is_unusual(thresholds)),
            "notes": " | ".join(context.notes),
            "environment": cfg.environment,
        })
    out = clinic.data_dir(cfg) / "ews_window_context.csv"
    tmp = out.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(out)
    print(f"обстановка по циклам: {out.name}")

    manifest = clinic.data_dir(cfg) / "state" / "miri_enrichment.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "clinic": clinic.code, "environment": cfg.environment,
        "cycles": len(contexts), "statuses": statuses,
        "thresholds": thresholds,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
