#!/usr/bin/env python3
"""Демо-прогон EWS на выгрузке другой лаборатории (Excel/CSV) -> интерактивная панель KPI.

    python scripts/ews_demo.py --input <файл.xlsx|csv> [--check-columns]
    python scripts/ews_demo.py --input <файл> --columns override.yaml --ref-start 2026-01-01
    python scripts/ews_demo.py --input <файл> --commentary data/agent_packets/demo_<slug>_commentary.md

Что делает: сопоставляет колонки по словарю синонимов (без угадывания), псевдонимизирует
на входе, считает KPI Венского консенсуса по правилам проекта, строит контрольные карты
(p-карта Laney, EWMA, CUSUM) и сравнения, пишет автономный HTML.

Выход (всё в data/, в git не попадает):
  data/demo/<slug>/EWS_Demo_<slug>.html     — панель
  data/demo/<slug>/demo_kpi.csv             — сводка KPI со штампом environment
  data/demo/<slug>/demo_manifest.json       — манифест прогона
  data/agent_packets/demo_<slug>.json       — пакет агрегатов для комментария ИИ (без PHI)
  data/local_private/demo_operators_<slug>.csv — псевдоним -> имя специалиста (только локально)

Демо всегда environment=draft, mode=demo: уровень alert не поднимается.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews.demo import dashboard, ingest, report  # noqa: E402
from ews.vendor import _io, privacy  # noqa: E402

CONTRACT_VERSION = "0.1"


def slugify(name: str) -> str:
    stem = re.sub(r"[^\w\-]+", "_", Path(name).stem, flags=re.UNICODE).strip("_")
    return (stem or "export")[:48]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return [_jsonable(r) for r in value.to_dict(orient="records")]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()[:10]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 5)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def agent_packet(res: dict) -> dict:
    """Только агрегаты: ни одной строки цикла, ни cycle_uid, ни имён специалистов."""
    emb = {}
    for metric, e in res["embryologists"].items():
        if metric.startswith("_"):
            emb[metric] = e
            continue
        t = e["test"]
        emb[metric] = {k: t.get(k) for k in ("test", "why", "p", "phi", "robust", "per_group",
                                             "pairwise", "excluded_small")}
    series = {m: {"p0": e["p0"], "phi": e["phi"], "rho": e["rho"], "pchart": e["pchart_info"],
                  "month_test": {k: e["month_test"].get(k) for k in ("test", "p", "phi", "why")},
                  "monthly": e["monthly"]}
              for m, e in res["series"].items()}
    cusum = {m: ({"h_watch": c["h_watch"], "h_alert": c["h_alert"], "per_year": c["per_year"],
                  "sim_rates": c["sim_rates"], "watch_events": len(c["events"]),
                  "alert_threshold_crossed": int(c["events"]["alert_threshold_crossed"].sum())
                  if not c["events"].empty else 0}
                 if c.get("available") else {"available": False, "reason": c.get("reason")})
             for m, c in res["cusum"].items()}
    packet = {
        "schema_version": CONTRACT_VERSION, "environment": "draft", "mode": "demo",
        "generated_at": res["generated_at"], "source_name": res["source_name"],
        "period": res.get("period"), "n_cycles": res["n_cycles"], "n_2pn": res["n_2pn"],
        "exclusions": res["exclusions"], "two_pn_zero_cycles": res.get("two_pn_zero_cycles"),
        "fr_with_zero_2pn": res.get("fr_with_zero_2pn"), "reference": res["reference"],
        "kpi": res["kpi"], "fr_by_method": res.get("fr_method"), "fr_method_test": res.get("fr_method_test"),
        "fr_ks": res.get("fr_ks"), "series": series, "cusum": cusum, "episodes": res["episodes"],
        "month_vs_rest": res["heat"], "embryologists": emb, "correlations": res["correlations"],
        "tgbdr_by_age": res["tgbdr_age"], "facts": res["facts"], "checks": res["checks"],
        "notes": res["notes"],
        "rules": [
            "Числа — только из этого пакета; ничего не пересчитывать и не округлять иначе.",
            "Факт и гипотеза разделяются явно.",
            "Демо: alert невозможен; watch не называть тревогой боевого уровня.",
            "CUSUM и EWMA одной метрики — одно измерение, не два подтверждения.",
            "Нет логов инкубатора и лотов: независимого подтверждения нет и быть не может.",
            "Не оценка специалиста; различия эмбриологов — с оговоркой о составе пациенток.",
            "ClR>100% и MII>OCC — наблюдения, не ошибки данных.",
        ],
    }
    return _jsonable(packet)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="выгрузка .xlsx/.xls/.csv")
    ap.add_argument("--sheet", default=None, help="лист Excel (имя или номер)")
    ap.add_argument("--columns", default=None, help="YAML с явным соответствием canonical: {имя: 'Колонка'}")
    ap.add_argument("--check-columns", action="store_true", help="только показать сопоставление колонок")
    ap.add_argument("--out", default=None, help="каталог вывода (по умолчанию data/demo/<slug>)")
    ap.add_argument("--window", type=int, default=20, help="окно скользящей доли, циклов")
    ap.add_argument("--ref-start", default=None, help="начало опорного окна карт YYYY-MM-DD")
    ap.add_argument("--ref-end", default=None, help="конец опорного окна карт YYYY-MM-DD")
    ap.add_argument("--replicates", type=int, default=300, help="повторов бутстрепа для порогов CUSUM")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--show-operator-names", action="store_true",
                    help="показать имена эмбриологов как в выгрузке (по умолчанию — псевдонимы)")
    ap.add_argument("--plotly", choices=["inline", "cdn"], default="inline",
                    help="inline — панель работает без интернета (~5 МБ); cdn — лёгкий файл")
    ap.add_argument("--commentary", default=None, help="Markdown-комментарий ИИ для вставки в панель")
    args = ap.parse_args()

    src = Path(args.input).expanduser()
    if not src.exists():
        print(f"[ОШИБКА] нет файла: {src}")
        return 2
    sheet = int(args.sheet) if args.sheet and str(args.sheet).isdigit() else args.sheet
    override = {}
    if args.columns:
        with Path(args.columns).open(encoding="utf-8") as fh:
            override = (yaml.safe_load(fh) or {}).get("canonical", {})

    if args.check_columns:
        raw = ingest.read_table(src, sheet)
        rep = ingest.resolve_columns([str(c) for c in raw.columns], ingest.load_aliases(), override)
        print(f"Файл: {src.name} — {len(raw)} строк, {len(raw.columns)} колонок\n")
        print("Сопоставлено:")
        for k, v in rep.mapped.items():
            print(f"  {k:<22} <- {v}{'  (override)' if k in rep.overridden else ''}")
        if rep.ambiguous:
            print("\nНЕОДНОЗНАЧНО (не сопоставлено, нужен override):")
            for k, v in rep.ambiguous.items():
                print(f"  {k:<22} ?  {v}")
        print(f"\nНе найдено: {', '.join(rep.missing) or '—'}")
        print(f"\nКолонки файла без сопоставления ({len(rep.unmapped_source)}): "
              f"{', '.join(rep.unmapped_source[:60])}")
        core = ["two_pn", "bl_total", "bl_good", "occ", "mii", "date_opu"]
        lacking = [c for c in core if c not in rep.mapped]
        print("\nКлючевые для панели: " + ("все на месте" if not lacking else "НЕТ " + ", ".join(lacking)))
        return 0

    started = time.time()
    slug = slugify(src.name)
    out_dir = Path(args.out) if args.out else ROOT / "data" / "demo" / slug
    secret = privacy.load_or_create_secret(ROOT / "data" / "local_private" / "cycle_uid.key")

    demo = ingest.load_demo(src, sheet=sheet, override=override, secret=secret,
                            show_operator_names=args.show_operator_names)
    print(f"[ingest] {src.name}: {demo.exclusions['rows_total']} строк -> "
          f"{demo.exclusions['lab_population']} циклов в популяции; "
          f"колонок сопоставлено {len(demo.columns.mapped)}")
    if demo.columns.ambiguous:
        print(f"[ingest] неоднозначные колонки (не использованы): {demo.columns.ambiguous}")
    if demo.frame.empty:
        print("[ОШИБКА] после исключений не осталось циклов — проверьте сопоставление (--check-columns)")
        return 3

    res = report.build(demo, rolling_window=args.window, ref_start=args.ref_start,
                       ref_end=args.ref_end, replicates=args.replicates, seed=args.seed)
    commentary = Path(args.commentary).read_text(encoding="utf-8") if args.commentary else None
    html = dashboard.render(res, commentary=commentary, plotly_js=args.plotly,
                            show_operator_names=args.show_operator_names)

    html_path = _io.atomic_write_text(html, out_dir / f"EWS_Demo_{slug}.html")
    kpi = res["kpi"].assign(environment="draft", mode="demo", generated_at=res["generated_at"],
                            source=res["source_name"])
    _io.atomic_write_csv(kpi, out_dir / "demo_kpi.csv")
    packet_path = ROOT / "data" / "agent_packets" / f"demo_{slug}.json"
    _io.atomic_write_text(json.dumps(agent_packet(res), ensure_ascii=False, indent=1), packet_path)
    if not demo.operator_crosswalk.empty:
        _io.atomic_write_csv(demo.operator_crosswalk,
                             ROOT / "data" / "local_private" / f"demo_operators_{slug}.csv")
    manifest = {"run_id": f"demo-{slug}-{datetime.now():%Y%m%dT%H%M%S}", "started_at": datetime.now().isoformat(),
                "environment": "draft", "mode": "demo", "contract_version": CONTRACT_VERSION,
                "source_name": src.name, "rows": demo.exclusions["rows_total"], "cycles": res["n_cycles"],
                "alert_level_available": False,
                "alert_block_reason": "mode=demo: пороги откалиброваны на оцениваемой выгрузке",
                "soft_step_status": {"miri": "unavailable", "lots": "unavailable", "envelope": "unavailable"},
                "settings_used": res["settings_used"], "commentary": bool(commentary),
                "outputs": {"html": html_path.name, "kpi": "demo_kpi.csv",
                            "agent_packet": packet_path.relative_to(ROOT).as_posix()},
                "elapsed_s": round(time.time() - started, 1)}
    _io.atomic_write_text(json.dumps(manifest, ensure_ascii=False, indent=1), out_dir / "demo_manifest.json")

    print(f"[ok] панель: {html_path.relative_to(ROOT).as_posix() if html_path.is_relative_to(ROOT) else html_path}")
    print(f"[ok] пакет для комментария ИИ: {packet_path.relative_to(ROOT).as_posix()}")
    for fact in res["facts"]:
        print("  •", fact)
    print(f"[ok] {manifest['elapsed_s']} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
