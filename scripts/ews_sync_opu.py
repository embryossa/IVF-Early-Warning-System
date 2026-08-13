"""Синхронизировать мастер-таблицу OPU в локальный Teable.

    python scripts/ews_sync_opu.py [--source <csv>] [--dry-run]

Источник по умолчанию — CSV-экспорт листа из machine.local.yaml
(`google_sheet.export_path`). На черновом стенде экспорт кладётся туда
интерактивно через Drive MCP; в лаборатории источником становится сам Teable,
куда синхронизирует рабочий ПК.

Направление одностороннее: лист -> Teable. Строки, исчезнувшие из листа, в Teable
не удаляются — удаление данных пациентки по расхождению выгрузок не то решение,
которое принимает скрипт.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import sync_opu, teable_client  # noqa: E402
from ews.config import ConfigError, embryo_lab_data_on_path, load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="CSV-экспорт листа; по умолчанию из machine.local.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="посчитать и показать, ничего не записывая")
    args = parser.parse_args()

    try:
        cfg = load()
        embryo_lab_data_on_path(cfg)
    except ConfigError as exc:
        print(f"[!] {exc}")
        return 2

    from ews.vendor.privacy import load_or_create_secret  # noqa: PLC0415

    source_path = Path(args.source) if args.source else cfg.path(
        cfg.external("google_sheet")["export_path"]
    )
    if not source_path.exists():
        print(f"[!] нет источника: {source_path}\n"
              f"    Положите свежий CSV-экспорт листа OPU по этому пути.")
        return 2

    frame = sync_opu.load_sheet_export(source_path)
    print(f"окружение: {cfg.environment}")
    print(f"источник:  {source_path.name} — {len(frame)} строк с пациенткой")

    client = teable_client.connect(cfg, dry_run=args.dry_run)
    teable_cfg = cfg.external("teable")
    base_id = str(teable_cfg.get("base_id") or "").strip()
    if not base_id:
        print("[!] teable.base_id пуст — сначала выполните scripts/ews_bootstrap_opu.py")
        return 2

    table_ids = teable_client.resolve_tables(cfg, client, base_id)
    if "OPU" not in table_ids:
        print("[!] в базе нет таблицы OPU — выполните scripts/ews_bootstrap_opu.py")
        return 2

    secret = load_or_create_secret(cfg.path(cfg.settings["privacy"]["secret_file"]))
    report = sync_opu.sync(cfg, client, table_ids, frame, secret,
                           source=source_path.name, dry_run=args.dry_run)

    prefix = "[dry] " if args.dry_run else ""
    print(f"\n{prefix}прочитано {report.read}: "
          f"создано {report.created}, обновлено {report.updated}, "
          f"без изменений {report.unchanged}, пропущено {report.skipped}")
    for note in report.notes:
        print(f"  · {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
