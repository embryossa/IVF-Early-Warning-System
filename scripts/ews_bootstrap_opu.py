"""Создать/обновить базу «EWS — OPU master» в локальном Teable.

Идемпотентно: недостающие таблицы создаются, недостающие поля добавляются, ничего
не переименовывается и не удаляется. Повторный запуск на готовой базе — штатный
способ применить расширение схемы.

    python scripts/ews_bootstrap_opu.py [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import opu_schema, teable_client  # noqa: E402
from ews.config import ConfigError, load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="показать, что было бы создано, ничего не меняя")
    args = parser.parse_args()

    try:
        cfg = load()
    except ConfigError as exc:
        print(f"[!] {exc}")
        return 2

    print(f"окружение: {cfg.environment}")
    client = teable_client.connect(cfg, dry_run=args.dry_run)
    teable_cfg = cfg.external("teable")
    base_id = str(teable_cfg.get("base_id") or "").strip()
    base_name = teable_cfg.get("base_name") or "EWS — OPU master"

    if not base_id:
        space_id = str(teable_cfg.get("space_id") or "").strip()
        if not space_id:
            print("[!] machine.local.yaml: teable.space_id пуст — неизвестно, где создавать базу")
            return 2
        if args.dry_run:
            print(f"  [dry] + база {base_name} в пространстве {space_id}")
            return 0
        base_id = client.create_base(space_id, base_name)
        print(f"  + база {base_name} ({base_id})")
    else:
        print(f"база уже задана: {base_id}")

    table_ids = teable_client.resolve_tables(cfg, client, base_id) if not args.dry_run else {}
    made_tables = made_fields = 0

    for table in opu_schema.schema():
        key, name, fields = table["key"], table["name"], table["fields"]
        if key not in table_ids:
            if args.dry_run:
                print(f"  [dry] + таблица {name} ({len(fields)} полей)")
                made_tables += 1
                continue
            # records: [] — иначе Teable засеивает новую таблицу тремя пустыми
            # строками, и они выглядят как фантомные циклы.
            created = client.create_table(base_id, {"name": name, "records": [],
                                                    "fields": fields})
            table_ids[key] = created["id"]
            made_tables += 1
            print(f"  + таблица {name} ({created['id']}, {len(created['fields'])} полей)")
            continue

        have = {fl["name"] for fl in client.fields(table_ids[key])}
        for spec in [fl for fl in fields if fl["name"] not in have]:
            if args.dry_run:
                print(f"  [dry] + поле {name} :: {spec['name']}")
                made_fields += 1
                continue
            try:
                client.create_field(table_ids[key], dict(spec))
            except Exception as exc:  # noqa: BLE001 — сообщаем и продолжаем
                print(f"  [!] поле {name} :: {spec['name']} не создано: {exc}")
                continue
            made_fields += 1
            print(f"  + поле {name} :: {spec['name']}")

    if not args.dry_run:
        teable_client.save_ids(cfg, base_id, table_ids.get("OPU", ""),
                               table_ids.get("Sync", ""))
        print(f"id записаны в config/machine.local.yaml")

    print(f"\nитог: таблиц создано {made_tables}, полей добавлено {made_fields}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
