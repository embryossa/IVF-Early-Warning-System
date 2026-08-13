"""Перенос вендоренных модулей из embryo-lab-data и проверка их актуальности.

    python scripts/ews_sync_vendor.py --check
    python scripts/ews_sync_vendor.py --sync --from <путь к embryo-lab-data>

`--check` сравнивает контрольные суммы вендоренных копий с манифестом и, если
источник доступен, — с оригиналами. Работает и без источника: на лабораторной
машине `embryo-lab-data` нет, и там проверка сводится к целостности копий.

`--sync` переносит файлы заново. Выполняется на машине, где источник есть,
ПЕРЕД переносом проекта.

Почему это отдельный шаг, а не ручное копирование: вендоренные файлы — копии, а не
форк. Правило «править только в источнике» держится не на дисциплине, а на том, что
расхождение видно этой командой. Молча разошедшийся `kpi.py` означал бы, что одна и
та же клиника считает свои KPI по-разному в двух проектах.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews.config import ROOT  # noqa: E402

VENDOR_DIR = ROOT / "src" / "ews" / "vendor"
MANIFEST = VENDOR_DIR / "vendor_manifest.json"

# файл в вендоре -> путь внутри embryo-lab-data
FILES = {
    "kpi.py": "src/embryo/kpi.py",
    "privacy.py": "src/embryo/privacy.py",
    "incubator.py": "src/embryo/incubator.py",
    "lots.py": "src/embryo/lots.py",
    "_io.py": "src/embryo/_io.py",
    "teable_client_impl.py": "teable/teable_common.py",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def check(source: Path | None) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    recorded = manifest.get("files", {})
    problems = 0

    print(f"вендор: {VENDOR_DIR.relative_to(ROOT)}")
    if manifest:
        print(f"перенесено: {manifest.get('synced_at')} из {manifest.get('source')}\n")

    for name, relative in FILES.items():
        local = VENDOR_DIR / name
        if not local.exists():
            print(f"  [НЕТ ] {name} — отсутствует")
            problems += 1
            continue
        local_hash = digest(local)
        expected = recorded.get(name)
        state = "ok" if expected is None or expected == local_hash else "ИЗМЕНЁН"
        if state == "ИЗМЕНЁН":
            problems += 1
            print(f"  [{state}] {name} — копия правлена на месте; правки делаются "
                  f"в embryo-lab-data и переносятся --sync")
            continue

        if source is not None:
            origin = source / relative
            if not origin.exists():
                print(f"  [warn] {name} — в источнике нет {relative}")
                continue
            origin_hash = digest(origin)
            if origin_hash != local_hash:
                problems += 1
                print(f"  [УСТАРЕЛ] {name} — источник изменился, нужен --sync")
                continue
            print(f"  [ok  ] {name} — совпадает с источником")
        else:
            print(f"  [ok  ] {name} — целостность копии подтверждена")

    if source is None:
        print("\nисточник не указан: сверка с оригиналами пропущена (штатно для "
              "лабораторной машины, где embryo-lab-data нет)")
    print("\nготово" if not problems else f"\nпроблем: {problems}")
    return 1 if problems else 0


def sync(source: Path) -> int:
    if not source.exists():
        print(f"[!] нет источника: {source}")
        return 2
    from datetime import date

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, relative in FILES.items():
        origin = source / relative
        if not origin.exists():
            print(f"  [!] нет {relative} — пропущено")
            continue
        shutil.copy2(origin, VENDOR_DIR / name)
        files[name] = digest(VENDOR_DIR / name)
        print(f"  + {name}")

    MANIFEST.write_text(json.dumps({
        "source": str(source),
        "synced_at": date.today().isoformat(),
        "note": "копии, не форк: править только в источнике",
        "files": files,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nперенесено файлов: {len(files)}; манифест обновлён")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--from", dest="source", help="путь к embryo-lab-data")
    args = parser.parse_args()

    source = Path(args.source) if args.source else None
    if args.sync:
        if source is None:
            print("[!] --sync требует --from <путь к embryo-lab-data>")
            return 2
        code = sync(source)
        if code:
            return code
        print()
    return check(source)


if __name__ == "__main__":
    raise SystemExit(main())
