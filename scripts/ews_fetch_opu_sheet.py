"""Обновить локальный CSV-экспорт мастер-таблицы OPU из листа клиники.

    python scripts/ews_fetch_opu_sheet.py [--clinic <code>] [--dry-run]

Зачем отдельный шаг. `ews_sync_opu.py` читает CSV по пути
`google_sheet.export_path` и ничего не знает о том, откуда этот файл взялся: на
черновом стенде его клали руками через Drive MCP. На лабораторной машине руками
класть некому — суточная задача идёт по расписанию в 15:45, и без этого шага она
каждый день добросовестно синхронизировала бы ОДИН И ТОТ ЖЕ вчерашний файл,
сообщая «без изменений». Это ровно тот класс отказа, против которого написан
preflight: шаг не падает, а тихо не делает ничего.

Лист клиники «OPU table» публичный, поэтому берётся прямой CSV-экспорт Google
Sheets — ни ключей, ни OAuth не требуется. Параметры листа (`sheet_id`, `gid`) —
в `config/machine.local.yaml`, секция `google_sheet`.

Запись атомарная и с проверкой: файл заменяется только если ответ похож на CSV
той же таблицы (есть колонки `ID` и `Date OPU`) и содержит хотя бы одну строку с
пациенткой. Иначе на диске остаётся прошлый экспорт: страница логина Google,
записанная поверх выгрузки, выглядела бы как «в таблице не осталось циклов».

Сеть недоступна — шаг МЯГКИЙ: печатает причину и возвращает 0, оставляя прошлый
CSV. Синхронизация тогда честно скажет «без изменений», и это правда.
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews.config import ConfigError, load  # noqa: E402

EXPORT_URL = ("https://docs.google.com/spreadsheets/d/{sheet_id}"
              "/export?format=csv&gid={gid}")
# Колонки, по которым узнаётся нужный лист. Имена — как в выгрузке (columns.yaml).
REQUIRED_COLUMNS = ("ID", "Patient Full Name", "Date OPU")


def looks_like_opu_export(text: str) -> tuple[bool, str]:
    """Похоже ли скачанное на экспорт мастер-таблицы, а не на страницу ошибки."""
    try:
        reader = csv.DictReader(io.StringIO(text))
        header = reader.fieldnames or []
    except csv.Error as exc:
        return False, f"не разбирается как CSV: {exc}"

    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        return False, f"в ответе нет колонок {', '.join(missing)} — это не тот лист"

    for row in reader:
        if (row.get("ID") or "").strip() or (row.get("Patient Full Name") or "").strip():
            return True, ""
    return False, "нет ни одной строки с пациенткой"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", help="код профиля из config/clinics/")
    parser.add_argument("--dry-run", action="store_true",
                        help="скачать и проверить, ничего не записывая")
    args = parser.parse_args()

    try:
        cfg = load(args.clinic)
        section = cfg.external("google_sheet")
    except ConfigError as exc:
        print(f"[!] {exc}")
        return 2

    sheet_id = str(section.get("sheet_id") or "").strip()
    gid = str(section.get("gid") or "").strip()
    target = cfg.path(str(section.get("export_path") or "").strip())
    if not sheet_id or not gid or not str(target):
        print("[!] machine.local.yaml: заполните google_sheet.sheet_id / gid / export_path")
        return 2

    import requests  # noqa: PLC0415

    url = EXPORT_URL.format(sheet_id=sheet_id, gid=gid)
    print(f"лист:   {sheet_id} (gid {gid})")
    print(f"приёмник: {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}")

    try:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — мягкий шаг
        print(f"[warn] лист недоступен ({exc.__class__.__name__}: {exc}).")
        print("       Прошлый экспорт оставлен как есть — синхронизация пойдёт по нему.")
        return 0

    response.encoding = "utf-8"
    text = response.text
    valid, reason = looks_like_opu_export(text)
    if not valid:
        print(f"[warn] ответ не принят: {reason}")
        print("       Прошлый экспорт оставлен как есть. Проверьте доступ к листу.")
        return 0

    previous = target.read_bytes() if target.exists() else None
    payload = text.encode("utf-8")
    if previous == payload:
        print(f"без изменений: {len(payload)} байт, лист не менялся с прошлого прогона")
        return 0

    if args.dry_run:
        was = f"{len(previous)} байт" if previous is not None else "файла не было"
        print(f"[dry] записал бы {len(payload)} байт (было: {was})")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)
    was = f"{len(previous)} байт" if previous is not None else "новый файл"
    print(f"обновлено: {len(payload)} байт (было: {was})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
