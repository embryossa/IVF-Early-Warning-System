"""Обновить локальный CSV-экспорт мастер-таблицы OPU из листа клиники.

    python scripts/ews_fetch_opu_sheet.py [--clinic <code>] [--dry-run]

Зачем отдельный шаг. `ews_sync_opu.py` читает CSV по пути
`google_sheet.export_path` и ничего не знает о том, откуда этот файл взялся: на
черновом стенде его клали руками через Drive MCP. На лабораторной машине руками
класть некому — суточная задача идёт по расписанию в 15:45, и без этого шага она
каждый день добросовестно синхронизировала бы ОДИН И ТОТ ЖЕ вчерашний файл,
сообщая «без изменений». Это ровно тот класс отказа, против которого написан
preflight: шаг не падает, а тихо не делает ничего.

Доступ по ссылке у листа закрыт (2026-11-22). Лист скачивается rclone как CSV
(экспортируется первый лист — Sheet1): сначала подключение сервисного аккаунта
`gdrive_sa:` (сервисный аккаунт клиники, только чтение, бессрочный ключ), затем
OAuth пользователя `gdrive:`; публичный CSV-экспорт — последний запасной путь.
Параметры листа (`sheet_id`, `gid`, необязательно `rclone_remotes`, `rclone`) — в
`config/machine.local.yaml`, секция `google_sheet`.

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


def _find_rclone(configured: str | None) -> str | None:
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    if configured and Path(configured).exists():
        return configured
    found = shutil.which("rclone")
    if found:
        return found
    root = Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"))
    hits = sorted(root.glob("Rclone.Rclone_*/rclone-*/rclone.exe")) if root.is_dir() else []
    return str(hits[-1]) if hits else None


def fetch_via_rclone(sheet_id: str, remotes: list[str], configured: str | None) -> tuple[str | None, str]:
    """CSV первого листа через rclone; (текст, источник) или (None, причина)."""
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    exe = _find_rclone(configured)
    if not exe:
        return None, "rclone не найден"
    errors = []
    for remote in remotes:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "opu.csv"
            r = subprocess.run([exe, "backend", "copyid", remote, sheet_id, str(dst),
                                "--drive-export-formats", "csv"], capture_output=True, timeout=180)
            if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                return dst.read_text(encoding="utf-8-sig"), f"rclone {remote}"
            errors.append(f"{remote}: код {r.returncode}")
    return None, "; ".join(errors)


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

    print(f"лист:   {sheet_id} (gid {gid})")
    print(f"приёмник: {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}")

    remotes = section.get("rclone_remotes") or ["gdrive_sa:", "gdrive:"]
    import os  # noqa: PLC0415
    snap = os.environ.get("OPU_SNAPSHOT_DIR")      # единый снимок оркестратора клиники
    if snap and (Path(snap) / "opu.csv").exists():
        text, source = (Path(snap) / "opu.csv").read_text(encoding="utf-8-sig"), f"снимок {Path(snap).name}"
    else:
        text, source = fetch_via_rclone(sheet_id, remotes, section.get("rclone"))
    if text is None:
        import requests  # noqa: PLC0415

        try:
            response = requests.get(EXPORT_URL.format(sheet_id=sheet_id, gid=gid), timeout=90)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 — мягкий шаг
            print(f"[warn] лист недоступен ни через rclone ({source}), ни по ссылке "
                  f"({exc.__class__.__name__}: {exc}).")
            print("       Прошлый экспорт оставлен как есть — синхронизация пойдёт по нему.")
            return 0
        response.encoding = "utf-8"
        text, source = response.text, "публичная ссылка"
    print(f"источник: {source}")
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
