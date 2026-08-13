"""Проверка и подготовка вендоренной модели конверта.

    python scripts/ews_setup_tabpfn.py --clinic <code> [--check] [--vendor-from <dir>]

`--check` — сказать, чего не хватает для запуска модели на ЭТОЙ машине и что именно
копировать. Ничего не меняет.

`--vendor-from` — собрать вендор из установленного окружения: скопировать скрипт,
веса и обучающую когорту в папку проекта. Выполняется на машине, где TabPFN уже
работает, ПЕРЕД переносом.

Почему это отдельный шаг, а не часть preflight: на лабораторной машине TabPFN нет и
не будет установлен отдельно. Единица переноса — папка проекта целиком, и она обязана
нести модель внутри себя. Молчаливая попытка скачать 233 МБ весов посреди суточного
прогона превратилась бы в загадочный таймаут на машине без интернета.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews.clinic import load_clinic  # noqa: E402
from ews.config import ROOT, ConfigError, load  # noqa: E402

VENDOR_DIR = ROOT / "external" / "tabpfn"
SCRIPT_NAME = "tabpfn3_ivf_regression_report_CPU_16K.py"
WEIGHTS_NAME = "tabpfn-v3-regressor-v3_default.ckpt"
DEV_COHORT_NAME = "dev_cohort.xlsx"


def _size(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.0f} МБ" if path.exists() else "—"


def check(cfg, clinic) -> int:
    """Что есть, чего нет и что копировать. Возвращает число проблем."""
    private = clinic.data_dir(cfg) / "local_private"
    venv_python = VENDOR_DIR / ".venv" / "Scripts" / "python.exe"
    items = [
        ("скрипт модели", VENDOR_DIR / SCRIPT_NAME, True,
         "скопируйте с рабочей машины через --vendor-from"),
        ("веса регрессора", VENDOR_DIR / "models" / WEIGHTS_NAME, True,
         "233 МБ; лежат в %APPDATA%/tabpfn на машине, где модель уже запускалась"),
        ("обучающая когорта", private / DEV_COHORT_NAME, True,
         "данные пациенток — копировать вручную, в git её нет и не будет"),
        ("окружение модели", venv_python, True,
         "py -3.12 -m venv external/tabpfn/.venv + pip install -r "
         "external/tabpfn/requirements.txt"),
        ("requirements", VENDOR_DIR / "requirements.txt", False, ""),
    ]

    problems = 0
    print("Вендор модели конверта:\n")
    for label, path, required, hint in items:
        exists = path.exists()
        mark = "ok  " if exists else ("НЕТ " if required else "нет ")
        detail = _size(path) if exists and path.is_file() else ""
        print(f"  [{mark}] {label:22s} {detail:>10s}  {path.relative_to(ROOT)}")
        if not exists and required:
            problems += 1
            print(f"         -> {hint}")

    print(f"\nкаталог весов передаётся модели через TABPFN_MODEL_CACHE_DIR — "
          f"скачивания из сети не будет")
    if problems:
        print(f"\nНЕ ГОТОВО: {problems} позиций. Пока их нет, конверт не "
              f"регистрируется, а прогон EWS идёт без риск-коррекции (мягкий шаг).")
    else:
        print("\nготово: модель запустится на этой машине")
    return problems


def vendor_from(source: Path, cfg, clinic) -> int:
    """Собрать вендор из установленного окружения."""
    if not source.exists():
        print(f"[!] нет каталога-источника: {source}")
        return 2

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    (VENDOR_DIR / "models").mkdir(exist_ok=True)
    private = clinic.data_dir(cfg) / "local_private"
    private.mkdir(parents=True, exist_ok=True)

    copied = 0
    script = source / SCRIPT_NAME
    if script.exists():
        shutil.copy2(script, VENDOR_DIR / SCRIPT_NAME)
        print(f"  + скрипт: {SCRIPT_NAME}")
        copied += 1

    import os

    appdata = os.environ.get("APPDATA")
    if appdata:
        weights = Path(appdata) / "tabpfn" / WEIGHTS_NAME
        if weights.exists():
            target = VENDOR_DIR / "models" / WEIGHTS_NAME
            if not target.exists():
                shutil.copy2(weights, target)
                print(f"  + веса: {WEIGHTS_NAME} ({_size(target)})")
                copied += 1
            else:
                print(f"  = веса уже на месте ({_size(target)})")

    # Обучающая когорта: PHI, поэтому только в local_private и только вручную
    # подтверждённым путём из конфига.
    dev_path = str((cfg.machine.get("tabpfn_regression") or {}).get("dev_cohort") or "")
    if dev_path and Path(dev_path).exists():
        shutil.copy2(dev_path, private / DEV_COHORT_NAME)
        print(f"  + обучающая когорта -> local_private/{DEV_COHORT_NAME}")
        print("    (данные пациенток: в git не попадает, в agent packet — тем более)")
        copied += 1
    elif dev_path:
        print(f"  [!] обучающая когорта не найдена: {dev_path}")
    else:
        print("  [!] machine.local.yaml: tabpfn_regression.dev_cohort не задан — "
              "обучающая когорта не скопирована")

    print(f"\nскопировано позиций: {copied}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", help="код профиля из config/clinics/")
    parser.add_argument("--check", action="store_true", help="только проверить")
    parser.add_argument("--vendor-from", help="каталог рабочего проекта TabPFN")
    args = parser.parse_args()

    try:
        cfg = load(args.clinic)
        clinic = load_clinic(cfg, args.clinic)
    except ConfigError as exc:
        print(f"[!] {exc}")
        return 2

    if args.vendor_from:
        code = vendor_from(Path(args.vendor_from), cfg, clinic)
        if code:
            return code
        print()
    return 1 if check(cfg, clinic) else 0


if __name__ == "__main__":
    raise SystemExit(main())
