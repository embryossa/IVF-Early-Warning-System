"""Сборка комплекта для переноса на другую машину.

    python scripts/ews_package.py --to D:/ews_transfer [--with-sensitive]

Кладёт рядом два комплекта, а не один, и это главное решение скрипта:

* **`code/`** — код, конфигурация, вендоренные модули, веса модели. ~250 МБ.
  Данных пациенток и токенов здесь нет.
* **`calibration/`** — пороги, опорные параметры, зарегистрированные конверты.
  PHI внутри нет (проверено: только `cycle_uid` и числа), а перенос экономит на
  новой машине часы — иначе бэкфит и регистрация конвертов считаются заново.
  Переносить законно только когда клиника и данные те же.
* **`sensitive/`** — обучающая когорта и токены. Собирается только с
  `--with-sensitive` и сопровождается предупреждением.

Почему раздельно. В папке проекта лежат `dev_cohort.xlsx` (16 тысяч циклов),
выгрузка OPU с ФИО и паспортными номерами, crosswalk псевдонимов и токены Teable и
Airtable. Всё это уезжает вместе с папкой, если копировать её целиком, — на флешке,
которую носят между домом и работой. Разделение не мешает переносу (обе папки просто
копируются подряд), но делает видимым, что именно вы несёте.

**Виртуальные окружения не копируются никогда.** Внутри них абсолютные пути, на
другой машине они не работают. 1.7 ГБ из 2.0 ГБ папки — это они; пересоздаются одной
командой (`scripts/ews_install.py`).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]

# Что едет в комплекте кода.
CODE_INCLUDE = [
    "src", "scripts", "docs", "config/clinics", "config/consensus",
    "config/settings.yaml", "config/machine.local.yaml.example",
    "config/settings.local.yaml.example", "tests",
    "requirements.txt", "README.md", "PLAN.md", "CLAUDE.md", ".gitignore",
    "external/tabpfn/tabpfn3_ivf_regression_report_CPU_16K.py",
    "external/tabpfn/tabpfn3_ivf_kpi_regression.py",
    "external/tabpfn/experiment_matched_subsample.py",
    "external/tabpfn/requirements.txt", "external/tabpfn/README.md",
    "external/tabpfn/models",
]

# Результаты калибровки. PHI не содержат — только cycle_uid и числа.
CALIBRATION = {
    "config/thresholds.{clinic}.local.yaml":
        "пороги карт — перенос законен, пока клиника и данные те же",
    "data/{clinic}/baseline": "опорные параметры, калибровка конверта, манифесты",
    "data/{clinic}/ews_envelopes.csv":
        "зарегистрированные конверты (заморожены, пересчёту не подлежат)",
    "data/{clinic}/state/confounder_baseline.json":
        "границы «необычного» по среде и лотам",
}

# Что едет только с --with-sensitive, с предупреждением.
SENSITIVE = {
    "data/{clinic}/local_private/dev_cohort.xlsx":
        "обучающая когорта TabPFN — 16 тысяч циклов пациенток",
    "config/settings.local.yaml": "токены Teable и Airtable",
}

# Состояние карт НЕ переносится намеренно: две машины идут по истории независимо,
# и подмена состояния чужим сломала бы память о том, какие циклы уже оценены.
# Пересобирается на месте: ews_run.py --replay.
NEVER_TRANSFER = {
    "data/{clinic}/state/ews_charts_state.json": "состояние карт — пересобрать --replay",
    "config/machine.local.yaml": "пути этой машины — заполняется заново из .example",
    "data/{clinic}/local_private/cycle_uid_crosswalk.csv": "crosswalk псевдонимов",
}

NEVER = ["**/.venv", "**/__pycache__", "**/.pytest_cache", "**/*.tmp"]


def copy_entry(source: Path, target: Path) -> int:
    if not source.exists():
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".venv", "__pycache__",
                                                      ".pytest_cache", "*.tmp"))
        return sum(1 for p in target.rglob("*") if p.is_file())
    shutil.copy2(source, target)
    return 1


def size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, help="куда собрать комплект")
    parser.add_argument("--clinic", default="two_line")
    parser.add_argument("--with-sensitive", action="store_true",
                        help="добавить обучающую когорту, токены и пороги")
    args = parser.parse_args()

    destination = Path(args.to)
    code_dir = destination / "code"
    print(f"комплект: {destination}")

    total = 0
    for relative in CODE_INCLUDE:
        source = ROOT / relative
        copied = copy_entry(source, code_dir / relative)
        total += copied
        if copied:
            print(f"  + {relative:58s} {size_mb(source):8.1f} МБ")
        elif not source.exists():
            print(f"  [нет] {relative}")

    print(f"\nкод: {total} файлов, {size_mb(code_dir):.0f} МБ")

    calibration_dir = destination / "calibration"
    print("\nКалибровка (PHI нет, перенос экономит часы пересчёта):")
    for template, what in CALIBRATION.items():
        relative = template.format(clinic=args.clinic)
        source = ROOT / relative
        if copy_entry(source, calibration_dir / relative):
            print(f"  + {relative:52s} — {what}")
        else:
            print(f"  [нет] {relative}")
    print(f"  всего {size_mb(calibration_dir):.1f} МБ")

    print("\nНЕ переносится намеренно:")
    for template, why in NEVER_TRANSFER.items():
        print(f"  · {template.format(clinic=args.clinic):52s} — {why}")

    if args.with_sensitive:
        sensitive_dir = destination / "sensitive"
        print("\nЧУВСТВИТЕЛЬНЫЙ КОМПЛЕКТ — данные пациенток и секреты:")
        for template, what in SENSITIVE.items():
            relative = template.format(clinic=args.clinic)
            source = ROOT / relative
            if copy_entry(source, sensitive_dir / relative):
                print(f"  + {relative:52s} — {what}")
            else:
                print(f"  [нет] {relative}")
        print(f"\n  всего {size_mb(sensitive_dir):.1f} МБ")
        print("\n  Это данные пациенток и рабочие токены на съёмном носителе.")
        print("  Носитель стоит зашифровать, а после переноса — удалить комплект")
        print("  с него, а не оставлять «на всякий случай».")
    else:
        print("\nчувствительная часть НЕ собрана (нужен --with-sensitive).")
        print("Без неё на новой машине не будет: обучающей когорты для конверта,")
        print("токенов Teable/Airtable и откалиброванных порогов.")

    print("\nвиртуальные окружения не копировались — они непереносимы "
          "(абсолютные пути внутри).")
    print("На новой машине: python scripts/ews_install.py --clinic "
          f"{args.clinic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
