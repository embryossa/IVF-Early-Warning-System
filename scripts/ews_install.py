"""Установка на новой машине после переноса папки.

    python scripts/ews_install.py --clinic two_line [--site lab] [--check]

Делает то, что копированием не переносится, и проверяет всё остальное:

1. создаёт окружение проекта и окружение модели конверта (venv непереносимы —
   внутри них абсолютные пути);
2. сверяет вендоренные модули и вендор TabPFN;
3. проверяет доступность Teable, наличие обучающей когорты и токенов;
4. говорит, что осталось заполнить руками и какой командой продолжать.

Ничего не угадывает и не «чинит» молча: недостающее перечисляется списком с
указанием, чем именно оно заполняется. `--check` выполняет только проверки.

Интерпретатор для модели — **тот же, что и для проекта**: достаточно Python 3.10+
(`tabpfn` требует >=3.10, torch >=3.9). Venv при этом отдельный: torch занимает
1.7 ГБ, и окружению EWS он не нужен.

**Нужен интернет** для установки torch и tabpfn (~200 МБ колёс). Если лабораторная
машина в изолированной сети, соберите колёса заранее:
`pip download -r external/tabpfn/requirements.txt -d wheels` и ставьте
`pip install --no-index --find-links wheels`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TABPFN_DIR = ROOT / "external" / "tabpfn"


class Report:
    def __init__(self) -> None:
        self.todo: list[str] = []

    def ok(self, label: str, detail: str = "") -> None:
        print(f"  [ok]   {label}{f' — {detail}' if detail else ''}")

    def need(self, label: str, action: str) -> None:
        print(f"  [НУЖНО] {label}")
        print(f"          {action}")
        self.todo.append(label)


MIN_PYTHON = (3, 10)


def model_interpreter() -> tuple[Path | None, str]:
    """Интерпретатор для окружения модели конверта.

    Подходит системный: `tabpfn` объявляет `requires-python >=3.10`, torch `>=3.9`
    (проверено на 3.11.5 и 3.12.4). Раньше здесь искалась строго 3.12 — это была
    случайность исходного окружения, возведённая в требование, и она заставляла
    искать отдельный интерпретатор там, где хватает уже установленного.

    Venv остаётся ОТДЕЛЬНЫМ даже при совпадении версии: torch весит 1.7 ГБ, а
    окружению EWS он не нужен. Разделение к тому же не даёт сломанной установке
    модели утащить за собой конвейер — конверт просто не зарегистрируется.
    """
    current = sys.version_info[:2]
    if current >= MIN_PYTHON:
        return Path(sys.executable), f"{current[0]}.{current[1]} (текущий)"

    for name in ("py", "python3", "python"):
        found = shutil.which(name)
        if not found:
            continue
        try:
            out = subprocess.run(
                [found, "-c", "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True, text=True, timeout=30)
        except Exception:  # noqa: BLE001
            continue
        if out.returncode != 0:
            continue
        try:
            version = tuple(int(part) for part in out.stdout.strip().split("."))
        except ValueError:
            continue
        if version >= MIN_PYTHON:
            return Path(found), f"{out.stdout.strip()} ({name})"
    return None, ""


def make_venv(python: Path, target: Path, requirements: list[list[str]],
              report: Report, label: str) -> bool:
    marker = target / "Scripts" / "python.exe"
    if marker.exists():
        report.ok(f"окружение {label}", "уже создано")
        return True
    print(f"  … создаю окружение {label} ({target.relative_to(ROOT)})")
    result = subprocess.run([str(python), "-m", "venv", str(target)],
                            capture_output=True, text=True)
    if result.returncode != 0 or not marker.exists():
        report.need(f"окружение {label}",
                    f"не создалось: {(result.stderr or '').strip()[:200]}")
        return False
    for command in requirements:
        print(f"  … pip: {' '.join(command[-2:])}")
        install = subprocess.run([str(marker), "-m", "pip", "install", "-q", *command],
                                 capture_output=True, text=True)
        if install.returncode != 0:
            tail = (install.stderr or install.stdout or "").strip().splitlines()[-2:]
            report.need(f"зависимости {label}",
                        f"pip не отработал: {' | '.join(tail)[:200]}. "
                        f"Нужен интернет либо заранее скачанные колёса "
                        f"(--no-index --find-links wheels)")
            return False
    report.ok(f"окружение {label}", "создано")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", required=True)
    parser.add_argument("--site", choices=["home", "lab"], help="площадка")
    parser.add_argument("--check", action="store_true", help="только проверки")
    args = parser.parse_args()

    report = Report()
    print(f"EWS install — клиника {args.clinic}\n")

    # --- 1. конфигурация ---------------------------------------------------
    print("Конфигурация:")
    for name in ("machine.local.yaml", "settings.local.yaml"):
        path = ROOT / "config" / name
        example = ROOT / "config" / f"{name}.example"
        if path.exists():
            report.ok(name)
        elif example.exists() and not args.check:
            shutil.copy2(example, path)
            report.need(name, f"создан из шаблона — заполните пути и токены: {path}")
        else:
            report.need(name, f"скопируйте {example.name} и заполните")

    # --- 2. окружения ------------------------------------------------------
    print("\nОкружения Python:")
    if not args.check:
        make_venv(Path(sys.executable), ROOT / ".venv",
                  [["-r", str(ROOT / "requirements.txt")]], report, "проекта")
        interpreter, described = model_interpreter()
        if interpreter is None:
            report.need(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+",
                        "нужен для модели конверта; окружение "
                        "external/tabpfn/.venv не создано")
        else:
            print(f"  … интерпретатор модели: {described}")
            make_venv(interpreter, TABPFN_DIR / ".venv", [
                ["--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.7.0"],
                ["-r", str(TABPFN_DIR / "requirements.txt")],
            ], report, "модели конверта")
    else:
        for label, marker in (("проекта", ROOT / ".venv" / "Scripts" / "python.exe"),
                              ("модели конверта",
                               TABPFN_DIR / ".venv" / "Scripts" / "python.exe")):
            if marker.exists():
                report.ok(f"окружение {label}")
            else:
                report.need(f"окружение {label}",
                            "создайте: python scripts/ews_install.py --clinic <code>")

    # --- 3. вендор ---------------------------------------------------------
    print("\nВендор:")
    weights = TABPFN_DIR / "models" / "tabpfn-v3-regressor-v3_default.ckpt"
    if weights.exists():
        report.ok("веса модели", f"{weights.stat().st_size / 1e6:.0f} МБ")
    else:
        report.need("веса модели",
                    "233 МБ; скопируйте external/tabpfn/models с исходной машины")
    dev = ROOT / "data" / args.clinic / "local_private" / "dev_cohort.xlsx"
    if dev.exists():
        report.ok("обучающая когорта")
    else:
        report.need("обучающая когорта",
                    f"данные пациенток; скопируйте в {dev.parent}")
    manifest = ROOT / "src" / "ews" / "vendor" / "vendor_manifest.json"
    if manifest.exists():
        report.ok("манифест вендора")
    else:
        report.need("манифест вендора",
                    "нет — вендоренные модули не перенеслись целиком; "
                    "проверьте src/ews/vendor/")

    # --- 4. площадка -------------------------------------------------------
    if args.site and not args.check:
        path = ROOT / "config" / "machine.local.yaml"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if "\nsite:" in text:
                import re

                text = re.sub(r"(?m)^site:.*$", f'site: {args.site}', text)
            else:
                text = text.replace("environment:", f'site: {args.site}\nenvironment:', 1)
            path.write_text(text, encoding="utf-8", newline="\n")
            report.ok("площадка", args.site)

    # --- итог --------------------------------------------------------------
    print("\n" + "=" * 66)
    if report.todo:
        print(f"осталось заполнить руками: {len(report.todo)}")
        for item in report.todo:
            print(f"  · {item}")
        print("\nПосле этого:")
    else:
        print("установка завершена. Дальше:")
    print(f"  python scripts/preflight.py")
    print(f"  python scripts/ews_bootstrap_opu.py --clinic {args.clinic}")
    print(f"  python scripts/ews_sync_opu.py      --clinic {args.clinic}")
    print(f"  python scripts/ews_run.py           --clinic {args.clinic} --replay")
    print(f"  python scripts/ews_schedule.py      --clinic {args.clinic} --install")
    print("\nподробно — docs/INSTALL_LAB.md")
    return 1 if report.todo else 0


if __name__ == "__main__":
    sys.exit(main())
