"""Проверка окружения перед прогоном EWS.

    python scripts/preflight.py

Возвращает 0, только если всё, что нужно для прогона, на месте. Не формальность:
в соседнем проекте расхождение окружения с requirements.txt трижды давало
МОЛЧАЛИВО пропущенный шаг вместо ошибки, и однажды это стоило неверного вывода в
отчёте, а не только времени. Поэтому пакеты, чьё отсутствие даёт мягкий отказ,
проверяются отдельной секцией.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews.config import ConfigError, embryo_lab_data_on_path, load  # noqa: E402

# Пакеты, без которых прогон падает громко.
HARD = ["pandas", "numpy", "yaml", "requests", "scipy"]
# Пакеты, отсутствие которых даёт тихо пропущенный шаг или медленный откат.
SOFT = {
    "python_calamine": "чтение Excel уйдёт на медленный откат openpyxl, без ошибки",
    "openpyxl": "не прочитаются .xlsx-выгрузки",
}

# Вендоренные модули, на которых стоит EWS. Живут внутри проекта: на лабораторной
# машине embryo-lab-data нет, и зависеть от него нельзя.
VENDOR_MODULES = ["ews.vendor.kpi", "ews.vendor.privacy", "ews.vendor.incubator",
                  "ews.vendor.lots", "ews.vendor._io",
                  "ews.vendor.teable_client_impl"]


class Check:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, label: str, detail: str = "") -> None:
        print(f"  [ok]   {label}{f' — {detail}' if detail else ''}")

    def fail(self, label: str, detail: str) -> None:
        print(f"  [FAIL] {label} — {detail}")
        self.failures.append(label)

    def warn(self, label: str, detail: str) -> None:
        print(f"  [warn] {label} — {detail}")
        self.warnings.append(label)


def check_packages(check: Check) -> None:
    print("\nПакеты:")
    for name in HARD:
        try:
            importlib.import_module(name)
            check.ok(name)
        except ImportError:
            check.fail(name, "нет; выполните pip install -r requirements.txt")
    for name, consequence in SOFT.items():
        try:
            importlib.import_module(name)
            check.ok(name)
        except ImportError:
            check.warn(name, f"нет — {consequence}")


def check_config(check: Check):
    print("\nКонфигурация:")
    try:
        cfg = load()
    except ConfigError as exc:
        check.fail("config", str(exc))
        return None
    check.ok("три слоя конфигурации", f"environment={cfg.environment}")

    reason = cfg.alert_block_reason()
    if reason:
        check.ok("гейт alert", f"закрыт — {reason}")
    else:
        check.ok("гейт alert", "открыт: боевое окружение и калибровка выполнены")
    return cfg


def check_paths(check: Check, cfg) -> None:
    print("\nПути (machine.local.yaml):")
    required = [("embryo_lab_data", "project_dir")]
    optional = [("digital_twin", "project_dir"), ("tabpfn_regression", "project_dir"),
                ("consensus", "pdf_dir"), ("miri_logs", "dump_dir")]

    for section, key in required:
        raw = str(cfg.machine.get(section, {}).get(key, "")).strip()
        if section == "embryo_lab_data":
            # Источник вендора. Нужен только на машине разработки; в лаборатории
            # его нет, и это штатно.
            if not raw:
                check.ok(f"{section}.{key}", "пуст — штатно, проект автономен")
            elif not Path(raw).exists():
                check.warn(f"{section}.{key}", "указан, но не существует")
            else:
                check.ok(f"{section}.{key}", "источник вендора доступен")
            continue
        if not raw:
            check.fail(f"{section}.{key}", "не заполнен")
        elif not Path(raw).exists():
            check.fail(f"{section}.{key}", f"не существует: {raw}")
        else:
            check.ok(f"{section}.{key}")

    for section, key in optional:
        raw = str(cfg.machine.get(section, {}).get(key, "")).strip()
        enabled = cfg.machine.get(section, {}).get("enabled", True)
        if not raw:
            check.warn(f"{section}.{key}", "пуст — соответствующий шаг будет пропущен")
        elif not Path(raw).exists():
            (check.fail if enabled else check.warn)(f"{section}.{key}",
                                                    f"не существует: {raw}")
        else:
            check.ok(f"{section}.{key}")


def check_embryo_modules(check: Check, cfg) -> None:
    print("\nВендоренные модули (проект автономен):")
    for name in VENDOR_MODULES:
        try:
            importlib.import_module(name)
            check.ok(name)
        except Exception as exc:  # noqa: BLE001
            check.fail(name, f"не импортируется: {exc}")


def check_teable(check: Check, cfg) -> None:
    print("\nTeable:")
    if not cfg.gate("teable"):
        check.warn("teable", "enabled.teable=false — шаг синхронизации пропускается")
        return
    try:
        from ews import teable_client  # noqa: PLC0415

        client = teable_client.connect(cfg)
        base_id = str(cfg.external("teable").get("base_id") or "").strip()
        if not base_id:
            check.warn("teable.base_id", "пуст — выполните scripts/ews_bootstrap_opu.py")
            return
        tables = teable_client.resolve_tables(cfg, client, base_id)
        if "OPU" not in tables:
            check.fail("таблица OPU", "нет в базе; выполните ews_bootstrap_opu.py")
            return
        check.ok("таблица OPU", f"{client.row_count(tables['OPU'])} строк")
        if "Sync" in tables:
            check.ok("таблица Sync", f"{client.row_count(tables['Sync'])} прогонов")
    except Exception as exc:  # noqa: BLE001
        check.fail("teable", f"недоступен: {exc}")


def main() -> int:
    print("EWS preflight")
    check = Check()
    check_packages(check)
    cfg = check_config(check)
    if cfg is not None:
        check_paths(check, cfg)
        check_embryo_modules(check, cfg)
        check_teable(check, cfg)

    print()
    if check.failures:
        print(f"НЕ ГОТОВО: {len(check.failures)} проблем — {', '.join(check.failures)}")
        return 1
    if check.warnings:
        print(f"готово, с оговорками ({len(check.warnings)}): {', '.join(check.warnings)}")
    else:
        print("готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
