"""Загрузка конфигурации EWS из трёх слоёв.

settings.yaml          — переносимые параметры, едут в git, при переезде НЕ правятся
machine.local.yaml     — абсолютные пути ЭТОЙ машины + environment (в .gitignore)
settings.local.yaml    — секреты (в .gitignore)

Разделение не косметическое: перенос проекта на лабораторную машину обязан сводиться
к правке одного machine.local.yaml. Если параметр приходится менять при переезде — он
лежит не в том слое (см. docs/PORTING.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import yaml

# Корень проекта — на два уровня выше src/ews/config.py.
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

DRAFT = "draft"
PRODUCTION = "production"


class ConfigError(RuntimeError):
    """Конфигурация не пригодна для запуска. Всегда с указанием, что чинить."""


def _load_yaml(path: Path, *, required: bool, hint: str = "") -> dict:
    if not path.exists():
        if required:
            raise ConfigError(f"нет файла {path.relative_to(ROOT)}. {hint}".strip())
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Config:
    settings: dict
    machine: dict
    secrets: dict
    clinic_code: str = ""

    # -- окружение ---------------------------------------------------------
    @property
    def environment(self) -> str:
        env = str(self.machine.get("environment", "")).strip().lower()
        if env not in (DRAFT, PRODUCTION):
            raise ConfigError(
                "machine.local.yaml: environment обязан быть 'draft' или 'production'. "
                "Без него артефакт чернового стенда неотличим от лабораторного."
            )
        return env

    @property
    def is_draft(self) -> bool:
        return self.environment == DRAFT

    @property
    def site(self) -> str:
        """Где идёт прогон: `home` | `lab`. Только происхождение артефакта.

        Отделено от `environment` намеренно. Окружение — про ДАННЫЕ (боевые они
        или учебные), площадка — про МАШИНУ и её доступ к источникам. Смешивать их
        нельзя: когда дома и на работе одна клиника с одной выгрузкой, пороги
        переносятся законно, а свежесть логов инкубатора всё равно различается.
        """
        return str(self.machine.get("site", "")).strip().lower() or "unspecified"

    @property
    def alert_level_available(self) -> bool:
        """Может ли прогон вообще поднять уровень alert.

        Три независимых гейта, все обязаны быть открыты: боевое окружение,
        выполненная калибровка и совпадение окружения калибровки с текущим.
        Проверка живёт здесь, а не в вызывающем коде, чтобы её нельзя было забыть
        в одном из мест.
        """
        return self.alert_block_reason() is None

    def alert_block_reason(self) -> str | None:
        thresholds = self.settings.get("thresholds", {})
        if self.is_draft:
            return "environment=draft: черновой стенд не поднимает уровень alert"
        if not thresholds.get("calibrated", False):
            return "thresholds.calibrated=false: пороги не откалиброваны на этой истории"
        calibrated_env = str(thresholds.get("calibrated_environment", "")).strip().lower()
        if calibrated_env and calibrated_env != self.environment:
            # Пороги, подобранные на черновом стенде, непригодны в лаборатории:
            # они подогнаны под её объём и разброс (docs/PORTING.md §4). Проверка
            # автоматическая — иначе она держится на памяти переносящего.
            return (f"пороги откалиброваны в окружении '{calibrated_env}', "
                    f"а прогон идёт в '{self.environment}': нужен повторный бэкфит")
        return None

    # -- пути --------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        raw = self.machine.get("paths", {}).get("data_dir", "data")
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path

    def path(self, relative: str) -> Path:
        """Путь внутри проекта из настройки, заданной относительно корня."""
        p = Path(relative)
        return p if p.is_absolute() else ROOT / p

    def external(self, key: str) -> dict:
        """Секция machine.local.yaml с проверкой, что она заполнена."""
        section = self.machine.get(key)
        if not isinstance(section, dict):
            raise ConfigError(f"machine.local.yaml: нет секции '{key}'")
        return section

    # -- секреты -----------------------------------------------------------
    def secret(self, section: str, key: str = "token") -> str:
        value = str(self.secrets.get(section, {}).get(key, "")).strip()
        if not value:
            raise ConfigError(
                f"settings.local.yaml: не заполнен {section}.{key}. "
                f"Секреты живут только там — не в machine.local.yaml и не в коде."
            )
        return value

    def gate(self, name: str) -> bool:
        return bool(self.secrets.get("enabled", {}).get(name, False))


def load(clinic: str | None = None) -> Config:
    settings = _load_yaml(CONFIG_DIR / "settings.yaml", required=True)
    machine = _load_yaml(
        CONFIG_DIR / "machine.local.yaml", required=True,
        hint="скопируйте machine.local.yaml.example и заполните под эту машину.",
    )
    secrets = _load_yaml(
        CONFIG_DIR / "settings.local.yaml", required=False,
        hint="скопируйте settings.local.yaml.example, если нужны токены.",
    )
    code = clinic or str(machine.get("embryo_lab_data", {}).get("clinic") or "").strip()

    # Подобранные пороги живут в отдельном СГЕНЕРИРОВАННОМ файле на клинику, а не в
    # settings.yaml. Три причины, все существенные: запись бэкфитом в settings.yaml
    # стирала бы комментарии, в которых живёт обоснование параметров; файл в
    # .gitignore физически не может уехать в другую лабораторию вместе с
    # репозиторием, а чужие пороги там непригодны (docs/PORTING.md §4); и у каждой
    # клиники они свои, потому что зависят от её объёма.
    thresholds = _load_yaml(CONFIG_DIR / f"thresholds.{code}.local.yaml",
                            required=False) if code else {}
    settings.setdefault("thresholds", {}).update(thresholds or {})
    settings["thresholds"].setdefault("calibrated", False)

    cfg = Config(settings=settings, machine=machine, secrets=secrets, clinic_code=code)
    cfg.environment  # ранняя проверка: падать на старте, а не посреди прогона
    return cfg


def embryo_lab_data_on_path(cfg: Config) -> Path | None:
    """Совместимость: раньше подключала модули соседнего проекта.

    Теперь всё нужное вендорено в `ews.vendor`, и проект автономен — на
    лабораторной машине embryo-lab-data нет. Функция оставлена, чтобы не менять
    вызывающий код, и возвращает путь к источнику, только если он реально есть
    (нужен для `ews_sync_vendor.py --check`).
    """
    raw = str((cfg.machine.get("embryo_lab_data") or {}).get("project_dir") or "").strip()
    if not raw:
        return None
    project_dir = Path(raw)
    return project_dir if project_dir.exists() else None
