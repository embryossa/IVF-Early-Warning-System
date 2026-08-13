"""Адаптер к Teable: переиспользует REST-клиент из embryo-lab-data.

Свой клиент здесь не пишется намеренно. teable/teable_common.py соседнего проекта
уже несёт пагинацию, батчи по 500, typecast и обработку загрузки вложений; вторая
реализация того же протокола разошлась бы с первой при первом же изменении Teable.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from .config import Config, ConfigError


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"не удалось загрузить {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def teable_common(cfg: Config):
    """REST-клиент Teable из вендоренной копии.

    Раньше загружался из соседнего проекта по пути из конфига. Так делать нельзя:
    на лабораторной машине embryo-lab-data нет, и система оказалась бы
    неработоспособной ровно там, где должна работать.
    """
    from .vendor import teable_client_impl  # noqa: PLC0415

    return teable_client_impl


def connect(cfg: Config, dry_run: bool = False) -> Any:
    """Готовый клиент к локальному Teable."""
    if not cfg.gate("teable"):
        raise ConfigError(
            "settings.local.yaml: enabled.teable=false. Включите после того, как вписан токен."
        )
    tc = teable_common(cfg)
    teable_cfg = cfg.external("teable")
    url = teable_cfg.get("url") or "http://127.0.0.1:3000"
    return tc.Teable(url, cfg.secret("teable"), dry_run=dry_run)


def resolve_tables(cfg: Config, client: Any, base_id: str) -> dict[str, str]:
    """Короткий ключ таблицы -> id. Имена вида '<key> — <русское название>'."""
    return {t["name"].split(" —")[0].strip(): t["id"] for t in client.tables(base_id)}


def save_ids(cfg: Config, base_id: str, opu_table_id: str, sync_table_id: str) -> None:
    """Записать id созданной базы обратно в machine.local.yaml.

    Идентификаторы базы машинозависимы (локальный Teable у каждой машины свой),
    поэтому их место именно там, а не в settings.yaml. Запись атомарная.
    """
    from .config import CONFIG_DIR

    path = CONFIG_DIR / "machine.local.yaml"
    text = path.read_text(encoding="utf-8")
    replacements = {
        "base_id": base_id,
        "opu_table_id": opu_table_id,
        "sync_table_id": sync_table_id,
    }
    lines = text.splitlines()
    in_teable = False
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not line.startswith((" ", "\t")):
            in_teable = stripped.startswith("teable:")
        if in_teable:
            for key, value in replacements.items():
                if stripped.startswith(f"{key}:"):
                    indent = line[: len(line) - len(line.lstrip())]
                    comment = line.split("#", 1)[1].strip() if "#" in line else ""
                    suffix = f"   # {comment}" if comment else ""
                    line = f'{indent}{key}: "{value}"{suffix}'
                    seen.add(key)
                    break
        out.append(line)

    missing = set(replacements) - seen
    if missing:
        # Ключа ещё нет в файле (например, sync_table_id в старом шаблоне) — дописываем
        # в конец секции teable, а не молча теряем.
        for index, line in enumerate(out):
            if line.strip().startswith("teable:"):
                insert_at = index + 1
                while insert_at < len(out) and out[insert_at].startswith((" ", "\t")):
                    insert_at += 1
                for key in sorted(missing):
                    out.insert(insert_at, f'  {key}: "{replacements[key]}"')
                break

    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)
