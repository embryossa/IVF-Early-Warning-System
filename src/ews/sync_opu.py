"""Синхронизация мастер-таблицы OPU в локальный Teable.

Направление одностороннее: лист — источник, Teable — зеркало. Upsert по ключу
`ID#Дата OPU` (тот же составной ключ, что и `_key` в embryo-lab-data: у повторных
попыток одной пациентки общий паспортный ID, и join по нему схлопнул бы разные циклы
в один).

Строка не перезаписывается, если её содержимое не изменилось: сравнивается хеш
значимых полей. Это не оптимизация ради скорости — это то, что делает поле
«Обновлено» осмысленным, а журнал синхронизаций пригодным для разбора.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import re

import pandas as pd

from .config import Config
from .opu_schema import (DATE_FIELDS, FUNNEL_T1, FUNNEL_T6, NUMERIC_FIELDS,
                         SHEET_TO_FIELD)

# Ключ синхронизации. Дата в ключе — как записана в листе (dd.mm.YYYY), чтобы ключ
# не менялся от того, как его сегодня распарсили.
KEY_SEPARATOR = "#"


@dataclass
class SyncReport:
    run_id: str
    started_at: datetime
    source: str
    environment: str
    read: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    notes: list[str] = dc_field(default_factory=list)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def as_record(self) -> dict:
        return {
            "Прогон": self.run_id,
            "Начат": self.started_at.isoformat(timespec="seconds"),
            "Источник": self.source,
            "Строк прочитано": self.read,
            "Создано": self.created,
            "Обновлено": self.updated,
            "Без изменений": self.unchanged,
            "Пропущено": self.skipped,
            "Окружение": self.environment,
            "Замечания": "\n".join(self.notes) if self.notes else "",
        }


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _to_number(value: object) -> float | None:
    """Число из ячейки листа. Запятая как десятичный разделитель — норма.

    Пустая ячейка -> None (пропуск), а НЕ 0: ноль здесь известное наблюдение
    (например, 0 бластоцист), и подменять им пропуск запрещено правилами проекта.
    """
    text = _clean(value)
    if not text:
        return None
    text = text.replace(" ", "").replace(" ", "").replace(",", ".")
    if not re.fullmatch(r"-?\d+(\.\d+)?", text):
        return None
    return float(text)


def _to_date(value: object) -> str | None:
    text = _clean(value)
    if not text:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def build_key(row: pd.Series) -> str:
    patient_id = _clean(row.get("ID"))
    date_opu = _clean(row.get("Date OPU"))
    if not patient_id or not date_opu:
        return ""
    return f"{patient_id.upper()}{KEY_SEPARATOR}{date_opu}"


def checkpoint_of(fields: dict, today: datetime | None = None) -> str:
    """До какого чекпойнта дошёл цикл ПО ДАННЫМ, а не по календарю.

    Календарь один здесь не годится: строка может быть заполнена с опозданием.
    Поэтому стадия определяется наличием значений, а дата пункции лишь отсекает
    циклы, которые физически ещё не могли дойти до закрытия.
    """
    today = today or datetime.now()
    date_opu = fields.get("Дата OPU")
    if not date_opu:
        return "— не наступил"
    opu = datetime.fromisoformat(str(date_opu))
    has_t6 = any(fields.get(name) is not None for name in FUNNEL_T6)
    has_t1 = any(fields.get(name) is not None for name in FUNNEL_T1)
    if has_t6 and today >= opu + timedelta(days=6):
        return "T+6 закрытие"
    if has_t1:
        return "T+1 оплодотворение"
    return "T0 пункция"


def row_to_fields(row: pd.Series, cycle_uid: str, environment: str,
                  today: datetime | None = None) -> dict:
    fields: dict[str, object] = {}
    for sheet_name, field_name in SHEET_TO_FIELD.items():
        raw = row.get(sheet_name)
        if field_name in DATE_FIELDS:
            value = _to_date(raw)
        elif field_name in NUMERIC_FIELDS:
            value = _to_number(raw)
        else:
            value = _clean(raw) or None
        fields[field_name] = value

    fields["Ключ"] = build_key(row)
    fields["cycle_uid"] = cycle_uid
    fields["Чекпойнт"] = checkpoint_of(fields, today=today)
    fields["Окружение"] = environment
    return fields


def content_hash(fields: dict) -> str:
    """Хеш значимого содержимого строки.

    Служебные поля исключены намеренно: иначе «Обновлено» менялось бы каждый прогон
    само от себя, и журнал перестал бы отличать реальную правку от холостого хода.
    """
    ignored = {"Хеш строки", "Обновлено", "Окружение", "Чекпойнт"}
    payload = "|".join(
        f"{name}={'' if fields.get(name) is None else fields.get(name)}"
        for name in sorted(fields)
        if name not in ignored
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_sheet_export(path: str | Path) -> pd.DataFrame:
    """Читает CSV-экспорт листа. Пустые хвостовые строки отбрасываются."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    has_patient = (frame["ID"].str.strip() != "") | (
        frame["Patient Full Name"].str.strip() != ""
    )
    return frame.loc[has_patient].copy()


def sync(cfg: Config, client, table_ids: dict[str, str], frame: pd.DataFrame,
         secret: bytes, source: str, dry_run: bool = False) -> SyncReport:
    """Upsert строк листа в таблицу OPU. Возвращает отчёт синхронизации."""
    from .config import ROOT  # локально: избегаем цикла на уровне модуля

    started = datetime.now()
    report = SyncReport(
        run_id=started.strftime("%Y%m%dT%H%M%S"),
        started_at=started,
        source=source,
        environment=cfg.environment,
        read=len(frame),
    )

    # embryo-lab-data даёт функцию HMAC; свою реализацию не пишем.
    from .vendor.privacy import make_cycle_uid  # noqa: PLC0415

    opu_table = table_ids["OPU"]
    existing = {}
    for record in client.all_records(opu_table, projection=["Ключ", "Хеш строки"]):
        key = _clean(record["fields"].get("Ключ"))
        if key:
            existing[key] = (record["id"], _clean(record["fields"].get("Хеш строки")))

    to_create: list[dict] = []
    to_update: list[tuple[str, dict]] = []
    seen_keys: set[str] = set()

    for _, row in frame.iterrows():
        key = build_key(row)
        if not key:
            report.skipped += 1
            report.note(f"строка без ID или даты пункции пропущена (# {_clean(row.get('#'))})")
            continue
        if key in seen_keys:
            # Дубль в самом листе. Не сливаем и не выбираем «полную» строку молча:
            # какая из них верна, решает владелец.
            report.skipped += 1
            report.note(f"дубль ключа в листе: {key} — строка пропущена, разберите вручную")
            continue
        seen_keys.add(key)

        fields = row_to_fields(row, make_cycle_uid(key, secret), cfg.environment)
        fields["Хеш строки"] = content_hash(fields)
        fields["Обновлено"] = datetime.now().isoformat(timespec="seconds")

        if key not in existing:
            to_create.append(fields)
        elif existing[key][1] != fields["Хеш строки"]:
            to_update.append((existing[key][0], fields))
        else:
            report.unchanged += 1

    if not dry_run:
        if to_create:
            client.create_records(opu_table, to_create)
        if to_update:
            client.update_records(opu_table, to_update)
    report.created = len(to_create)
    report.updated = len(to_update)

    stale = set(existing) - seen_keys
    if stale:
        # Строка есть в Teable, но исчезла из листа. Не удаляем: удаление данных
        # пациентки по расхождению выгрузок — не то решение, которое принимает скрипт.
        report.note(
            f"{len(stale)} строк есть в Teable, но отсутствуют в листе — оставлены как есть"
        )

    if not dry_run and "Sync" in table_ids:
        client.create_records(table_ids["Sync"], [report.as_record()])

    _ = ROOT  # путь корня используется вызывающим кодом для журнала
    return report
