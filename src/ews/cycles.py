"""Загрузка циклов из Teable и разнесение их по чекпойнтам.

Псевдонимизация выполняется здесь, на границе ingest: дальше по конвейеру PHI не
уходит вообще. Всё, что видит слой оценки, панель и агенты, — `cycle_uid`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from .clinic import Clinic
from .config import Config

CHECKPOINT_OPU = "opu"
CHECKPOINT_FERTILISATION = "fertilisation"
CHECKPOINT_CLOSURE = "closure"
CHECKPOINT_NONE = "none"

# Поля таблицы OPU в Teable -> канонические имена. Зеркало opu_schema.SHEET_TO_FIELD,
# развёрнутое обратно: Teable хранит русские подписи, конвейер считает в канонических.
TEABLE_TO_CANONICAL = {
    "Возраст": "age", "№ попытки": "attempt_no", "Фолликулы": "follicles",
    "КАФ": "afc", "ОКК": "occ", "MII": "mii", "Инсеминировано": "inseminated",
    "2pN": "two_pn", "3pN": "three_pn", "1pN": "one_pn", "0pN": "zero_pn",
    "Deg": "degrade", "Дробление D3": "cleavage_d3", "Бласт": "bl_total",
    "Хор.бласт": "bl_good", "Криоконсервировано": "frozen", "Перенесено": "transferred",
    "День переноса": "transfer_day", "Исход": "transfer_outcome",
    "Метод": "fertilization_method", "Сперма": "sperm_source",
    "Эмбриолог": "embryologist", "Врач": "doctor", "AMH": "amh", "BMI": "bmi",
    "FSH": "fsh",
    # Итоговые вероятности ансамбля Digital Twin. Медленный контур подтверждения
    # по исходу берёт из DT только их — запускать сам пайплайн не требуется.
    "DIGITAL TWIN (лист)": "digital_twin",
}
NUMERIC_CANONICAL = {
    "age", "attempt_no", "follicles", "afc", "occ", "mii", "inseminated", "two_pn",
    "three_pn", "one_pn", "zero_pn", "degrade", "cleavage_d3", "bl_total",
    "bl_good", "frozen", "transferred", "transfer_day", "amh", "bmi", "fsh",
    "digital_twin",
}


@dataclass
class CycleFrame:
    """Канонический кадр циклов плюс метаданные загрузки."""
    data: pd.DataFrame
    source: str
    loaded_at: datetime

    def at_checkpoint(self, checkpoint: str) -> pd.DataFrame:
        return self.data.loc[self.data["checkpoint"] == checkpoint]


def _number(value: object) -> float | None:
    """Пустая ячейка -> пропуск, НЕ ноль: ноль здесь известное наблюдение."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def assign_checkpoint(row: pd.Series, today: datetime,
                      closure_offset_days: int) -> str:
    """До какого чекпойнта дошёл цикл ПО ДАННЫМ, а не только по календарю.

    Календарь один не годится: строку могут заполнить с опозданием. Поэтому стадия
    определяется наличием значений, а дата пункции лишь отсекает циклы, которые
    физически ещё не могли дойти до закрытия.
    """
    date_opu = row.get("date_opu")
    if pd.isna(date_opu):
        return CHECKPOINT_NONE
    closure_due = pd.Timestamp(date_opu) + timedelta(days=closure_offset_days)
    has_closure = any(row.get(field) is not None and not pd.isna(row.get(field))
                      for field in ("cleavage_d3", "bl_total", "bl_good", "frozen"))
    has_fertilisation = any(row.get(field) is not None and not pd.isna(row.get(field))
                            for field in ("two_pn", "three_pn", "one_pn", "zero_pn",
                                          "degrade"))
    if has_closure and pd.Timestamp(today) >= closure_due:
        return CHECKPOINT_CLOSURE
    if has_fertilisation:
        return CHECKPOINT_FERTILISATION
    return CHECKPOINT_OPU


def load_from_teable(cfg: Config, clinic: Clinic, client, table_id: str,
                     today: datetime | None = None) -> CycleFrame:
    """Прочитать таблицу OPU и привести к каноническому кадру без PHI.

    Из Teable забираются только нужные поля: ФИО, ДР и паспортный ID не запрашиваются
    вовсе. Связь с пациенткой держит `cycle_uid`, обратное сопоставление живёт в
    crosswalk и выполняется отдельным локальным шагом.
    """
    today = today or datetime.now()
    closure_offset = int(cfg.settings["checkpoints"]["t6"]["offset_days"])

    projection = ["cycle_uid", "Дата OPU", *TEABLE_TO_CANONICAL.keys()]
    records = client.all_records(table_id, projection=projection)

    rows: list[dict] = []
    for record in records:
        fields = record["fields"]
        row: dict = {"cycle_uid": str(fields.get("cycle_uid") or "").strip()}
        if not row["cycle_uid"]:
            continue
        raw_date = fields.get("Дата OPU")
        row["date_opu"] = pd.to_datetime(raw_date, errors="coerce", utc=True)
        if pd.notna(row["date_opu"]):
            # Teable отдаёт UTC; дата пункции — календарная дата клиники.
            row["date_opu"] = (row["date_opu"].tz_convert(clinic.timezone)
                               .tz_localize(None).normalize())
        for teable_name, canonical in TEABLE_TO_CANONICAL.items():
            value = fields.get(teable_name)
            row[canonical] = (_number(value) if canonical in NUMERIC_CANONICAL
                              else (str(value).strip() if value not in (None, "") else None))
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["cycle_uid", "date_opu", "checkpoint"])
    else:
        frame["checkpoint"] = frame.apply(
            lambda row: assign_checkpoint(row, today, closure_offset), axis=1)
        frame["closure_due"] = frame["date_opu"] + pd.Timedelta(days=closure_offset)
        frame["fr_denominator"] = fr_denominator(frame)
        frame = frame.sort_values("date_opu", kind="stable").reset_index(drop=True)
    return CycleFrame(data=frame, source=f"teable:{table_id}", loaded_at=today)


def fr_denominator(frame: pd.DataFrame) -> pd.Series:
    """Знаменатель FR и DegR — то же правило, что `_fr_denominator` в embryo.kpi.

    MII, если MII > 0, иначе число инсеминированных. Наблюдаемое «инсеминировано = 0»
    (витрификация ооцитов, отказ от оплодотворения: методы Cryo / NO) — инсеминации
    не было, FR для цикла не определён, а не равен нулю. Без этого такие циклы шли в
    карту FR как полный провал оплодотворения.
    """
    mii = pd.to_numeric(frame.get("mii"), errors="coerce")
    inseminated = pd.to_numeric(frame.get("inseminated"), errors="coerce")
    denominator = mii.where(mii.notna() & mii.gt(0), inseminated)
    return denominator.mask(inseminated.eq(0))


# Метрики: (числитель, знаменатель). Совпадают с embryo.kpi.KPI_SPECS, но здесь
# нужны на уровне ОДНОГО цикла — kpi.py считает когортно.
METRIC_PARTS = {
    "OCCR": ("occ", "follicles"),
    "MIIR": ("mii", "occ"),
    "FR": ("two_pn", "fr_denominator"),
    "ClR": ("cleavage_d3", "two_pn"),
    "BlR": ("bl_total", "two_pn"),
    "TGBDR": ("bl_good", "two_pn"),
    "DegR": ("degrade", "fr_denominator"),
    "3pNR": ("three_pn", "occ"),
}


def metric_counts(row: pd.Series, metric: str) -> tuple[float, float] | None:
    """(числитель, знаменатель) метрики для одного цикла; None — наблюдения нет.

    Знаменатель 0 или пропуск — наблюдения нет. Числитель больше знаменателя
    (ClR>100%, MII>ОКК) — штатная ситуация по правилам проекта, но для карты
    непригодна: доля вне [0,1] не является вероятностью Бернулли.
    """
    numerator_col, denominator_col = METRIC_PARTS[metric]
    numerator, denominator = row.get(numerator_col), row.get(denominator_col)
    if numerator is None or denominator is None:
        return None
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return None
    if numerator > denominator:
        return None
    return float(numerator), float(denominator)
