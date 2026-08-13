"""Профиль лаборатории: всё, что различается между клиниками.

Разделение, на котором стоит воспроизводимость: `config/settings.yaml` — МЕТОД
(математика карт, целевые частоты, правила чекпойнтов), одинаковый везде;
`config/clinics/<code>/` — ЛАБОРАТОРИЯ (колонки выгрузки, опорное окно, набор карт).
Запуск на новой лаборатории не требует правок кода — только нового профиля.
См. docs/RUNBOOK.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import CONFIG_DIR, Config, ConfigError

CLINICS_DIR = CONFIG_DIR / "clinics"


@dataclass(frozen=True)
class Clinic:
    code: str
    profile: dict
    columns: dict

    # -- идентификация -----------------------------------------------------
    @property
    def name(self) -> str:
        return self.profile.get("name", self.code)

    @property
    def timezone(self) -> str:
        return self.profile.get("timezone", "Asia/Tashkent")

    @property
    def base_name(self) -> str:
        return (self.profile.get("teable", {}).get("base_name")
                or f"EWS — OPU master ({self.name})")

    # -- опорное окно ------------------------------------------------------
    @property
    def reference_window(self) -> dict:
        window = self.profile.get("reference_window") or {}
        if not window.get("start"):
            raise ConfigError(
                f"клиника {self.code}: не задано reference_window.start. "
                f"Без опорного окна нет центра относительного контура."
            )
        return window

    def window_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        window = self.reference_window
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"]) if window.get("end") else pd.Timestamp(date.today())
        return start, end

    @property
    def declared_shifts(self) -> dict:
        return self.reference_window.get("declared_shifts") or {}

    # -- карты -------------------------------------------------------------
    @property
    def cusum_charts(self) -> list[str]:
        charts = list((self.profile.get("charts") or {}).get("cusum") or [])
        forbidden = set((self.profile.get("charts") or {}).get("subjective_denominator") or {})
        # Проверка, а не соглашение: метрика с субъективным знаменателем не может
        # попасть на карту по недосмотру при копировании профиля.
        collision = sorted(set(charts) & forbidden)
        if collision:
            raise ConfigError(
                f"клиника {self.code}: {', '.join(collision)} перечислены и в charts.cusum, "
                f"и в subjective_denominator. Карта на метрике с субъективным знаменателем "
                f"ловит изменения в том, как считают, а не в том, как работают "
                f"(docs/METHODOLOGY.md §5.5)."
            )
        if not charts:
            raise ConfigError(f"клиника {self.code}: charts.cusum пуст — нечего калибровать")
        return charts

    @property
    def subjective_denominator(self) -> dict:
        return (self.profile.get("charts") or {}).get("subjective_denominator") or {}

    @property
    def amh_constant(self) -> float | None:
        """Константа AMH для пропусков; None — приор на AMH не считается вовсе."""
        value = (self.profile.get("imputation") or {}).get("amh_constant")
        return float(value) if value is not None else None

    @property
    def attempt_default(self) -> float | None:
        """Чем заполнять пустой номер попытки; None — оставлять пропуском."""
        value = (self.profile.get("imputation") or {}).get("attempt_default")
        return float(value) if value is not None else None

    def source_enabled(self, name: str) -> bool:
        return bool((self.profile.get("sources") or {}).get(name, False))

    # -- колонки -----------------------------------------------------------
    def source_column(self, canonical: str) -> str | None:
        return (self.columns.get("canonical") or {}).get(canonical)

    @property
    def numeric(self) -> dict[str, int]:
        return self.columns.get("numeric") or {}

    @property
    def date_columns(self) -> list[str]:
        return list(self.columns.get("dates") or [])

    @property
    def date_formats(self) -> list[str]:
        return list(self.columns.get("date_formats") or ["%d.%m.%Y"])

    @property
    def phi_columns(self) -> list[str]:
        return list(self.columns.get("phi") or [])

    def checkpoint_fields(self, checkpoint: str) -> list[str]:
        return list((self.columns.get("checkpoint_fields") or {}).get(checkpoint) or [])

    # -- пути артефактов ---------------------------------------------------
    def data_dir(self, cfg: Config) -> Path:
        """Артефакты каждой клиники лежат отдельно: смешать их нельзя."""
        return cfg.data_dir / self.code

    def thresholds_path(self) -> Path:
        return CONFIG_DIR / f"thresholds.{self.code}.local.yaml"


def available() -> list[str]:
    if not CLINICS_DIR.exists():
        return []
    return sorted(p.name for p in CLINICS_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith("_"))


def load_clinic(cfg: Config, code: str | None = None) -> Clinic:
    code = code or str(cfg.external("embryo_lab_data").get("clinic") or "").strip()
    if not code:
        raise ConfigError(
            f"не указана клиника. Доступны: {', '.join(available()) or '(нет профилей)'}. "
            f"Передайте --clinic или задайте embryo_lab_data.clinic в machine.local.yaml."
        )
    directory = CLINICS_DIR / code
    if not directory.exists():
        raise ConfigError(
            f"нет профиля клиники '{code}' ({directory}). Доступны: "
            f"{', '.join(available()) or '(нет профилей)'}. Новый профиль — копия "
            f"config/clinics/_template/, см. docs/RUNBOOK.md."
        )
    profile_path, columns_path = directory / "clinic.yaml", directory / "columns.yaml"
    for path in (profile_path, columns_path):
        if not path.exists():
            raise ConfigError(f"клиника {code}: нет {path.name}")
    with profile_path.open(encoding="utf-8") as fh:
        profile = yaml.safe_load(fh) or {}
    with columns_path.open(encoding="utf-8") as fh:
        columns = yaml.safe_load(fh) or {}
    return Clinic(code=code, profile=profile, columns=columns)


# ---------------------------------------------------------------------------
# Приведение выгрузки к каноническому кадру
# ---------------------------------------------------------------------------

def to_canonical(clinic: Clinic, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Выгрузка -> канонический кадр. Возвращает (кадр, список отсутствующих колонок).

    Отсутствующая колонка не заменяется догадкой и не роняет прогон: она попадает в
    список, а метрики, которым она нужна, окажутся недоступны явно.
    """
    out = pd.DataFrame(index=frame.index)
    missing: list[str] = []
    numeric = clinic.numeric
    decimal_comma = bool(clinic.columns.get("decimal_comma", False))

    for canonical, source in (clinic.columns.get("canonical") or {}).items():
        if source not in frame.columns:
            missing.append(f"{canonical} ({source})")
            out[canonical] = pd.Series(pd.NA, index=frame.index, dtype="object")
            continue
        series = frame[source].astype(str).str.strip().replace("", None)
        if canonical in numeric:
            if decimal_comma:
                series = series.str.replace(",", ".", regex=False)
            out[canonical] = pd.to_numeric(series, errors="coerce")
        else:
            out[canonical] = series

    # Доменные подстановки профиля. Применяются ЗДЕСЬ, на границе приведения к
    # каноническому кадру, чтобы весь конвейер — KPI, приор, вход модели — видел
    # одно и то же значение, а не каждый своё.
    if clinic.attempt_default is not None and "attempt_no" in out.columns:
        out["attempt_no"] = pd.to_numeric(out["attempt_no"], errors="coerce").fillna(
            clinic.attempt_default)

    # Сырое представление даты пункции нужно как часть ключа цикла: ключ не должен
    # меняться от того, как его сегодня распарсили.
    date_opu_source = clinic.source_column("date_opu")
    out["date_opu_raw"] = (frame[date_opu_source].astype(str).str.strip()
                           if date_opu_source in frame.columns else "")

    for canonical in clinic.date_columns:
        source = clinic.source_column(canonical)
        if source not in frame.columns:
            out[canonical] = pd.NaT
            continue
        parsed = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        raw = frame[source].astype(str).str.strip()
        for fmt in clinic.date_formats:
            attempt = pd.to_datetime(raw, format=fmt, errors="coerce")
            parsed = parsed.fillna(attempt)
        out[canonical] = parsed

    return out, missing


def cycle_key(clinic: Clinic, row: pd.Series) -> str:
    """Составной ключ цикла из профиля клиники."""
    parts = []
    for field in clinic.columns.get("cycle_key") or ["patient_id", "date_opu_raw"]:
        value = row.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
            return ""
        parts.append(str(value).strip().upper())
    return "#".join(parts) if all(parts) else ""
