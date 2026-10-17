"""Чтение чужой выгрузки и приведение к каноническому кадру EWS.

Граница приватности проходит здесь: идентификаторы пациентки превращаются в
``cycle_uid`` (HMAC) и удаляются до любых расчётов. Имена эмбриологов и врачей по
умолчанию заменяются псевдонимами («Эмбриолог A») — панель показывает наблюдения о
процессе, а не оценку конкретного специалиста.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..vendor import privacy

ROOT = Path(__file__).resolve().parents[3]
ALIASES_FILE = ROOT / "config" / "demo" / "column_aliases.yaml"

PHI_CANONICAL = ("patient_id", "patient_name", "dob")
COUNT_COLUMNS = ("follicles", "occ", "mii", "inseminated", "two_pn", "three_pn", "one_pn",
                 "zero_pn", "degrade", "cleavage_d3", "bl_total", "bl_good", "frozen",
                 "transferred")
NUMERIC_COLUMNS = COUNT_COLUMNS + ("age", "attempt_no", "amh", "afc", "bmi", "fsh")


def normalize_name(value: object) -> str:
    """Нормализация имени колонки для ТОЧНОГО сравнения с синонимом.

    Убираются регистр, ё/е, знаки препинания и повторные пробелы. Больше ничего:
    это не нечёткий поиск, а защита от разницы «2 pN» / «2pN » / «2PN».
    """
    text = str(value).replace("ё", "е").replace("Ё", "Е").lower().strip()
    text = re.sub(r"[^\w+/]+", " ", text)
    return re.sub(r"\s+", "", text)


@dataclass
class ColumnReport:
    mapped: dict[str, str] = field(default_factory=dict)       # canonical -> исходное имя
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unmapped_source: list[str] = field(default_factory=list)
    overridden: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"mapped": self.mapped, "ambiguous": self.ambiguous, "missing": self.missing,
                "unmapped_source": self.unmapped_source, "overridden": self.overridden}


@dataclass
class DemoFrame:
    frame: pd.DataFrame            # канонический кадр БЕЗ PHI, одна строка = цикл
    columns: ColumnReport
    source_name: str               # только имя файла, без пути
    notes: list[str]
    exclusions: dict[str, int]
    operator_crosswalk: pd.DataFrame


def load_aliases(path: Path = ALIASES_FILE) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def read_table(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    """Excel или CSV. Все ячейки читаются как объекты: типы приводятся позже явно."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            return pd.read_excel(path, sheet_name=sheet or 0, engine="calamine")
        except (ImportError, ValueError):
            return pd.read_excel(path, sheet_name=sheet or 0)
    if suffix in (".csv", ".tsv", ".txt"):
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "cp1251"):
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=encoding,
                                   dtype=str, keep_default_na=False, na_values=[""])
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(f"не удалось прочитать CSV ни в UTF-8, ни в cp1251: {last_error}")
    raise ValueError(f"неподдерживаемый формат файла: {suffix} (нужен .xlsx/.xls/.csv)")


def resolve_columns(columns: list[str], aliases: dict,
                    override: dict[str, str] | None = None) -> ColumnReport:
    report = ColumnReport()
    by_norm: dict[str, list[str]] = {}
    for col in columns:
        by_norm.setdefault(normalize_name(col), []).append(col)

    override = {k: v for k, v in (override or {}).items() if v}
    for canonical, source in override.items():
        if source not in columns:
            raise ValueError(f"override: колонки '{source}' для '{canonical}' нет в файле")
        report.mapped[canonical] = source
        report.overridden.append(canonical)

    used = set(report.mapped.values())
    for canonical, synonyms in (aliases.get("canonical") or {}).items():
        if canonical in report.mapped:
            continue
        hits: list[str] = []
        for synonym in synonyms:
            for col in by_norm.get(normalize_name(synonym), []):
                if col not in hits and col not in used:
                    hits.append(col)
        if len(hits) == 1:
            report.mapped[canonical] = hits[0]
            used.add(hits[0])
        elif len(hits) > 1:
            report.ambiguous[canonical] = hits
        else:
            report.missing.append(canonical)
    report.unmapped_source = [c for c in columns if c not in set(report.mapped.values())]
    return report


def _to_number(series: pd.Series, *, leading: bool = False) -> pd.Series:
    """Число из ячейки; пусто -> NaN (НЕ ноль). ``leading`` — «34 года» -> 34."""
    text = series.astype("string").str.strip().str.replace(" ", "", regex=False)
    text = text.str.replace(",", ".", regex=False)
    if leading:
        text = text.str.extract(r"^\s*(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(text.astype(object), errors="coerce").astype(float)


def _to_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    serial = numeric.between(20000, 80000)          # серийные даты Excel
    if serial.any():
        out[serial] = pd.to_datetime(numeric[serial], unit="D", origin="1899-12-30")
    rest = ~serial & series.notna()
    if rest.any():
        text = series[rest].astype(str).str.strip()
        iso = text.str.match(r"^\d{4}-\d{2}-\d{2}")
        parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
        parsed[iso] = pd.to_datetime(text[iso].str[:10], errors="coerce", format="%Y-%m-%d")
        parsed[~iso] = pd.to_datetime(text[~iso], errors="coerce", dayfirst=True, format="mixed")
        out[rest] = parsed
    return out


def _token_set(values: list[str]) -> set[str]:
    return {normalize_name(v) for v in values}


def derive_method(frame: pd.DataFrame, aliases: dict) -> pd.Series:
    """Метод оплодотворения: ICSI / IVF / IVF+ICSI / исходное значение / NaN.

    Две формы выгрузки: одна колонка со значением («IVF/ICSI» = «ICSI») или флаги
    по методам (колонки IVF и ICSI со значением «Да»). Флаги обоих методов в одной
    строке — смешанный цикл, а не ошибка.
    """
    tokens = aliases.get("method_tokens", {})
    icsi_tok = _token_set(tokens.get("icsi", []))
    ivf_tok = _token_set(tokens.get("ivf", []))
    result = pd.Series(pd.NA, index=frame.index, dtype="string")

    if "fertilization_method" in frame.columns:
        raw = frame["fertilization_method"].astype("string").str.strip()
        for idx, value in raw.items():
            if pd.isna(value) or not value:
                continue
            parts = {normalize_name(p) for p in re.split(r"[/+,;&]| и | and ", value.lower()) if p.strip()}
            has_icsi = bool(parts & icsi_tok) or normalize_name(value) in icsi_tok
            has_ivf = bool(parts & ivf_tok) or normalize_name(value) in ivf_tok
            if has_icsi and has_ivf:
                result[idx] = "IVF+ICSI"
            elif has_icsi:
                result[idx] = "ICSI"
            elif has_ivf:
                result[idx] = "IVF"
            else:
                result[idx] = value                     # Cryo / NO / прочее — как есть
    if result.isna().all() and ("ivf_flag" in frame.columns or "icsi_flag" in frame.columns):
        truthy = _token_set(aliases.get("flag_true_tokens", []))

        def flag(col: str) -> pd.Series:
            if col not in frame.columns:
                return pd.Series(False, index=frame.index)
            return frame[col].map(lambda v: (not pd.isna(v)) and normalize_name(v) in truthy)

        ivf, icsi = flag("ivf_flag"), flag("icsi_flag")
        result[ivf & icsi] = "IVF+ICSI"
        result[icsi & ~ivf] = "ICSI"
        result[ivf & ~icsi] = "IVF"
    return result


def _match(series: pd.Series, tokens: list[str]) -> pd.Series:
    wanted = _token_set(tokens)
    return series.map(lambda v: (not pd.isna(v)) and normalize_name(v) in wanted).astype(bool)


def pseudonymize(series: pd.Series, prefix: str) -> tuple[pd.Series, pd.DataFrame]:
    """Имена специалистов -> «Эмбриолог A». Порядок букв — по числу циклов."""
    clean = series.astype("string").str.strip()
    clean = clean.where(clean.notna() & (clean != ""), pd.NA)
    order = clean.value_counts().index.tolist()
    letters = [chr(ord("A") + i) if i < 26 else f"Z{i}" for i in range(len(order))]
    mapping = {name: f"{prefix} {letter}" for name, letter in zip(order, letters)}
    crosswalk = pd.DataFrame({"pseudonym": list(mapping.values()), "source_name": list(mapping.keys())})
    return clean.map(mapping).astype("string"), crosswalk


def load_demo(path: Path, *, sheet: str | int | None = None,
              override: dict[str, str] | None = None,
              secret: bytes | None = None,
              show_operator_names: bool = False,
              aliases: dict | None = None) -> DemoFrame:
    aliases = aliases or load_aliases()
    raw = read_table(path, sheet)
    raw = raw.loc[:, [not str(c).startswith("Unnamed") or raw[c].notna().any() for c in raw.columns]]
    raw = raw.dropna(how="all")
    raw.columns = [str(c) for c in raw.columns]
    report = resolve_columns(list(raw.columns), aliases, override)

    frame = pd.DataFrame(index=raw.index)
    for canonical, source in report.mapped.items():
        frame[canonical] = raw[source]
    notes: list[str] = []

    # --- псевдонимизация на границе ingest, до любых расчётов ---------------
    date_raw = frame["date_opu"].astype("string") if "date_opu" in frame else None
    if "patient_id" in frame.columns:
        key = frame["patient_id"].astype("string").str.strip()
        if date_raw is not None:
            key = key + "|" + date_raw.fillna("")
        key = key.fillna("") + "|row" + pd.Series(raw.index.astype(str), index=raw.index)
    else:
        key = pd.Series([f"row{i}" for i in raw.index], index=raw.index)
        notes.append("В файле нет идентификатора пациентки — cycle_uid построен по номеру строки.")
    frame["_uid_key"] = f"{path.name}|" + key
    secret = secret or privacy.secrets.token_bytes(32)
    frame = privacy.add_cycle_uid(frame, "_uid_key", secret)
    frame = frame.drop(columns=["_uid_key", *[c for c in PHI_CANONICAL if c in frame.columns]])

    # --- типы ----------------------------------------------------------------
    for col in NUMERIC_COLUMNS:
        if col in frame.columns:
            frame[col] = _to_number(frame[col], leading=(col == "age"))
    if "date_opu" in frame.columns:
        frame["date_opu"] = _to_date(frame["date_opu"])
        bad = frame["date_opu"].isna().sum()
        if bad:
            notes.append(f"Дата пункции не распознана или пуста в {bad} строках — они не входят во "
                         "временные ряды и карты, но входят в сводные KPI.")
    else:
        notes.append("Нет колонки даты пункции — временной анализ и контрольные карты недоступны.")

    # Донорские ооциты: Age в таких строках — возраст донора (в выгрузках бывает
    # заглушкой-константой), а не пациентки; в case-mix и TGBDR по возрасту не идёт.
    if "oocyte_source" in frame.columns:
        donor = _match(frame["oocyte_source"], aliases.get("donor_oocyte_tokens", []))
        frame["donor_oocytes"] = donor
        if donor.any():
            if "age" in frame.columns:
                frame.loc[donor, "age"] = pd.NA
            notes.append(f"Донорские ооциты в {int(donor.sum())} строках — возраст в них относится к донору "
                         "и исключён из анализа по возрасту (case-mix, TGBDR по возрасту).")

    frame["method"] = derive_method(frame, aliases)
    crosswalks = []
    for col, prefix in (("embryologist", "Эмбриолог"), ("doctor", "Врач")):
        if col in frame.columns:
            if show_operator_names:
                frame[col] = frame[col].astype("string").str.strip()
            else:
                frame[col], cw = pseudonymize(frame[col], prefix)
                crosswalks.append(cw.assign(role=col))
    crosswalk = pd.concat(crosswalks, ignore_index=True) if crosswalks else pd.DataFrame()

    # --- популяция лаборатории (как kpi.competency_exclusion_mask) -----------
    comp = aliases.get("competency_exclusion", {})
    method_raw = frame["fertilization_method"] if "fertilization_method" in frame else frame["method"]
    treatment = frame["treatment_type"] if "treatment_type" in frame else pd.Series(pd.NA, index=frame.index)
    stale = _match(treatment, aliases.get("funnel_exclusion", {}).get("stale_funnel_tokens", []))
    no_retrieval = _match(method_raw, comp.get("no_retrieval_tokens", []))
    freeze_all = _match(method_raw, comp.get("freeze_all_tokens", []))
    if "termination_reason" in frame.columns:
        reason = frame["termination_reason"].astype("string").str.strip()
        open_cycle = ~(reason.notna() & (reason != ""))
    else:
        open_cycle = frame["bl_total"].isna() if "bl_total" in frame else pd.Series(False, index=frame.index)
    retrieval_only = _match(treatment, comp.get("embryologist_only_treatment_tokens", []))
    frame["retrieval_only"] = retrieval_only
    frame["surgical_sperm"] = (_match(frame["sperm_source"], comp.get("surgical_sperm_tokens", []))
                               if "sperm_source" in frame else False)

    # Строки итогов и физиологически невозможные счётчики. Это НЕ то же, что ClR>100%
    # или MII>OCC (те — наблюдения и остаются): сотни-тысячи ооцитов в одном цикле
    # бывают только в строке «Итого», дописанной под таблицей. Такая строка в сумме
    # Σ/Σ удваивает лабораторию, поэтому исключается и печатается явно.
    plaus = aliases.get("plausibility", {})
    summary_re = plaus.get("summary_row_pattern", r"^\s*(?:итого|всего|total|sum|сумма)\b")
    text_cells = raw.astype("string").apply(lambda c: c.str.contains(summary_re, case=False, regex=True, na=False))
    summary_row = text_cells.any(axis=1).reindex(frame.index, fill_value=False)
    max_count = float(plaus.get("max_count_per_cycle", 100))
    counts = [c for c in COUNT_COLUMNS if c in frame.columns]
    implausible = (frame[counts] > max_count).any(axis=1) if counts else pd.Series(False, index=frame.index)
    implausible = implausible & ~summary_row
    if summary_row.any() or implausible.any():
        notes.append(f"Исключено как строки итогов: {int(summary_row.sum())}; со счётчиком > {max_count:.0f} "
                     f"на цикл (физиологически невозможно, вероятно итог или ошибка ввода): "
                     f"{int(implausible.sum())}.")

    # Открытый цикл без Bl, но с «только пункция» — законное наблюдение врача.
    open_cycle = open_cycle & ~retrieval_only
    excluded = stale | no_retrieval | freeze_all | open_cycle | summary_row | implausible
    exclusions = {
        "rows_total": int(len(frame)),
        "summary_or_implausible": int((summary_row | implausible).sum()),
        "stale_funnel": int(stale.sum()),
        "no_retrieval": int(no_retrieval.sum()),
        "freeze_all": int(freeze_all.sum()),
        "open_cycle_no_bl": int((open_cycle & ~stale & ~no_retrieval & ~freeze_all).sum()),
        "retrieval_only_doctor_only": int(retrieval_only.sum()),
        "surgical_sperm_excluded_from_fr": int(pd.Series(frame["surgical_sperm"]).sum()),
        "lab_population": int((~excluded).sum()),
    }
    frame = frame.loc[~excluded].copy()
    if "date_opu" in frame.columns:
        frame = frame.sort_values("date_opu", kind="stable")
    frame = frame.reset_index(drop=True)
    return DemoFrame(frame=frame, columns=report, source_name=path.name, notes=notes,
                     exclusions=exclusions, operator_crosswalk=crosswalk)


def fr_by_method_denominator(frame: pd.DataFrame) -> pd.Series:
    """Знаменатель FR по Венскому консенсусу для каждого метода.

    ICSI — инъецированные MII (MII>0, иначе инсеминированные — как в kpi.compute_rates);
    IVF — инсеминированные ОКК (иначе ОКК); смешанный — все инсеминированные.
    """
    def col(name: str) -> pd.Series:
        return frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)

    mii, ins, occ = col("mii"), col("inseminated"), col("occ")
    icsi_den = mii.where(mii > 0, ins)
    ivf_den = ins.where(ins > 0, occ)
    mixed_den = ins.where(ins > 0, mii)
    method = frame.get("method", pd.Series(pd.NA, index=frame.index)).astype("string")
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    out[method == "ICSI"] = icsi_den[method == "ICSI"]
    out[method == "IVF"] = ivf_den[method == "IVF"]
    out[method == "IVF+ICSI"] = mixed_den[method == "IVF+ICSI"]
    return out
