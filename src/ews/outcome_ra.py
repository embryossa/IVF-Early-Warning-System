"""Медленный контур с риск-коррекцией: исход переноса против ожидания Digital Twin.

`outcome.py` сравнивает СЫРЫЕ частоты беременности между периодами. Сырая частота
двигается вместе с составом пациенток: месяц старших пациенток с малым числом
бластоцист даёт «просадку», в которой лаборатория не виновата. Здесь каждому
переносу сопоставлено ожидание `p_i` — итоговая вероятность ансамбля Digital Twin
(L7 BEFE, колонка `DIGITAL TWIN` мастер-таблицы), и наблюдаемое сравнивается с
ожидаемым для ЭТИХ пациенток:

* `O/E = Σ y_i / Σ p_i` по месяцам и по группам с пределами funnel plot
  (Spiegelhalter 2005, `settings.charts.funnel`);
* VLAD — накопленная разница `Σ (y_i − p_i)`: «беременностей больше/меньше
  ожидаемого» нарастающим итогом (Lovegrove et al. 1997);
* risk-adjusted CUSUM Бернулли (Steiner et al. 2000) на снижение шансов
  беременности, с порогом, подобранным симуляцией под целевую частоту ложных
  сигналов в год.

**Калибровка на центр.** DT 7.1 не калиброван на клинику: в analytics-режиме его
средняя вероятность может стоять ниже или выше реальной частоты центра (на RBMO
сдвиг по центрам был главным источником ошибки, дискриминация внутри центра —
AUC около 0.55). Без поправки карта на ухудшение молчала бы годами в клинике,
работающей выше прогноза. Поэтому применяется сдвиг только интерсепта,
`logit p* = logit p + δ`, и `δ` **замораживается** на первых `min_known` исходах
(файл `state/outcome_calibration.json`) — по тому же правилу, что и конверт: ожидание,
переоценённое по тем фактам, с которыми его сравнивают, ничего не предупреждает.
Пока исходов меньше, ожидание — DT как есть, и это печатается.

**Чего контур не делает.** Не поднимает тревог и не участвует в эскалации
(неподвижное решение владельца, см. `outcome.py`). Группы по лотам, среде и
эмбриологам — кандидат-объяснения, не причины и не оценка лота или специалиста:
исход зависит от переноса, эндометрия и того, что DT не видит.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm

from . import charts

DEFAULTS = {
    "expected_column": "digital_twin",
    "min_known_calibration": 40,
    "min_known_group": 10,
    "min_known_period": 20,
    "cusum": {"odds_ratio": 0.5, "target_signals_per_year": 0.5, "replicates": 2000},
    "lot_items": [],
    "inner_limit": 0.95,
    "outer_limit": 0.998,
}

POSITION_LOW_OUTER = "ниже внешнего предела"
POSITION_LOW_INNER = "ниже внутреннего предела"
POSITION_INSIDE = "в пределах"
POSITION_HIGH_INNER = "выше внутреннего предела"
POSITION_HIGH_OUTER = "выше внешнего предела"
POSITION_FEW = "мало исходов"


def settings_of(settings: dict) -> dict:
    """Параметры контура: `settings.outcome_ra` поверх значений по умолчанию.

    Пределы funnel берутся из `charts.funnel`, если там заданы, — чтобы у карт
    процесса и у исхода были одни и те же 95% / 99.8%.
    """
    section = dict(DEFAULTS)
    section["cusum"] = dict(DEFAULTS["cusum"])
    own = (settings or {}).get("outcome_ra") or {}
    for key, value in own.items():
        if key == "cusum" and isinstance(value, dict):
            section["cusum"].update(value)
        else:
            section[key] = value
    funnel = ((settings or {}).get("charts") or {}).get("funnel") or {}
    section["inner_limit"] = float(funnel.get("inner_limit", section["inner_limit"]))
    section["outer_limit"] = float(funnel.get("outer_limit", section["outer_limit"]))
    return section


# --- кадр переносов -------------------------------------------------------------

def outcome_frame(frame: pd.DataFrame, expected_column: str = "digital_twin") -> pd.DataFrame:
    """Переносы с известным исходом и ожиданием модели, в хронологическом порядке.

    Знаменатель — только перенос с известным исходом (0/1). Ожидание вне (0, 1)
    отбрасывается; значение больше 1 читается как проценты. Пропуск ожидания —
    не «средний риск»: такой перенос в контур не входит, и их число печатается.
    """
    columns = ["cycle_uid", "date_opu", "y", "p_model", "embryologist"]
    if frame is None or frame.empty or "transfer_outcome" not in frame.columns:
        return pd.DataFrame(columns=columns)
    outcome = pd.to_numeric(frame["transfer_outcome"], errors="coerce")
    known = outcome.isin([0, 1])
    p = pd.to_numeric(frame.get(expected_column), errors="coerce")
    p = p.where(p <= 1.0, p / 100.0)
    usable = known & p.gt(0) & p.lt(1) & frame["date_opu"].notna()
    out = pd.DataFrame({
        "cycle_uid": frame.loc[usable, "cycle_uid"].astype(str),
        "date_opu": pd.to_datetime(frame.loc[usable, "date_opu"]),
        "y": outcome[usable].astype(float),
        "p_model": p[usable].astype(float),
        "embryologist": (frame.loc[usable, "embryologist"]
                         if "embryologist" in frame.columns else pd.NA),
    })
    out.attrs["known_without_expected"] = int((known & ~usable).sum())
    return out.sort_values(["date_opu", "cycle_uid"], kind="stable").reset_index(drop=True)


# --- калибровка на центр ----------------------------------------------------------

@dataclass
class Calibration:
    """Сдвиг интерсепта δ: замороженный или ещё не набранный."""
    delta: float = 0.0
    frozen: bool = False
    n: int = 0
    observed: float | None = None
    expected_raw: float | None = None
    se: float | None = None
    frozen_at: str | None = None
    last_date_opu: str | None = None
    expected_column: str = "digital_twin"
    note: str = ""

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return expit(logit(p) + self.delta)


def fit_delta(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """δ такой, что Σ expit(logit p + δ) = Σ y, и его стандартная ошибка.

    Это калибровка «в целом» (calibration-in-the-large) — единственный параметр,
    который здесь оценивается: наклон DT не трогается, переобучения модели нет.
    SE — из информации Фишера, `1 / sqrt(Σ p*(1−p*))`.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    target = float(y.sum())
    if target <= 0 or target >= len(y):
        raise ValueError("все исходы одинаковы — сдвиг не определён")
    base = logit(p)
    delta = brentq(lambda d: float(expit(base + d).sum()) - target, -8.0, 8.0)
    fitted = expit(base + delta)
    se = 1.0 / np.sqrt(float((fitted * (1 - fitted)).sum()))
    return float(delta), float(se)


def calibration_path(data_dir: Path) -> Path:
    return data_dir / "state" / "outcome_calibration.json"


def load_calibration(path: Path, expected_column: str) -> Calibration | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("expected_column", "digital_twin") != expected_column:
        # Сдвиг, замороженный для другой модели, к этой не применим.
        return None
    return Calibration(delta=float(data["delta"]), frozen=True, n=int(data["n"]),
                       observed=data.get("observed"), expected_raw=data.get("expected_raw"),
                       se=data.get("se"), frozen_at=data.get("frozen_at"),
                       last_date_opu=data.get("last_date_opu"),
                       expected_column=expected_column,
                       note=f"сдвиг заморожен {data.get('frozen_at')} по {data['n']} исходам")


def resolve_calibration(transfers: pd.DataFrame, existing: Calibration | None,
                        min_known: int, expected_column: str) -> Calibration:
    """Замороженный сдвиг, если есть; иначе — заморозить, как только набралось.

    Калибровочная выборка — первые ``min_known`` исходов по дате пункции. Всё, что
    позже, сравнивается с ожиданием, к построению которого отношения не имело.
    """
    if existing is not None:
        return existing
    n = len(transfers)
    if n < min_known:
        return Calibration(expected_column=expected_column, n=n, note=(
            f"сдвиг на центр не заморожен: исходов {n} из нужных {min_known}; "
            f"ожидание — DT как есть, без калибровки на клинику"))
    head = transfers.iloc[:min_known]
    try:
        delta, se = fit_delta(head["y"].to_numpy(), head["p_model"].to_numpy())
    except ValueError as exc:
        return Calibration(expected_column=expected_column, n=n,
                           note=f"сдвиг на центр не определён: {exc}")
    return Calibration(
        delta=delta, frozen=True, n=min_known, se=se,
        observed=float(head["y"].mean()), expected_raw=float(head["p_model"].mean()),
        frozen_at=date.today().isoformat(),
        last_date_opu=pd.Timestamp(head["date_opu"].iloc[-1]).date().isoformat(),
        expected_column=expected_column,
        note=(f"сдвиг заморожен сейчас по первым {min_known} исходам "
              f"(факт {head['y'].mean():.0%} против DT {head['p_model'].mean():.0%})"))


def save_calibration(path: Path, calibration: Calibration, environment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "delta": calibration.delta, "se": calibration.se, "n": calibration.n,
        "observed": calibration.observed, "expected_raw": calibration.expected_raw,
        "frozen_at": calibration.frozen_at, "last_date_opu": calibration.last_date_opu,
        "expected_column": calibration.expected_column, "environment": environment,
        "note": "сдвиг интерсепта DT на клинику; заморожен и не пересчитывается — "
                "удаление файла означает новую калибровку и должно быть записано",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# --- O/E и funnel ----------------------------------------------------------------

def oe_row(y: np.ndarray, p: np.ndarray, inner: float, outer: float,
           min_known: int) -> dict:
    """O/E группы и положение относительно пределов funnel.

    Пределы — под нулевой гипотезой «ожидание верно»: дисперсия числа
    беременностей `V = Σ p(1−p)` (пуассон-биномиальная), пределы `1 ± z·√V / E`.
    Это пределы для O/E, а не доверительный интервал O/E: вопрос «совместима ли
    группа с ожиданием», а не «каков её истинный уровень».
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = int(len(y))
    if n == 0:
        return {"n": 0, "observed": 0, "expected": 0.0, "oe": None,
                "inner_low": None, "inner_high": None, "outer_low": None,
                "outer_high": None, "position": POSITION_FEW}
    observed, expected = float(y.sum()), float(p.sum())
    sd = float(np.sqrt((p * (1 - p)).sum()))
    z_in = float(norm.ppf(0.5 + inner / 2))
    z_out = float(norm.ppf(0.5 + outer / 2))
    oe = observed / expected if expected > 0 else None
    limits = {
        "inner_low": max(0.0, 1 - z_in * sd / expected), "inner_high": 1 + z_in * sd / expected,
        "outer_low": max(0.0, 1 - z_out * sd / expected), "outer_high": 1 + z_out * sd / expected,
    }
    if n < min_known or oe is None:
        position = POSITION_FEW
    elif oe < limits["outer_low"]:
        position = POSITION_LOW_OUTER
    elif oe < limits["inner_low"]:
        position = POSITION_LOW_INNER
    elif oe > limits["outer_high"]:
        position = POSITION_HIGH_OUTER
    elif oe > limits["inner_high"]:
        position = POSITION_HIGH_INNER
    else:
        position = POSITION_INSIDE
    return {"n": n, "observed": int(observed), "expected": round(expected, 2),
            "oe": None if oe is None else round(oe, 3),
            **{k: round(v, 3) for k, v in limits.items()}, "position": position}


def strata(transfers: pd.DataFrame, dimension: str, column: str, cfg: dict) -> list[dict]:
    rows = []
    for group, part in transfers.groupby(column, dropna=False, sort=True):
        label = "не указано" if pd.isna(group) else str(group)
        row = oe_row(part["y"], part["p_expected"], cfg["inner_limit"],
                     cfg["outer_limit"], cfg["min_known_group"])
        rows.append({"dimension": dimension, "group": label, **row})
    return rows


def pseudonymize_staff(series: pd.Series, prefix: str = "Эмбриолог") -> pd.Series:
    """Имена -> «Эмбриолог A/B/…» по убыванию числа переносов.

    Панель и пакеты описывают процесс, а не человека: имён здесь не бывает.
    """
    clean = series.astype("string").str.strip()
    clean = clean.where(clean.notna() & (clean != ""), pd.NA)
    order = clean.value_counts().index.tolist()
    mapping = {name: f"{prefix} {chr(ord('A') + i) if i < 26 else i}"
               for i, name in enumerate(order)}
    return clean.map(mapping).astype("string")


# --- VLAD и CUSUM ----------------------------------------------------------------

def solve_cusum_h(p: np.ndarray, odds_ratio: float, target_per_year: float,
                  per_year: float, replicates: int, rng: np.random.Generator,
                  bounds: tuple[float, float] = (0.2, 20.0), max_iter: int = 18) -> float:
    """Порог h под целевую частоту ложных сигналов в год.

    Год моделируется как ``per_year`` переносов с ожиданиями, разыгранными из
    наблюдаемых `p*` (состав пациенток клиники), а исходы — из этих же `p*`: то
    есть процесс в контроле. Семантика события — как у `CusumRunner` (alert):
    пересечение и перезапуск.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    horizon = max(int(round(per_year)), 1)
    draws = rng.choice(p, size=(replicates, horizon), replace=True)
    outcomes = (rng.random((replicates, horizon)) < draws).astype(float)
    weights = charts.cusum_weight(outcomes, draws, odds_ratio)

    def rate(h: float) -> float:
        signals = np.zeros(replicates)
        current = np.zeros(replicates)
        for step in range(horizon):
            current = np.maximum(0.0, current + weights[:, step])
            crossed = current > h
            signals += crossed
            current[crossed] = 0.0
        return float(signals.mean())

    low, high = bounds
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        if rate(mid) > target_per_year:
            low = mid
        else:
            high = mid
    return float(high)


def trajectories(transfers: pd.DataFrame, odds_ratio: float, h: float | None) -> pd.DataFrame:
    """VLAD и CUSUM по переносам в хронологическом порядке (сигналы с перезапуском)."""
    out = transfers.copy()
    out["vlad"] = (out["y"] - out["p_expected"]).cumsum()
    weights = charts.cusum_weight(out["y"].to_numpy(), out["p_expected"].to_numpy(),
                                  odds_ratio)
    values, signals, current = [], [], 0.0
    for weight in weights:
        current = max(0.0, current + float(weight))
        signal = h is not None and current > h
        signals.append(bool(signal))
        values.append(current)
        if signal:
            current = 0.0
    out["cusum"] = values
    out["cusum_signal"] = signals
    return out


# --- лоты и среда -----------------------------------------------------------------

def lot_assignments(transfers: pd.DataFrame, timeline: pd.DataFrame, window_days: int,
                    items: list[str] | None = None) -> pd.DataFrame:
    """Какой лот каждой позиции был в окне культивирования каждого переноса.

    Одна позиция — один период лота в окне: лот приписывается (`stable`). Больше
    одного периода — «смена лота в окне»: такой перенос не приписывается ни
    старому, ни новому лоту (правило владельца: не угадывать).
    """
    from .vendor import lots  # noqa: PLC0415

    wanted = {str(i).strip().lower() for i in (items or []) if str(i).strip()}
    rows = []
    for _, row in transfers.iterrows():
        _, events = lots.analyze_window_events(timeline, row["date_opu"],
                                               days=window_days, cycle_uid=row["cycle_uid"])
        if events.empty:
            continue
        for item, group in events.groupby("item_name", sort=False):
            name = str(item).strip()
            if wanted and name.lower() not in wanted:
                continue
            if (group["event_type"] == "lot_change").any():
                lot = LOT_CHANGED
            else:
                lot = normalize_lot(group["lot_after"].iloc[0])
            rows.append({"cycle_uid": row["cycle_uid"], "item": name, "lot": lot})
    out = pd.DataFrame(rows, columns=["cycle_uid", "item", "lot"])
    return merge_scans(out)


LOT_CHANGED = "смена лота в окне"


GS = "\x1d"
# Прикладные идентификаторы GS1 фиксированной длины (AI -> длина значения) и
# переменной длины (значение до разделителя GS или конца). Достаточно тех, что
# встречаются на расходниках ЭКО; незнакомый AI — разбор прекращается.
GS1_FIXED = {"00": 18, "01": 14, "02": 14, "11": 6, "12": 6, "13": 6, "15": 6,
             "16": 6, "17": 6, "20": 2}
GS1_VARIABLE = ("240", "241", "250", "251", "10", "21", "22", "30", "37",
                "90", "91", "92", "93", "94", "95", "96", "97", "98", "99")


def gs1_lot(raw) -> str | None:
    """Номер лота — значение AI (10) из скана GS1 DataMatrix; None — не GS1.

    Полный скан несёт ещё GTIN (01), срок годности (17), серийный номер (21):
    две коробки одного лота дают разные строки, и без разбора один лот
    превратился бы в два. Понимает и сырой скан с GS, и запись «(10)…».
    """
    import re  # noqa: PLC0415

    text = str(raw).strip()
    bracket = re.search(r"\(10\)([^(\x1d]+)", text)
    if bracket:
        return bracket.group(1).strip()
    if GS not in text and not re.match(r"^(01|02|00)\d{14}", text):
        return None
    for segment in text.split(GS):
        rest = segment.strip()
        while rest:
            fixed = next((ai for ai in GS1_FIXED if rest.startswith(ai)), None)
            if fixed:
                rest = rest[len(fixed) + GS1_FIXED[fixed]:]
                continue
            variable = next((ai for ai in GS1_VARIABLE if rest.startswith(ai)), None)
            if variable is None:
                break
            if variable == "10":
                return rest[2:]
            rest = ""
    return None


def normalize_lot(value) -> str:
    """Код лота как его сравнивает формула «Lot Changed» в Airtable.

    Сначала из GS1-скана берётся только AI (10); затем верхний регистр, только
    A-Z и 0-9: разделители, скобки и дефисы в разных сканах одного лота бывают
    разными.
    """
    import re  # noqa: PLC0415

    lot = gs1_lot(value) or value
    return re.sub(r"[^A-Z0-9]", "", str(lot).upper()) or "без кода"


def merge_scans(assigned: pd.DataFrame) -> pd.DataFrame:
    """Короткий скан лота и полный DataMatrix того же лота — один лот.

    Правило то же, что у Airtable: один нормализованный код входит в другой.
    Приводится к самому длинному коду позиции, в который входит.
    """
    if assigned.empty:
        return assigned
    out = assigned.copy()
    for item, part in out.groupby("item"):
        codes = sorted({c for c in part["lot"] if c != LOT_CHANGED}, key=len, reverse=True)
        mapping = {}
        for code in codes:
            mapping[code] = next((longer for longer in codes
                                  if len(longer) > len(code) and code in longer), code)
        mask = (out["item"] == item) & (out["lot"] != LOT_CHANGED)
        out.loc[mask, "lot"] = out.loc[mask, "lot"].map(mapping)
    return out


def lot_label(lot: str, limit: int = 16) -> str:
    """Не-GS1 код бывает длинным; на панели хватает хвоста. Служебные метки — как есть."""
    text = str(lot)
    if text == LOT_CHANGED or len(text) <= limit:
        return text
    return f"…{text[-limit:]}"


def environment_groups(transfers: pd.DataFrame, context_csv: Path,
                       baseline: dict) -> pd.Series:
    """Группа нагрузки средой инкубатора по артефакту недельного прохода.

    Граница — верхний дециль критических минут по когорте (`confounder_baseline`),
    как в подтверждении тревог: абсолютного порога у среды нет (критический эпизод
    есть у 100% окон). Окно без разобранных логов — отдельная группа, не «норма».
    """
    labels = pd.Series("логи не разобраны", index=transfers.index, dtype="string")
    if not context_csv.exists():
        return labels
    context = pd.read_csv(context_csv, encoding="utf-8-sig")
    context = context.drop_duplicates("cycle_uid").set_index("cycle_uid")
    limit = (baseline or {}).get("critical_minutes")
    for index, uid in transfers["cycle_uid"].items():
        if uid not in context.index or context.at[uid, "miri_status"] != "enriched":
            continue
        minutes = float(context.at[uid, "critical_minutes"] or 0.0)
        if limit is None:
            labels[index] = "разобрано, границы нет"
        elif minutes > float(limit):
            labels[index] = "выше верхнего дециля"
        else:
            labels[index] = "в пределах дециля"
    return labels


# --- итог -------------------------------------------------------------------------

@dataclass
class OutcomeRA:
    transfers: pd.DataFrame
    calibration: Calibration
    strata: pd.DataFrame
    h: float | None
    odds_ratio: float
    per_year: float
    verdict: bool | None
    note: str
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        overall = self.strata.loc[self.strata["dimension"] == "всего"]
        row = overall.iloc[0].to_dict() if not overall.empty else {}
        signals = int(self.transfers["cusum_signal"].sum()) if not self.transfers.empty else 0
        return {
            "transfers_with_outcome": int(len(self.transfers)),
            "known_without_expected": int(self.transfers.attrs.get("known_without_expected", 0)),
            "expected_column": self.calibration.expected_column,
            "calibration": {"frozen": self.calibration.frozen,
                            "delta": round(self.calibration.delta, 4),
                            "se": None if self.calibration.se is None else round(self.calibration.se, 4),
                            "n": self.calibration.n, "note": self.calibration.note},
            "overall": {k: row.get(k) for k in ("n", "observed", "expected", "oe", "position")},
            "vlad_last": (round(float(self.transfers["vlad"].iloc[-1]), 2)
                          if not self.transfers.empty else None),
            "cusum": {"odds_ratio": self.odds_ratio, "h": None if self.h is None else round(self.h, 3),
                      "transfers_per_year": round(self.per_year, 1), "signals": signals,
                      "last": (round(float(self.transfers["cusum"].iloc[-1]), 3)
                               if not self.transfers.empty else None)},
            "verdict": self.verdict, "note": self.note, "notes": self.notes,
        }


def period_verdict(transfers: pd.DataFrame, split: pd.Timestamp, cfg: dict) -> tuple[bool | None, str]:
    """Подтверждает ли исход тренд процесса — с поправкой на состав пациенток.

    Опорное окно сравнивается с ожиданием DT (после сдвига на центр): если O/E окна
    ниже внутреннего предела, изменение дошло до результата И не объясняется тем,
    какие пациентки пришли. Сырые частоты так не умеют.
    """
    recent = transfers.loc[transfers["date_opu"] >= split]
    if len(recent) < cfg["min_known_period"]:
        return None, (f"исходов в опорном окне {len(recent)} из нужных "
                      f"{cfg['min_known_period']} — риск-скорректированный контур ещё не созрел")
    row = oe_row(recent["y"], recent["p_expected"], cfg["inner_limit"], cfg["outer_limit"],
                 cfg["min_known_period"])
    text = (f"O/E опорного окна {row['oe']:.2f} при пределах {row['inner_low']:.2f}–"
            f"{row['inner_high']:.2f} (n={row['n']}, факт {row['observed']}, "
            f"ожидание {row['expected']:.1f})")
    if row["position"] in (POSITION_LOW_INNER, POSITION_LOW_OUTER):
        return True, f"{text}: беременностей меньше ожидаемого для этих пациенток"
    if row["position"] in (POSITION_HIGH_INNER, POSITION_HIGH_OUTER):
        return False, f"{text}: исход выше ожидания — просадка процесса до результата не дошла"
    return False, f"{text}: исход совместим с ожиданием — тренд процесса результатом не подтверждается"


def build(frame: pd.DataFrame, settings: dict, data_dir: Path, split: pd.Timestamp,
          *, timeline: pd.DataFrame | None = None, window_days: int = 6,
          context_csv: Path | None = None, env_baseline: dict | None = None,
          freeze: bool = True, environment: str = "draft",
          rng: np.random.Generator | None = None) -> OutcomeRA:
    """Собрать контур целиком. ``freeze=False`` (dry-run) не пишет сдвиг на диск."""
    cfg = settings_of(settings)
    rng = rng or np.random.default_rng(20261017)
    column = str(cfg["expected_column"])
    closed = frame
    if "checkpoint" in frame.columns:
        closed = frame.loc[frame["checkpoint"] == "closure"]
    transfers = outcome_frame(closed, column)
    notes: list[str] = []
    skipped = transfers.attrs.get("known_without_expected", 0)
    if skipped:
        notes.append(f"переносов с исходом, но без ожидания {column}: {skipped} — в контур не вошли")

    path = calibration_path(data_dir)
    existing = load_calibration(path, column)
    calibration = resolve_calibration(transfers, existing, int(cfg["min_known_calibration"]), column)
    if calibration.frozen and existing is None and freeze:
        save_calibration(path, calibration, environment)

    transfers["p_expected"] = calibration.apply(transfers["p_model"].to_numpy()) if len(transfers) else []
    transfers["embryologist"] = pseudonymize_staff(transfers["embryologist"]) if len(transfers) else []
    transfers["month"] = transfers["date_opu"].dt.to_period("M").astype(str) if len(transfers) else []
    transfers["environment_group"] = (environment_groups(transfers, context_csv, env_baseline or {})
                                      if context_csv is not None and len(transfers)
                                      else "логи не разобраны")

    # Объём переносов с исходом в год — горизонт симуляции порога.
    span_days = ((transfers["date_opu"].max() - transfers["date_opu"].min()).days
                 if len(transfers) > 1 else 0)
    per_year = len(transfers) * 365.25 / span_days if span_days >= 60 else float(len(transfers))
    odds_ratio = float(cfg["cusum"]["odds_ratio"])
    h = None
    if len(transfers) >= int(cfg["min_known_group"]):
        h = solve_cusum_h(transfers["p_expected"].to_numpy(), odds_ratio,
                          float(cfg["cusum"]["target_signals_per_year"]), per_year,
                          int(cfg["cusum"]["replicates"]), rng)
    transfers = trajectories(transfers, odds_ratio, h)

    rows = [{"dimension": "всего", "group": "все переносы",
             **oe_row(transfers["y"], transfers["p_expected"], cfg["inner_limit"],
                      cfg["outer_limit"], cfg["min_known_group"])}]
    if calibration.frozen:
        after = transfers.loc[transfers["date_opu"] > pd.Timestamp(calibration.last_date_opu)]
        rows.append({"dimension": "всего", "group": "после калибровки",
                     **oe_row(after["y"], after["p_expected"], cfg["inner_limit"],
                              cfg["outer_limit"], cfg["min_known_group"])})
    rows += strata(transfers, "месяц", "month", cfg)
    rows += strata(transfers, "эмбриолог", "embryologist", cfg)
    rows += strata(transfers, "среда инкубатора", "environment_group", cfg)

    if timeline is not None and not timeline.empty and len(transfers):
        assigned = lot_assignments(transfers, timeline, window_days, cfg.get("lot_items"))
        if assigned.empty:
            notes.append("лоты: ни одно окно переноса не пересеклось с историей лотов")
        merged = assigned.merge(transfers[["cycle_uid", "y", "p_expected"]], on="cycle_uid")
        single = 0
        for item, part in merged.groupby("item", sort=True):
            # Один лот на все переносы — сравнивать не с чем: группа повторила бы
            # итог «всего» и только раздувала бы число сравнений.
            if part["lot"].nunique() < 2:
                single += 1
                continue
            for lot, group in part.groupby("lot", sort=True):
                rows.append({"dimension": f"лот: {item}", "group": lot_label(lot),
                             **oe_row(group["y"], group["p_expected"], cfg["inner_limit"],
                                      cfg["outer_limit"], cfg["min_known_group"])})
        if single:
            notes.append(f"лоты: {single} позиций с одним лотом на все переносы не показаны — "
                         f"сравнивать не с чем")
    elif timeline is None:
        notes.append("лоты: история из Airtable не загружена — группы по лотам не строились")

    verdict, note = period_verdict(transfers, split, cfg)
    table = pd.DataFrame(rows)
    flagged = table.loc[table["position"].isin([POSITION_LOW_OUTER, POSITION_LOW_INNER])
                        & (table["dimension"] != "всего")]
    if len(flagged):
        notes.append(f"групп ниже предела: {len(flagged)} из {len(table)} — при таком числе "
                     f"сравнений часть выходов ожидаема случайно; это поводы посмотреть, не выводы")
    return OutcomeRA(transfers=transfers, calibration=calibration, strata=table, h=h,
                     odds_ratio=odds_ratio, per_year=per_year, verdict=verdict, note=note,
                     notes=notes)


def fetch_timeline(cfg) -> tuple[pd.DataFrame | None, str]:
    """История лотов из Airtable — мягкий шаг: без токена или сети контур строится без неё."""
    airtable = cfg.machine.get("airtable") or {}
    if not cfg.gate("airtable"):
        return None, "Airtable выключен в settings.local.yaml (enabled.airtable)"
    try:
        from .vendor import lots  # noqa: PLC0415

        items, changes = lots.fetch_consumable_lots({"consumable_lots": {"airtable": {
            "token": cfg.secret("airtable"),
            "base_id": str(airtable.get("base_id") or ""),
            "items_table_id": str(airtable.get("items_table_id") or ""),
            "changes_table_id": str(airtable.get("changes_table_id") or ""),
        }}})
        timeline = lots.build_lot_timeline(items, changes)
    except Exception as exc:  # noqa: BLE001 — мягкий шаг
        return None, f"история лотов недоступна: {type(exc).__name__}"
    return timeline, f"история лотов: {timeline['item_record_id'].nunique()} позиций, {len(timeline)} периодов"


def write_artifacts(result: OutcomeRA, data_dir: Path, environment: str,
                    packet_dir: Path | None, clinic_code: str) -> list[Path]:
    """CSV по переносам и по группам + пакет агрегатов для агентов (без cycle_uid)."""
    written = []
    cycles_out = result.transfers[["cycle_uid", "date_opu", "y", "p_model", "p_expected",
                                   "vlad", "cusum", "cusum_signal", "embryologist",
                                   "environment_group"]].copy()
    cycles_out["environment"] = environment
    strata_out = result.strata.copy()
    strata_out["environment"] = environment
    for table, name in ((cycles_out, "ews_outcome_ra_transfers.csv"),
                        (strata_out, "ews_outcome_ra_strata.csv")):
        path = data_dir / name
        tmp = path.with_suffix(".csv.tmp")
        table.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(path)
        written.append(path)
    if packet_dir is not None:
        packet_dir.mkdir(parents=True, exist_ok=True)
        path = packet_dir / f"outcome_ra_{clinic_code}.json"
        path.write_text(json.dumps({
            "environment": environment, "clinic": clinic_code, "summary": result.summary(),
            "strata": result.strata.to_dict("records"),
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        written.append(path)
    return written
