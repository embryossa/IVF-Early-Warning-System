"""Литературный приор: третий контур, независимый от нас самих.

Механистический каскад Digital Twin (L1) считает ожидаемую долю перехода на каждой
стадии из начальных параметров пациентки — возраста, AMH, АФК. Формулы откалиброваны
по опубликованным большим данным, а не по нашей выгрузке. Это делает приор
принципиально иным типом опоры, чем два уже имеющихся контура:

| контур | опора | что отвечает | двигается ли вместе с нами |
|---|---|---|---|
| относительный | опорное окно своей лаборатории | «изменились ли мы относительно себя» | **да** |
| абсолютный | пороги Венского консенсуса | «где мы относительно стандарта» | нет, но когортный и статичный |
| **литературный** | формулы L1 на признаках пациентки | «где мы относительно ожидаемого ДЛЯ ЭТОЙ пациентки» | **нет** |

Третья строка — то, чего не хватало. Собственная базовая линия самореферентна: если
лаборатория медленно сползает, а окно переобъявляется, карты не увидят ничего.
Венский порог не сползает, но он один на всех пациенток. Литературный приор не
двигается вместе с нами И учитывает пациентку.

Второе свойство каскада — он достраивает недостающее. Каждая стадия принимает
известное значение как вырожденное распределение, а неизвестное генерирует. Поэтому
цепочку можно продолжить с любого места, где обрываются наши данные.

Источники формул (перенесены из `src/ivf_digital_twin.py` дословно, вместе со
ссылками; при расхождении первоисточником считается тот файл):

* зрелость — Herasight Table A1, упрощённый столбец 3;
* оплодотворение — Herasight столбец 5, ICSI;
* бластуляция — Romanski et al. 2022 (3362 пациентки), Sainte-Rose et al. 2021
  (4952 зиготы).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Индикатор стимулированного цикла в формулах Herasight. У нас все циклы
# стимулированные, поэтому он всегда 1 — вынесен в константу, чтобы это было видно,
# а не спрятано в числе.
STIMULATED = 1.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class PriorRate:
    metric: str
    stage: str
    source: str
    note: str


PRIOR_RATES = {
    "MIIR": PriorRate(
        "MIIR", "maturity", "Herasight Table A1 (col 3)",
        "зависит от возраста и AMH; популяционное среднее ≈ 90.7%"),
    "FR": PriorRate(
        "FR", "fertilisation", "Herasight (col 5), ICSI",
        "зависит только от возраста; популяционное среднее ≈ 72.3%"),
    "BlR": PriorRate(
        "BlR", "blastulation", "Romanski 2022; Sainte-Rose 2021",
        "плоская до 40 лет (0.70), затем спад — ниже 40 индивидуальной "
        "дискриминации не даёт вовсе"),
}


def maturity_rate(age, amh) -> pd.Series:
    """Ожидаемая доля зрелых ооцитов MII/ОКК."""
    age = pd.to_numeric(age, errors="coerce")
    amh = pd.to_numeric(amh, errors="coerce")
    return _sigmoid(2.4665 + 0.005 * age - 0.782 * STIMULATED + 0.24 * amh - 0.069)


def fertilisation_rate(age) -> pd.Series:
    """Ожидаемая доля оплодотворения 2PN/MII."""
    age = pd.to_numeric(age, errors="coerce")
    return _sigmoid(1.1678 + 0.004 * age - 0.303 * STIMULATED - 0.051)


def blastulation_rate(age) -> pd.Series:
    """Ожидаемая доля бластуляции Bl/2PN."""
    age = pd.to_numeric(age, errors="coerce")
    return pd.Series(np.clip(0.70 - 0.012 * np.maximum(0.0, age - 40.0), 0.30, 0.75),
                     index=age.index if hasattr(age, "index") else None)


def expected_rates(frame: pd.DataFrame,
                   amh_constant: float | None = None) -> pd.DataFrame:
    """Ожидаемые доли по трём контурам для каждого цикла кадра.

    ``amh_constant`` подставляется ТОЛЬКО в пропуски: реальные значения AMH, если
    они есть в выгрузке, всегда имеют приоритет. При сплошной подстановке формула
    зрелости вырождается в фиксированный уровень — размах по возрасту 25-45 лет
    всего 0.008, — и приор MIIR перестаёт различать пациенток. Как опора уровня он
    остаётся годен; проверка corr не пустит его в риск-коррекцию сама.
    """
    out = pd.DataFrame(index=frame.index)
    amh = pd.to_numeric(frame.get("amh"), errors="coerce")
    if amh_constant is not None:
        amh = amh.fillna(float(amh_constant))
    out["MIIR"] = maturity_rate(frame.get("age"), amh)
    out["FR"] = fertilisation_rate(frame.get("age"))
    out["BlR"] = blastulation_rate(frame.get("age"))
    return out


# Метрика -> (числитель, знаменатель) в каноническом кадре.
PARTS = {"MIIR": ("mii", "occ"), "FR": ("two_pn", "mii"), "BlR": ("bl_total", "two_pn")}


def gap_table(frame: pd.DataFrame,
              amh_constant: float | None = None) -> pd.DataFrame:
    """Разрыв «наша когортная доля минус литературное ожидание», по метрикам.

    Когортная доля — Σ числителей / Σ знаменателей, как везде в проекте, а не
    среднее пациентских долей. Ожидание усредняется по тем же циклам, что вошли в
    когортную долю, иначе сравниваются разные популяции.
    """
    expected = expected_rates(frame, amh_constant=amh_constant)
    rows = []
    for metric, (numerator_col, denominator_col) in PARTS.items():
        numerator = pd.to_numeric(frame.get(numerator_col), errors="coerce")
        denominator = pd.to_numeric(frame.get(denominator_col), errors="coerce")
        prior = expected[metric]
        observable = (numerator.notna() & denominator.notna() & (denominator > 0)
                      & (numerator <= denominator))
        usable = observable & prior.notna()
        if not usable.any():
            # Метрика не выпадает молча: невозможность посчитать приор — это
            # результат, и он обязан быть виден. Здесь это почти всегда значит,
            # что нет признака, на котором стоит формула (у зрелости — AMH).
            spec = PRIOR_RATES[metric]
            missing_inputs = "AMH" if metric == "MIIR" else "возраст"
            rows.append({
                "metric": metric, "n_cycles": int(observable.sum()),
                "observed": (round(float(numerator[observable].sum()
                                         / denominator[observable].sum()), 3)
                             if observable.any() else None),
                "literature": None, "gap_pp": None, "corr_rate": None,
                "source": spec.source,
                "note": f"приор не посчитан: нет {missing_inputs} — {spec.note}",
            })
            continue
        observed = float(numerator[usable].sum() / denominator[usable].sum())
        expectation = float(prior[usable].mean())
        per_cycle = numerator[usable] / denominator[usable]
        correlation = (float(np.corrcoef(per_cycle, prior[usable])[0, 1])
                       if per_cycle.nunique() > 1 and prior[usable].nunique() > 1
                       else float("nan"))
        spec = PRIOR_RATES[metric]
        rows.append({
            "metric": metric, "n_cycles": int(usable.sum()),
            "observed": round(observed, 3),
            "literature": round(expectation, 3),
            "gap_pp": round((observed - expectation) * 100, 1),
            "corr_rate": round(correlation, 2),
            "source": spec.source, "note": spec.note,
        })
    return pd.DataFrame(rows)


def usable_for_risk_adjustment(gaps: pd.DataFrame,
                               min_correlation: float = 0.3) -> dict[str, bool]:
    """Годится ли приор как риск-коррекция ПО КАЖДОЙ метрике.

    Разрыв уровня — не препятствие: систематическое смещение снимается сдвигом.
    Препятствие — отсутствие индивидуальной дискриминации: если приор не
    различает пациенток, подставлять его вместо базовой доли бессмысленно.
    Проверка отдельная от разрыва именно поэтому.
    """
    if gaps.empty:
        return {}
    return {row["metric"]: bool(row["corr_rate"] is not None
                                and not pd.isna(row["corr_rate"])
                                and row["corr_rate"] >= min_correlation)
            for _, row in gaps.iterrows()}
