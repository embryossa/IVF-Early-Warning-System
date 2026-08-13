"""Проверка: помогает ли обучение на подвыборке, похожей на нашу лабораторию.

    python experiment_matched_subsample.py --dev-path <dev.xlsx> --opu-path <наши.xlsx>
                                           --actuals <наши_факты.csv>

Модель долей, обученная на всей когорте 16k, на наших циклах не воспроизвелась
(corr доли 0.03 / -0.11 / 0.15 против 0.40 / 0.28 / 0.56 на отложенной выборке той же
когорты). Причина — популяции разные: у dev медиана числа бластоцист 1 и BlR 0.48,
у нас ~5 и 0.59. Гипотеза: обучение на подвыборке «нашего типа» вернёт дискриминацию.

**Отбор ведётся ТОЛЬКО по входным признакам** — возраст, фолликулы, ОКК. Отбирать по
уровню стадий (BlR, FR) нельзя: это отбор по исходу, он гарантирует хорошую видимость
подгонки и не значит ничего. Здесь это не осторожность, а условие осмысленности
эксперимента.

Сравниваются три режима обучения, оценка у всех одна — НАШИ циклы:
  full     — вся когорта 16k (текущее состояние)
  matched  — подвыборка по входным признакам
  narrow   — более узкая подвыборка (проверка «чем ближе, тем лучше»)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabpfn3_ivf_kpi_regression import (  # noqa: E402
    QUANTILE_NAMES, TASKS, build_rate, fit_predict, prepare_training)

# Признаки отбора и их значения у нашей лаборатории (опорное окно).
OUR_PROFILE = {"Число ОКК": 13.4, "Количество фолликулов": 18.6, "Возраст": 33.0}


def matched_mask(dev: pd.DataFrame, tolerance: float) -> pd.Series:
    """Строки dev, похожие на нашу лабораторию ПО ВХОДНЫМ признакам.

    Похожесть — попадание в коридор вокруг наших медиан, ширина коридора задаётся
    множителем. Никаких стадий-исходов в критерии нет.
    """
    mask = pd.Series(True, index=dev.index)
    for column, centre in OUR_PROFILE.items():
        values = pd.to_numeric(dev[column], errors="coerce")
        low, high = centre / tolerance, centre * tolerance
        mask &= values.between(low, high)
    return mask


def evaluate_on_ours(dev: pd.DataFrame, ours: pd.DataFrame, actuals: pd.DataFrame,
                     label: str) -> list[dict]:
    rows = []
    for task in TASKS:
        try:
            x_train, y_train = prepare_training(dev, task)
        except SystemExit:
            continue
        if len(x_train) < 200:
            rows.append({"режим": label, "задача": task.name, "n_обуч": len(x_train),
                         "corr": None, "note": "мало строк для обучения"})
            continue
        x_predict = ours[list(task.features)].apply(pd.to_numeric, errors="coerce")
        usable = x_predict[list(task.required or task.features)].notna().all(axis=1)
        if not usable.any():
            continue
        predicted = fit_predict(x_train, y_train, x_predict.loc[usable])
        p50 = predicted[:, QUANTILE_NAMES.index("p50")]

        actual = actuals.loc[usable, task.name].to_numpy(dtype=float)
        keep = ~np.isnan(actual)
        if keep.sum() < 10:
            continue
        correlation = float(np.corrcoef(actual[keep], p50[keep])[0, 1])
        inside = float(((actual[keep] >= predicted[keep, 0])
                        & (actual[keep] <= predicted[keep, -1])).mean())
        rows.append({
            "режим": label, "задача": task.name, "n_обуч": len(x_train),
            "n_оценки": int(keep.sum()), "corr": round(correlation, 2),
            "смещение": round(float(np.mean(p50[keep] - actual[keep])), 3),
            "покрытие": round(inside, 2),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-path", type=Path, required=True)
    parser.add_argument("--opu-path", type=Path, required=True)
    parser.add_argument("--actuals", type=Path, required=True)
    args = parser.parse_args()

    dev = pd.read_excel(args.dev_path)
    ours = pd.read_excel(args.opu_path)
    actuals = pd.read_csv(args.actuals, encoding="utf-8-sig")

    print(f"обучающая когорта: {len(dev)}, наших циклов: {len(ours)}")
    results: list[dict] = []
    for label, tolerance in (("full", None), ("matched", 1.8), ("narrow", 1.4)):
        subset = dev if tolerance is None else dev.loc[matched_mask(dev, tolerance)]
        share = len(subset) / len(dev) * 100
        print(f"\n[{label}] строк: {len(subset)} ({share:.0f}% когорты)")
        if len(subset) < 200:
            print(f"[{label}] слишком мало для обучения — режим пропущен")
            continue
        for column in OUR_PROFILE:
            values = pd.to_numeric(subset[column], errors="coerce")
            print(f"    {column:24s} медиана {values.median():6.1f} "
                  f"(наша {OUR_PROFILE[column]})")
        results += evaluate_on_ours(subset, ours, actuals, label)

    print("\n" + "=" * 74)
    print("ОЦЕНКА НА НАШИХ ЦИКЛАХ (corr доли — решающее число)")
    print(pd.DataFrame(results).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
