"""TabPFN-3 regression по ДОЛЯМ (KPI), а не по абсолютным счётам.

    python tabpfn3_ivf_kpi_regression.py --dev-path <dev.xlsx> --opu-path <opu.xlsx>
                                         --output-csv <out.csv> [--evaluate]

Зачем отдельная модель. Прежняя (`tabpfn3_ivf_regression_report_CPU_16K.py`)
предсказывает СЧЁТ стадии. Измерено на опорном окне: её прогноз коррелирует с фактом
по счёту (0.86-0.97) и не коррелирует по доле (0.13 / -0.08 / 0.16). Высокая
корреляция счёта — артефакт знаменателя: и факт, и прогноз растут вместе с числом
ооцитов. Риск-коррекции карт нужна именно ДОЛЯ — какая часть у ЭТОЙ пациентки дойдёт
до следующей стадии. Здесь доля и является целью обучения.

Три задачи под три карты EWS, каждая — только на знании своего чекпойнта:

    miir  = MII / ОКК        признаки: возраст, попытка, фолликулы, ОКК        -> T0
    fr    = 2PN / MII        признаки: + MII                                   -> T0
    blr   = Bl / 2PN         признаки: + 2PN                                   -> T+1

Что НЕ делается и почему:

* `tgbdr` (Good Bl / 2PN) не обучается: её знаменатель известен, но полезный прогноз
  требует числа бластоцист, то есть величины, которую система и мониторит.
* Строки с малым знаменателем отбрасываются из ОБУЧЕНИЯ: доля по 1-2 эмбрионам —
  это шум, и модель, обученная на нём, учится шуму. В ПРЕДСКАЗАНИИ такие строки
  остаются: отказ предсказывать честнее, чем предсказание, обученное на мусоре.
* Квантили обрезаются в [0, 1]: доля вне этого диапазона не является долей.

`--evaluate` считает главное число — корреляцию предсказанной доли с фактической на
отложенной выборке. Если она не заметно выше нуля, модель не решает задачу, ради
которой её обучали, и включать риск-коррекцию нельзя.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# TabPFN по умолчанию запрещает CPU при >1000 строк как «слишком медленно». У нас
# обучение идёт в контексте на 16k строк и занимает минуты — это осознанный обмен,
# тот же, что в прежней модели (она ставит тот же флаг). Выставляется ДО импорта.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, max(1, (os.cpu_count() or 2) - 1))))

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_NAMES = ("p10", "p25", "p50", "p75", "p90")
RANDOM_STATE = 42
MIN_DENOMINATOR_FOR_TRAINING = 3
MAX_CONTEXT = 16000


@dataclass(frozen=True)
class RateTask:
    name: str
    numerator: str
    denominator: str
    features: tuple[str, ...]
    checkpoint: str
    label: str
    # Признаки, без которых предсказывать нечего. Остальные из `features` —
    # необязательные: TabPFN штатно работает с пропусками, и отбрасывать строку
    # из-за редко заполняемой колонки значит терять цикл на ровном месте. У этой
    # лаборатории «№ попытки» заполнен в 4 строках из 77 — при обязательности
    # всех признаков модель предсказала бы для четырёх циклов вместо семидесяти.
    required: tuple[str, ...] = ()


# Имена колонок обучающей когорты. MII в ней представлен «Числом инсеминированных» —
# так же, как в прежней модели (её COLUMN_ALIASES ставит их в один ключ).
AGE = "Возраст"
ATTEMPT = "№ попытки"
FOLLICLES = "Количество фолликулов"
OCC = "Число ОКК"
MII = "Число инсеминированных"
PN2 = "2 pN"
BL = "Число Bl"

TASKS: tuple[RateTask, ...] = (
    RateTask("miir", MII, OCC, (AGE, ATTEMPT, FOLLICLES, OCC), "opu",
             "Доля зрелых ооцитов (MII/ОКК)", required=(AGE, OCC)),
    RateTask("fr", PN2, MII, (AGE, ATTEMPT, FOLLICLES, OCC, MII), "opu",
             "Частота оплодотворения (2PN/MII)", required=(AGE, MII)),
    RateTask("blr", BL, PN2, (AGE, ATTEMPT, FOLLICLES, OCC, MII, PN2), "fertilisation",
             "Частота бластуляции (Bl/2PN)", required=(AGE, PN2)),
)


def log(message: str) -> None:
    print(message, flush=True)


def build_rate(frame: pd.DataFrame, task: RateTask) -> pd.Series:
    """Доля из числителя и знаменателя. Вне [0,1] — не доля, а брак наблюдения."""
    numerator = pd.to_numeric(frame.get(task.numerator), errors="coerce")
    denominator = pd.to_numeric(frame.get(task.denominator), errors="coerce")
    rate = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = numerator.notna() & denominator.notna() & (denominator > 0)
    rate.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return rate.where((rate >= 0) & (rate <= 1))


def prepare_training(dev: pd.DataFrame, task: RateTask) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in task.features if c not in dev.columns]
    if missing:
        raise SystemExit(f"[{task.name}] в обучающей когорте нет колонок: {missing}")
    rate = build_rate(dev, task)
    denominator = pd.to_numeric(dev[task.denominator], errors="coerce")
    keep = rate.notna() & (denominator >= MIN_DENOMINATOR_FOR_TRAINING)
    features = dev.loc[keep, list(task.features)].apply(pd.to_numeric, errors="coerce")
    # Обязательны только required; остальные пропуски TabPFN обрабатывает сам.
    keep_rows = features[list(task.required or task.features)].notna().all(axis=1)
    return features.loc[keep_rows], rate.loc[keep].loc[keep_rows]


def fit_predict(x_train: pd.DataFrame, y_train: pd.Series,
                x_predict: pd.DataFrame) -> np.ndarray:
    """Квантили доли для строк предсказания. Возвращает [n, 5], обрезанные в [0,1]."""
    from tabpfn import TabPFNRegressor

    if len(x_train) > MAX_CONTEXT:
        # TabPFN учится в контексте: больше MAX_CONTEXT строк он не берёт, и
        # случайная подвыборка честнее, чем молчаливое обрезание по порядку файла.
        sample = x_train.sample(MAX_CONTEXT, random_state=RANDOM_STATE)
        x_train, y_train = sample, y_train.loc[sample.index]

    model = TabPFNRegressor(device="cpu", n_estimators=1, random_state=RANDOM_STATE,
                            memory_saving_mode="auto", ignore_pretraining_limits=True)
    model.fit(x_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))
    output = model.predict(x_predict.to_numpy(dtype=float), output_type="quantiles",
                           quantiles=list(QUANTILES))
    matrix = np.asarray(output, dtype=float)
    if matrix.shape[0] == len(QUANTILES):
        matrix = matrix.T
    matrix = np.clip(matrix, 0.0, 1.0)
    # Численная страховка: квантили обязаны не убывать.
    return np.maximum.accumulate(matrix, axis=1)


def evaluate(dev: pd.DataFrame, task: RateTask, holdout: float = 0.25) -> dict:
    """Главная проверка: предсказывает ли модель ДОЛЮ на отложенной выборке."""
    features, rate = prepare_training(dev, task)
    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(len(features))
    cut = int(len(features) * (1 - holdout))
    train_idx, test_idx = order[:cut], order[cut:]

    predicted = fit_predict(features.iloc[train_idx], rate.iloc[train_idx],
                            features.iloc[test_idx])
    p50 = predicted[:, QUANTILE_NAMES.index("p50")]
    actual = rate.iloc[test_idx].to_numpy(dtype=float)

    inside = ((actual >= predicted[:, 0]) & (actual <= predicted[:, -1])).mean()
    correlation = float(np.corrcoef(actual, p50)[0, 1])
    baseline_mae = float(np.abs(actual - rate.iloc[train_idx].mean()).mean())
    return {
        "task": task.name, "n_train": len(train_idx), "n_test": len(test_idx),
        "corr_rate": correlation,
        "mae": float(np.abs(actual - p50).mean()),
        "mae_baseline": baseline_mae,
        "coverage_p10_p90": float(inside),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-path", type=Path, required=True)
    parser.add_argument("--opu-path", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--evaluate", action="store_true",
                        help="проверить на отложенной выборке и выйти")
    parser.add_argument("--checkpoint", choices=["opu", "fertilisation", "all"],
                        default="all", help="какие задачи считать")
    args = parser.parse_args()

    dev = pd.read_excel(args.dev_path)
    log(f"обучающая когорта: {len(dev)} строк, {dev.shape[1]} колонок")

    tasks = [t for t in TASKS
             if args.checkpoint in ("all", t.checkpoint)]

    if args.evaluate:
        log("\nПРОВЕРКА НА ОТЛОЖЕННОЙ ВЫБОРКЕ (25%)")
        log(f"{'задача':8s}{'n обуч':>8s}{'n тест':>8s}{'corr доля':>11s}"
            f"{'MAE':>8s}{'MAE базы':>10s}{'покрытие':>10s}")
        for task in tasks:
            started = time.time()
            result = evaluate(dev, task)
            log(f"{result['task']:8s}{result['n_train']:8d}{result['n_test']:8d}"
                f"{result['corr_rate']:11.2f}{result['mae']:8.3f}"
                f"{result['mae_baseline']:10.3f}{result['coverage_p10_p90']:10.2f}"
                f"   ({time.time() - started:.0f} с)")
        log("\ncorr доля — решающее число: если она не заметно выше нуля, модель")
        log("не решает задачу, ради которой обучалась, и риск-коррекцию включать нельзя.")
        return 0

    if not args.opu_path or not args.output_csv:
        raise SystemExit("нужны --opu-path и --output-csv (или --evaluate)")

    opu = pd.read_excel(args.opu_path)
    key = "cycle_uid" if "cycle_uid" in opu.columns else opu.columns[0]
    out = pd.DataFrame({key: opu[key]})

    for task in tasks:
        missing = [c for c in task.features if c not in opu.columns]
        if missing:
            log(f"[{task.name}] пропущена: в OPU нет {missing}")
            continue
        x_train, y_train = prepare_training(dev, task)
        x_predict = opu[list(task.features)].apply(pd.to_numeric, errors="coerce")
        usable = x_predict[list(task.required or task.features)].notna().all(axis=1)
        log(f"[{task.name}] обучение на {len(x_train)} строках, "
            f"предсказание для {int(usable.sum())} из {len(opu)}")
        if not usable.any():
            continue
        started = time.time()
        predicted = fit_predict(x_train, y_train, x_predict.loc[usable])
        for index, name in enumerate(QUANTILE_NAMES):
            column = f"TPFN3R_{task.name}_{name}"
            out[column] = np.nan
            out.loc[usable, column] = predicted[:, index]
        log(f"[{task.name}] готово за {time.time() - started:.0f} с")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    log(f"\nзаписано: {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
