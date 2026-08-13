"""Регистрация и заморозка прогнозного конверта.

Конверт — это то, чего лаборатория вправе ожидать от ЭТОЙ пациентки, зафиксированное
до того, как результат стал известен. Всё остальное в системе сравнивается с ним.

Два неподвижных правила:

1. **Заморозка.** Строка конверта пишется один раз и никогда не переписывается.
   Конверт, пересчитанный после того, как факт стал известен, — не прогноз, и
   сравнение с ним не является ранним предупреждением.
2. **Только знание своего момента.** На вход модели подаются ТОЛЬКО колонки, которые
   на этот чекпойнт реально заполнены. Остальные обнуляются явно, а не «наверняка
   модель их не использует»: гарантия должна быть в данных, а не в предположении о
   чужом коде.

Отсюда — двухэтапная регистрация, продиктованная тем, что модель умеет предсказывать
из имеющегося знания:

| стадия | нужно модели | регистрируется |
|---|---|---|
| `mii` | возраст, ОКК | T0 |
| `pn2` | возраст, MII | T0 |
| `blasts_early` | возраст, MII | T0 |
| `blasts_after_fert` | возраст, 2PN | T+1 |
| `good`, `frozen` | возраст, **Bl** | нигде: Bl известен только на закрытии |

`good`/`frozen` как конверт бесполезны: они обусловлены ровно той величиной, которую
мы и мониторим. Поэтому TGBDR риск-коррекции не получает и живёт в абсолютном контуре.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .clinic import Clinic
from .config import Config, ConfigError

# Что известно на каждом чекпойнте. Всё, чего здесь нет, на вход модели не попадает.
KNOWLEDGE_AT: dict[str, tuple[str, ...]] = {
    "opu": ("age", "attempt_no", "follicles", "occ", "mii", "amh", "bmi"),
    "fertilisation": ("age", "attempt_no", "follicles", "occ", "mii", "amh", "bmi",
                      "two_pn"),
}

# Какие стадии регистрируются на каком чекпойнте.
STAGES_AT: dict[str, tuple[str, ...]] = {
    "opu": ("mii", "pn2", "blasts_early"),
    "fertilisation": ("blasts_after_fert",),
}

# Канонические имена -> имена колонок, которые понимает внешний скрипт (его
# COLUMN_ALIASES). Пишем канонические алиасы, а не русские: они однозначны.
MODEL_INPUT_COLUMNS: dict[str, str] = {
    "age": "Age", "attempt_no": "# Attempt", "follicles": "N folicules OPU",
    "occ": "OCC", "mii": "MII", "two_pn": "2pN", "amh": "AMH", "bmi": "BMI",
}

QUANTILES = ("p10", "p25", "p50", "p75", "p90")

# --- Модель ДОЛЕЙ (KPI) ----------------------------------------------------
# Вторая модель конверта: та же архитектура, но целью обучения является ДОЛЯ, а не
# счёт. Появилась после измерения, показавшего, что модель счётов не предсказывает
# долю (corr 0.13/-0.08/0.16), а риск-коррекции нужна именно доля.
# Имена колонок — точные имена обучающей когорты, не алиасы: скрипт долей читает их
# напрямую.
RATE_MODEL_COLUMNS: dict[str, str] = {
    "age": "Возраст",
    "attempt_no": "№ попытки",
    "follicles": "Количество фолликулов",
    "occ": "Число ОКК",
    "mii": "Число инсеминированных",
    "two_pn": "2 pN",
}
RATE_STAGES_AT: dict[str, tuple[str, ...]] = {
    "opu": ("miir", "fr"),
    "fertilisation": ("blr",),
}
RATE_PREFIX = "TPFN3R_"

ENVELOPE_KEY = "cycle_uid"


def envelope_path(cfg: Config, clinic: Clinic) -> Path:
    return clinic.data_dir(cfg) / "ews_envelopes.csv"


def load_envelopes(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def registered_stages(envelopes: pd.DataFrame) -> dict[str, set[str]]:
    """cycle_uid -> множество уже зарегистрированных стадий."""
    out: dict[str, set[str]] = {}
    if envelopes.empty:
        return out
    for _, row in envelopes.iterrows():
        stages = out.setdefault(str(row[ENVELOPE_KEY]), set())
        for column in envelopes.columns:
            if column.startswith("TPFN3_") and column.endswith("_p50"):
                if pd.notna(row[column]):
                    stages.add(column[len("TPFN3_"):-len("_p50")])
    return out


def pending(frame: pd.DataFrame, envelopes: pd.DataFrame,
            checkpoint: str) -> pd.DataFrame:
    """Циклы, которым на этом чекпойнте конверт ещё не зарегистрирован.

    Цикл, уже прошедший чекпойнт, но пропущенный (система не работала в тот день),
    НЕ добирается задним числом на этом пути: для него конверт был бы
    ретроспективным. Такие идут отдельной командой с явной пометкой.
    """
    have = registered_stages(envelopes)
    wanted = set(STAGES_AT[checkpoint])
    # Регистрация чекпойнта атомарна: его стадии считаются одним прогоном модели и
    # пишутся вместе. Цикл, у которого есть ХОТЬ ОДНА стадия этого чекпойнта, уже
    # зарегистрирован — дописывать к нему нечего, и повторный прогон только упёрся
    # бы в защиту от перезаписи.
    mask = frame["cycle_uid"].map(lambda uid: not (wanted & have.get(uid, set())))
    return frame.loc[mask].copy()


def build_rate_model_input(frame: pd.DataFrame, checkpoint: str) -> pd.DataFrame:
    """Вход модели ДОЛЕЙ: только знание своего чекпойнта, как и у модели счётов."""
    known = set(KNOWLEDGE_AT[checkpoint])
    out = pd.DataFrame({"cycle_uid": frame["cycle_uid"].to_numpy()})
    for canonical, column in RATE_MODEL_COLUMNS.items():
        if canonical in known and canonical in frame.columns:
            # .to_numpy() обязателен: у кадра-источника свой индекс (подмножество
            # исходного), а у out — свежий RangeIndex. Присваивание Series молча
            # выровняло бы их по индексу и потеряло часть значений, не сказав ни
            # слова. На 91 цикле это стоило 10 строк из 77.
            out[column] = pd.to_numeric(frame[canonical], errors="coerce").to_numpy()
        else:
            out[column] = np.nan
    return out


def build_model_input(clinic: Clinic, frame: pd.DataFrame,
                      checkpoint: str) -> pd.DataFrame:
    """Вход модели: только те колонки, что известны на этот чекпойнт.

    Постчекпойнтные величины не «наверняка не используются моделью» — их физически
    нет во входном файле. Это единственная гарантия, которая не зависит от чужого
    кода.
    """
    known = set(KNOWLEDGE_AT[checkpoint])
    out = pd.DataFrame({"cycle_uid": frame["cycle_uid"].to_numpy()})
    for canonical, column in MODEL_INPUT_COLUMNS.items():
        if canonical in known and canonical in frame.columns:
            out[column] = pd.to_numeric(frame[canonical], errors="coerce").to_numpy()
        else:
            out[column] = np.nan
    if "date_opu" in frame.columns:
        out["Date OPU"] = pd.to_datetime(frame["date_opu"], errors="coerce").to_numpy()
    return out


def run_regression(cfg: Config, clinic: Clinic, model_input: pd.DataFrame,
                   checkpoint: str) -> tuple[pd.DataFrame, str]:
    """Прогнать внешнюю TabPFN-3 regression. Возвращает (кадр, статус).

    Сбой мягкий: конверт не зарегистрирован — цикл идёт дальше без риск-коррекции,
    прогон не падает. Но факт записывается, а не проглатывается.
    """
    import os
    import subprocess

    from .config import ROOT  # noqa: PLC0415

    section = cfg.machine.get("tabpfn_regression") or {}
    if not section.get("enabled", False):
        return pd.DataFrame(), "disabled"

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    python = resolve(str(section.get("python") or ""))
    project_dir = resolve(str(section.get("project_dir") or ""))
    script = project_dir / str(section.get("script") or "")
    private = clinic.data_dir(cfg) / "local_private"
    dev_cohort = private / "dev_cohort.xlsx"

    missing = [name for name, path in (("окружение модели", python),
                                       ("скрипт модели", script),
                                       ("обучающая когорта", dev_cohort))
               if not path.exists()]
    if missing:
        # Мягкий отказ с ИМЕНАМИ: «не настроено» без указания, чего именно нет, —
        # это полчаса поиска на чужой машине.
        return pd.DataFrame(), f"not_ready: нет {', '.join(missing)}"

    private.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    input_path = private / f"envelope_input_{checkpoint}_{stamp}.xlsx"
    output_path = private / f"envelope_output_{checkpoint}_{stamp}.csv"
    model_input.to_excel(input_path, index=False)

    # Каталог весов передаётся модели явно: без этого TabPFN полез бы качать 233 МБ
    # с HuggingFace, что на лабораторной машине без сети выглядит как зависание.
    env = dict(os.environ)
    cache_dir = section.get("model_cache_dir")
    if cache_dir:
        env["TABPFN_MODEL_CACHE_DIR"] = str(resolve(str(cache_dir)))

    command = [str(python), str(script.name),
               "--dev-path", str(dev_cohort),
               "--opu-path", str(input_path),
               "--output-csv", str(output_path)]
    started = datetime.now().timestamp()
    try:
        completed = subprocess.run(command, cwd=str(project_dir), env=env,
                                   capture_output=True, text=True, timeout=7200)
    except Exception as exc:  # noqa: BLE001 — мягкий шаг
        return pd.DataFrame(), f"failed: {exc}"
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        return pd.DataFrame(), f"exit {completed.returncode}: {' | '.join(tail)}"
    if not output_path.exists() or output_path.stat().st_mtime < started - 2:
        return pd.DataFrame(), "no_fresh_output"
    return pd.read_csv(output_path, encoding="utf-8-sig"), "ok"


def extract_stages(frame: pd.DataFrame, checkpoint: str) -> pd.DataFrame:
    """Оставить cycle_uid + квантили ТОЛЬКО тех стадий, что положены этому чекпойнту.

    Внешний скрипт считает все задачи, какие может. Стадии, обусловленные ещё не
    известными величинами, отбрасываются здесь: попадание их в конверт означало бы
    прогноз, знающий будущее.
    """
    if frame.empty or "cycle_uid" not in frame.columns:
        return pd.DataFrame()
    wanted = STAGES_AT[checkpoint]
    keep = ["cycle_uid"]
    for stage in wanted:
        keep += [f"TPFN3_{stage}_{q}" for q in QUANTILES
                 if f"TPFN3_{stage}_{q}" in frame.columns]
    return frame[[c for c in keep if c in frame.columns]].copy()


def append(cfg: Config, clinic: Clinic, rows: pd.DataFrame, checkpoint: str,
           mode: str, model_version: str, run_id: str) -> int:
    """Дописать замороженные конверты. Повторная регистрация стадии запрещена.

    ``mode``: `prospective` — зарегистрировано в свой момент, до того как результат
    стал известен; `retrospective` — восстановлено задним числом по историческим
    данным. Ретроспективный конверт годится для калибровки и измерения разнородности,
    но НЕ является доказательством, что система «поймала бы» событие. Различие
    хранится в самой строке, а не в памяти того, кто запускал.
    """
    if rows.empty:
        return 0
    path = envelope_path(cfg, clinic)
    existing = load_envelopes(path)
    have = registered_stages(existing)

    fresh = rows.copy()
    fresh["registered_at"] = datetime.now().isoformat(timespec="seconds")
    fresh["checkpoint"] = checkpoint
    fresh["registration_mode"] = mode
    fresh["model_version"] = model_version
    fresh["run_id"] = run_id
    fresh["environment"] = cfg.environment

    # Блокирует ЛЮБОЕ пересечение, а не только полное совпадение набора: одна
    # переписанная стадия — уже потерянная заморозка, и по выходным числам это
    # незаметно.
    incoming = {column[len("TPFN3_"):-len("_p50")] for column in fresh.columns
                if column.startswith("TPFN3_") and column.endswith("_p50")}
    already = fresh["cycle_uid"].map(lambda uid: bool(incoming & have.get(uid, set())))
    if already.any():
        collisions = sorted(fresh.loc[already, "cycle_uid"].astype(str))[:5]
        raise ConfigError(
            f"попытка перезаписать замороженный конверт: {int(already.sum())} циклов "
            f"на чекпойнте {checkpoint} уже имеют стадии {sorted(incoming)} "
            f"(например {', '.join(collisions)}). Конверт пишется один раз — "
            f"это неподвижное правило."
        )

    combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)
    return len(fresh)


def as_lookup(envelopes: pd.DataFrame) -> dict[str, dict]:
    """cycle_uid -> плоский словарь квантилей для слоя оценки.

    Строки разных чекпойнтов одного цикла сливаются: конверт цикла — это объединение
    того, что было зарегистрировано на T0 и на T+1, каждое в свой момент.
    """
    lookup: dict[str, dict] = {}
    if envelopes.empty:
        return lookup
    for _, row in envelopes.iterrows():
        entry = lookup.setdefault(str(row[ENVELOPE_KEY]), {})
        for column in envelopes.columns:
            if not column.startswith("TPFN3_") or pd.isna(row[column]):
                continue
            stage_quantile = column[len("TPFN3_"):]
            entry[stage_quantile] = float(row[column])
            entry[f"{stage_quantile}__mode"] = row.get("registration_mode")
    return lookup
