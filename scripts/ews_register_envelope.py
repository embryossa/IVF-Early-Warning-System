"""Регистрация прогнозного конверта: прогон TabPFN-3 на новых строках.

    python scripts/ews_register_envelope.py --clinic <code> [--retrospective] [--dry-run]

Конверт замораживается в момент регистрации и больше не пересчитывается. Регистрация
двухэтапная, потому что модель умеет предсказывать ровно то, на что у неё есть знание:

    T0  (день пункции)          -> mii, pn2, blasts_early
    T+1 (проверка оплодотворения) -> blasts_after_fert

`good` и `frozen` не регистрируются нигде: они обусловлены числом бластоцист, то есть
самой величиной, которую система мониторит. Прогноз, знающий исход, прогнозом не является.

`--retrospective` восстанавливает конверты для уже прошедших циклов. Такие строки
помечаются `registration_mode=retrospective` и годятся для калибровки и измерения
разнородности, но НЕ являются доказательством, что система «поймала бы» событие: их
модель обучалась в том числе на этих циклах.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import cycles, envelope, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import ConfigError, embryo_lab_data_on_path, load  # noqa: E402

# Чекпойнт цикла -> чекпойнт регистрации. Цикл, дошедший до закрытия, на
# проспективную регистрацию не идёт: его момент прошёл.
CHECKPOINT_ORDER = ["opu", "fertilisation", "closure"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", help="код профиля из config/clinics/")
    parser.add_argument("--retrospective", action="store_true",
                        help="восстановить конверты для уже прошедших циклов")
    parser.add_argument("--dry-run", action="store_true",
                        help="показать, что было бы зарегистрировано")
    args = parser.parse_args()

    try:
        cfg = load(args.clinic)
        clinic = load_clinic(cfg, args.clinic)
        embryo_lab_data_on_path(cfg)
    except ConfigError as exc:
        print(f"[!] {exc}")
        return 2

    print(f"клиника: {clinic.name} ({clinic.code})  окружение: {cfg.environment}")
    mode = "retrospective" if args.retrospective else "prospective"
    print(f"режим регистрации: {mode}")

    client = teable_client.connect(cfg)
    table_id = str(cfg.external("teable").get("opu_table_id") or "").strip()
    if not table_id:
        print("[!] teable.opu_table_id пуст — выполните scripts/ews_bootstrap_opu.py")
        return 2
    frame = cycles.load_from_teable(cfg, clinic, client, table_id)

    path = envelope.envelope_path(cfg, clinic)
    existing = envelope.load_envelopes(path)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    total = 0

    for checkpoint in ("opu", "fertilisation"):
        data = frame.data
        if not args.retrospective:
            # Проспективно регистрируем только те циклы, что СЕЙЧАС стоят ровно на
            # этом чекпойнте. Дошедший до закрытия свой момент упустил.
            data = data.loc[data["checkpoint"] == checkpoint]
        else:
            # Ретроспективно — все, кто этот чекпойнт уже прошёл.
            reached = CHECKPOINT_ORDER.index(checkpoint)
            data = data.loc[data["checkpoint"].map(
                lambda c: c in CHECKPOINT_ORDER and CHECKPOINT_ORDER.index(c) >= reached)]

        todo = envelope.pending(data, existing, checkpoint)
        stages = ", ".join(envelope.STAGES_AT[checkpoint])
        if todo.empty:
            print(f"\n{checkpoint}: нечего регистрировать ({stages})")
            continue
        print(f"\n{checkpoint}: {len(todo)} циклов, стадии {stages}")

        model_input = envelope.build_model_input(clinic, todo, checkpoint)
        known = ", ".join(envelope.KNOWLEDGE_AT[checkpoint])
        print(f"  вход модели: {len(model_input)} строк, известные признаки: {known}")
        print(f"  остальные колонки обнулены явно — гарантия в данных, "
              f"а не в предположении о чужом коде")

        if args.dry_run:
            print("  [dry] прогон модели пропущен")
            continue

        result, status = envelope.run_regression(cfg, clinic, model_input, checkpoint)
        if status != "ok":
            print(f"  [warn] модель не отработала: {status} — конверт не "
                  f"зарегистрирован, прогон не прерывается (мягкий шаг)")
            continue

        stage_frame = envelope.extract_stages(result, checkpoint)
        if stage_frame.empty:
            print("  [warn] в выходе модели нет нужных стадий")
            continue
        written = envelope.append(cfg, clinic, stage_frame, checkpoint, mode,
                                  model_version="tabpfn3_regression_CPU_16K",
                                  run_id=run_id)
        existing = envelope.load_envelopes(path)
        total += written
        print(f"  зарегистрировано и заморожено: {written}")

    print(f"\nвсего записано строк конверта: {total}")
    if total:
        print(f"файл: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
