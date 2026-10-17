"""Прогон EWS: чекпойнты, оценка на закрытии, тревоги, панель.

    python scripts/ews_run.py --clinic two_line [--replay] [--dry-run]

Без ключей — суточный прогон: обрабатываются только циклы, закрывшиеся с прошлого
раза (состояние карт хранится и переживает перезапуск).

`--replay` — прогон по всей истории с нуля: состояние карт сбрасывается и цикл за
циклом восстанавливается хронологически. Это ретроспектива «что система сказала бы»,
основной способ посмотреть на неё до боевого запуска. Порядок обязателен: карта
секвенциальна, перестановка циклов изменила бы траекторию.

Тревоги существуют только на закрытии цикла. T0 и T+1 регистрируются и в оценку не
идут — неподвижное решение владельца.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import (confounders, cycles, envelope, envelope_qc,  # noqa: E402
                 evaluate, outcome, outcome_ra, panel, prior, teable_client)
from ews.clinic import load_clinic  # noqa: E402
from ews.config import ConfigError, embryo_lab_data_on_path, load  # noqa: E402


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Атомарная запись: временный файл рядом, затем replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def checkpoints_frame(frame: pd.DataFrame, environment: str) -> pd.DataFrame:
    out = frame[["cycle_uid", "checkpoint", "date_opu", "closure_due"]].copy()
    out["recorded_at"] = datetime.now().isoformat(timespec="seconds")
    out["environment"] = environment
    return out


def alerts_frame(evaluations: list, environment: str, clinic_code: str) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(evaluations, start=1):
        if item.level == evaluate.LEVEL_OK:
            continue
        triggered = {m: c for m, c in item.charts.items() if c.get("crossed")}
        rows.append({
            "alert_id": f"{clinic_code}-{index:05d}",
            "cycle_uid": item.cycle_uid,
            "raised_at": (item.closure_date.date().isoformat()
                          if pd.notna(item.closure_date) else ""),
            "level": item.level,
            "chart": ", ".join(sorted(triggered)),
            "statistic": json.dumps({m: c["statistic"] for m, c in triggered.items()},
                                    ensure_ascii=False),
            "expected_source": ", ".join(sorted({c["expected_source"]
                                                 for c in triggered.values()})),
            "contour": item.contour or "",
            "corroboration": json.dumps(item.corroboration, ensure_ascii=False),
            "corroboration_rank": item.corroboration_rank,
            "miri_status": item.miri_status,
            "action": item.action,
            "verdict": item.verdict_summary,
            "notes": " | ".join(item.notes),
            "environment": environment,
        })
    return pd.DataFrame(rows)


def outcome_ra_step(cfg, clinic, data: pd.DataFrame, data_dir: Path,
                    freeze: bool) -> "outcome_ra.OutcomeRA | None":
    """Риск-скорректированный медленный контур. Мягкий шаг: сбой не валит прогон.

    Лоты тянутся из Airtable, только если клиника их ведёт; среда — из артефакта
    недельного прохода (`ews_window_context.csv`), логи заново не разбираются.
    """
    try:
        timeline, lots_note = (outcome_ra.fetch_timeline(cfg) if clinic.source_enabled("lots")
                               else (None, "лоты выключены в профиле клиники"))
        env_baseline, _ = confounders.load_baseline(cfg, clinic)
        result = outcome_ra.build(
            data, cfg.settings, data_dir, clinic.window_bounds()[0],
            timeline=timeline,
            window_days=int(cfg.settings["soft_steps"]["miri"]["cultivation_window_days"]),
            context_csv=data_dir / "ews_window_context.csv", env_baseline=env_baseline,
            freeze=freeze, environment=cfg.environment)
    except Exception as exc:  # noqa: BLE001 — мягкий шаг, как confounders
        print(f"\n[warn] риск-скорректированный контур не построен: {exc}")
        return None
    summary = result.summary()
    overall = summary["overall"]
    if not summary["transfers_with_outcome"]:
        print("\nисход против ожидания DT: переносов с исходом и ожиданием нет")
        return result
    print(f"\nисход против ожидания DT: переносов {summary['transfers_with_outcome']}, "
          f"факт {overall['observed']}, ожидание {overall['expected']}, "
          f"O/E {overall['oe']} ({overall['position']})")
    print(f"  {summary['calibration']['note']}")
    print(f"  {lots_note}")
    print(f"  VLAD {summary['vlad_last']:+}  RA-CUSUM {summary['cusum']['last']} "
          f"(порог {summary['cusum']['h']}, сигналов {summary['cusum']['signals']})")
    print(f"  {result.note}")
    for note in result.notes:
        print(f"  {note}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", help="код профиля из config/clinics/")
    parser.add_argument("--replay", action="store_true",
                        help="пересобрать состояние карт по всей истории")
    parser.add_argument("--dry-run", action="store_true",
                        help="посчитать и показать, ничего не записывая")
    args = parser.parse_args()

    try:
        cfg = load(args.clinic)
        clinic = load_clinic(cfg, args.clinic)
        embryo_lab_data_on_path(cfg)
    except ConfigError as exc:
        print(f"[!] {exc}")
        return 2

    data_dir = clinic.data_dir(cfg)
    baseline_path = data_dir / "baseline" / "incontrol_parameters.json"
    baselines = evaluate.baselines_from_manifest(baseline_path)
    if not baselines:
        print(f"[!] нет опорных параметров ({baseline_path}). "
              f"Сначала scripts/ews_baseline.py --clinic {clinic.code}")
        return 2

    thresholds = cfg.settings["thresholds"]
    block = cfg.alert_block_reason()
    print(f"клиника: {clinic.name} ({clinic.code})  окружение: {cfg.environment}")
    print(f"уровень alert: {'ЗАБЛОКИРОВАН — ' + block if block else 'доступен'}")

    # --- источник циклов ---------------------------------------------------
    client = teable_client.connect(cfg)
    teable_cfg = cfg.external("teable")
    table_id = str(teable_cfg.get("opu_table_id") or "").strip()
    if not table_id:
        print("[!] teable.opu_table_id пуст — выполните scripts/ews_bootstrap_opu.py")
        return 2
    frame = cycles.load_from_teable(cfg, clinic, client, table_id)
    counts = frame.data["checkpoint"].value_counts().to_dict()
    print(f"\nциклов загружено: {len(frame.data)}  по чекпойнтам: {counts}")

    # --- состояние карт ----------------------------------------------------
    state_path = data_dir / "state" / "ews_charts_state.json"
    if args.replay:
        states: dict = {}
        evaluated: set = set()
        print("replay: состояние карт сброшено, история пересобирается с нуля")
    else:
        states, evaluated = evaluate.load_state(state_path)
        if not states:
            print("состояния карт нет — первый прогон эквивалентен replay")
        else:
            print(f"инкрементально: уже оценено {len(evaluated)} циклов, "
                  f"повторно они не подаются")

    envelopes = envelope.as_lookup(envelope.load_envelopes(
        envelope.envelope_path(cfg, clinic)))
    if envelopes:
        modes = {}
        for entry in envelopes.values():
            for key, value in entry.items():
                if key.endswith("__mode"):
                    modes[value] = modes.get(value, 0) + 1
        print(f"конвертов зарегистрировано: {len(envelopes)} циклов, "
              f"квантилей по режимам: {modes}")
    else:
        print("конвертов нет — карты пойдут на базовой доле, без риск-коррекции")

    # QA-слой конверта: факт против P10-P90. Это единственное, для чего конверт
    # пригоден на этих данных — модель хорошо предсказывает СЧЁТ (corr 0.86-0.97) и
    # не предсказывает ДОЛЮ, поэтому риск-коррекция карт закрыта, а интервал по
    # счёту остаётся осмысленной опорой. Тревог не поднимает.
    closed_frame = frame.data.loc[frame.data["checkpoint"] == cycles.CHECKPOINT_CLOSURE]
    qc_cycles, qc_summary = envelope_qc.build(closed_frame, envelopes, cfg.settings)
    coverage_dropped, coverage_note = envelope_qc.coverage_alarm(qc_summary)
    if not qc_summary.empty:
        print("\nQA конверта (факт против P10-P90):")
        for _, row in qc_summary.iterrows():
            delta = f"{row['delta_pp']:+.1f} пп" if row["delta_pp"] is not None else "—"
            print(f"  {row['stage']:20s} n={row['n']:3d} покрытие {row['coverage']:.3f} "
                  f"(опорное {row['baseline']}) {delta:>10s}  "
                  f"вырожденных {row['degenerate_share']:.0%}")
        if coverage_dropped:
            print(f"  [!] {coverage_note}")

    # Литературный контур: третья опора, независимая от нашей истории.
    literature = prior.gap_table(frame.data, amh_constant=clinic.amh_constant)

    # Просевшее покрытие блокирует эскалацию (settings: localisation.
    # coverage_drop_blocks_escalation): до разбора §4 тревоги остаются watch.
    escalation_block = None
    if coverage_dropped and cfg.settings.get("localisation", {}).get(
            "coverage_drop_blocks_escalation", True):
        escalation_block = "покрытие конверта просело — сначала разбор §4"
    evaluations = evaluate.apply_evaluations(cfg, clinic, frame.data, baselines,
                                             thresholds, states, envelopes,
                                             literature=literature,
                                             already_evaluated=evaluated,
                                             escalation_block=escalation_block)

    levels = pd.Series([item.level for item in evaluations]).value_counts().to_dict()
    print(f"оценено закрытых циклов: {len(evaluations)}  итоги: {levels or '—'}")

    sources: dict[str, int] = {}
    for item in evaluations:
        for info in item.charts.values():
            key = info.get("expected_source", "?")
            sources[key] = sources.get(key, 0) + 1
    print(f"источник ожидания по картам: {sources}")

    # Confounders собираются ТОЛЬКО для сработавших циклов: разбор логов всей
    # когорты занимает минуты, а объяснять надо лишь то, что сработало. Границы
    # «необычного» берутся из базы, которую пересчитывает недельный проход.
    raised = [item for item in evaluations if item.level != evaluate.LEVEL_OK]
    thresholds_env, baseline_note = confounders.load_baseline(cfg, clinic)
    if raised:
        uids = {item.cycle_uid for item in raised}
        subset = frame.data.loc[frame.data["cycle_uid"].isin(uids)]
        contexts = confounders.collect(cfg, clinic, subset)
        print(f"\nconfounders собраны для {len(contexts)} сработавших циклов "
              f"({baseline_note})")
        for item in raised:
            context = contexts.get(item.cycle_uid)
            if context is None:
                continue
            item.miri_status = context.miri_status
            if not thresholds_env:
                item.notes.append(baseline_note)
                continue
            if context.is_unusual(thresholds_env):
                item.corroboration.append({
                    "type": "unusual_window",
                    "detail": f"критических минут {context.critical_minutes:.0f}, "
                              f"смен лота {context.lot_changes}",
                })
                item.corroboration_rank = evaluate.RANK_STRONG

    alerts = alerts_frame(evaluations, cfg.environment, clinic.code)
    if not alerts.empty:
        print("\nтревоги:")
        for _, row in alerts.iterrows():
            print(f"  {row['raised_at']}  {row['level']:5s}  {row['chart']:12s} "
                  f"{row['action']:10s} {row['cycle_uid']}")
            if row["verdict"]:
                print(f"      -> {row['verdict']}")
    else:
        print("\nтревог нет — это нормальный и самый частый исход")

    print("\nсостояние карт:")
    for metric, state in sorted(states.items()):
        h_watch = thresholds.get("h_watch", {}).get(metric)
        h_alert = thresholds.get("h_alert", {}).get(metric)
        print(f"  {metric:6s} статистика {state.statistic:6.2f}  "
              f"порог watch {h_watch}  alert {h_alert}  "
              f"наблюдений {state.observations:5d}  сигналов {state.signals}")

    # Медленный контур: подтверждение по исходу. Тревог не поднимает — считается
    # до записи артефактов, потому что попадает и в панель, и в манифест.
    outcome_windows = outcome.compare_periods(frame.data, clinic.window_bounds()[0])
    outcome_verdict, outcome_note = outcome.confirms_process_trend(outcome_windows)
    print(f"\nмедленный контур: {outcome_note}")

    # Тот же контур с поправкой на пациенток: факт против ожидания DT, VLAD и
    # RA-CUSUM, O/E по месяцам, эмбриологам, среде и лотам. Тоже без тревог.
    outcome_ra_result = outcome_ra_step(cfg, clinic, frame.data, data_dir,
                                        freeze=not args.dry_run)

    if args.dry_run:
        print("\n(dry-run: ничего не записано)")
        return 0

    _write_csv(checkpoints_frame(frame.data, cfg.environment),
               data_dir / "ews_checkpoints.csv")
    if not qc_cycles.empty:
        _write_csv(qc_cycles, data_dir / "ews_envelope_qc.csv")
    # Тревоги дописываются, а не переписываются: в инкрементальном прогоне
    # оцениваются только новые циклы, и перезапись стёрла бы весь журнал.
    alerts_path = data_dir / "ews_alerts.csv"
    if not args.replay and alerts_path.exists() and not alerts.empty:
        previous = pd.read_csv(alerts_path, encoding="utf-8-sig")
        alerts = pd.concat([previous, alerts], ignore_index=True)
        alerts = alerts.drop_duplicates(subset=["cycle_uid", "chart", "raised_at"],
                                        keep="first")
    elif not args.replay and alerts_path.exists() and alerts.empty:
        alerts = pd.read_csv(alerts_path, encoding="utf-8-sig")
    _write_csv(alerts, alerts_path)
    evaluated |= {item.cycle_uid for item in evaluations}
    evaluate.save_state(state_path, states, cfg, clinic, evaluated)

    panel_path = data_dir / "EWS_Panel.html"
    panel.render(cfg, clinic, frame, evaluations, states, baselines, thresholds,
                 panel_path, literature=literature, outcome_windows=outcome_windows,
                 outcome_note=outcome_note, envelope_qc=qc_summary,
                 coverage_note=coverage_note, outcome_ra=outcome_ra_result,
                 alerts_journal=alerts)
    if outcome_ra_result is not None:
        outcome_ra.write_artifacts(outcome_ra_result, data_dir, cfg.environment,
                                   cfg.path(cfg.settings["privacy"]["agent_packet_dir"]),
                                   clinic.code)

    manifest = {
        "run_id": datetime.now().strftime("%Y%m%dT%H%M%S"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "clinic": clinic.code,
        "environment": cfg.environment,
        "mode": "replay" if args.replay else "incremental",
        "cycles_loaded": len(frame.data),
        "cycles_evaluated": len(evaluations),
        "levels": levels,
        "alert_gate": block or "open",
        "outcome_contour": {"verdict": outcome_verdict, "note": outcome_note},
        "outcome_ra": (outcome_ra_result.summary() if outcome_ra_result is not None
                       else {"status": "not_built"}),
        "confounder_baseline": baseline_note,
        "envelope_qc": (qc_summary.to_dict("records") if not qc_summary.empty else []),
        "coverage_blocks_escalation": coverage_dropped,
        "soft_step_status": {
            "miri": "disabled" if not clinic.source_enabled("miri") else "enabled",
            "lots": "disabled" if not clinic.source_enabled("lots") else "enabled",
        },
        "contract_version": "0.1",
    }
    (data_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nартефакты: {data_dir}")
    print(f"панель:    {panel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
