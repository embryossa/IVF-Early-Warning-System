"""Досчитать прогноз Digital Twin 7.1 для циклов без него и с новым этапом.

    python scripts/ews_dt_refresh.py --clinic <code> [--dry-run]

Шаг суточной цепочки между синхронизацией листа и регистрацией конверта: берёт
циклы из зеркала EWS, отправляет в службу DT 7.1 (REST, подключение «Аналитика»)
те, у которых в базе «Two-Line IVF — Digital Twin» нет вероятности или изменились
входы, и пишет результат туда же (Runs, Predictions, Cycles). Следующий шаг
цепочки — повторная синхронизация — забирает вероятности в зеркало EWS.

Шаг мягкий. Служба не отвечает или пакет не досчитался за отведённое время — EWS
идёт дальше на прежних вероятностях, недосчитанный пакет забирается следующим
прогоном (`state/dt_refresh.json`). Устройство и правила подстановки — в
`src/ews/dt_refresh.py`.

`--dry-run` ничего не пишет и не ставит в очередь: показывает, что ушло бы в
расчёт, и проверяет пакет службой (`/analytics/validate`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import dt_refresh, teable_client  # noqa: E402
from ews.clinic import load_clinic  # noqa: E402
from ews.config import ConfigError, load  # noqa: E402


def _state_path(cfg, clinic) -> Path:
    return cfg.path(f"data/{clinic.code}/state/dt_refresh.json")


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_results(cfg, clinic, client, tables: dict, results: dict, targets: dict,
                  state: dict, settings: dict, offset: float) -> dict:
    """Записать готовый пакет в базу DT. Возвращает счётчики."""
    batch_id = results["id"]
    finished = results.get("finished") or datetime.utcnow().isoformat() + "Z"
    when = datetime.fromisoformat(finished.replace("Z", "+00:00")).astimezone(
        ZoneInfo(clinic.timezone))
    run_id = dt_refresh.run_id_for(batch_id, when, clinic.code)
    stamp = when.isoformat()
    n_sim = int(settings.get("n_sim", 2000))
    cards = str(settings.get("cards_url", "")).rstrip("/")

    rows = {row["cycle_id"]: row for row in results.get("rows", [])}
    complete = [uid for uid, row in rows.items() if row["status"] == "complete" and uid in targets]
    invalid = Counter()
    for uid, row in rows.items():
        if row["status"] != "complete":
            code = (row.get("error") or {}).get("code") or row["status"]
            invalid[code] += 1
            state["scored"][uid] = {"fingerprint": targets[uid]["fingerprint"] if uid in targets
                                    else None, "invalid": code, "run_id": run_id}

    # Сколько прогонов у цикла уже было — по ключам Predictions.
    seen: dict[str, set] = {}
    for record in client.all_records(tables["Predictions"], projection=["Ключ"]):
        composite = str(record["fields"].get("Ключ") or "")
        if "::" in composite:
            uid_part, run_part = composite.split("::", 1)
            seen.setdefault(uid_part, set()).add(run_part)

    predictions, creates, updates = [], [], []
    for uid in complete:
        target = targets[uid]["target"]
        values = rows[uid].get("values") or {}
        record = target.record
        # Строка базы DT несёт свой cycle_uid (пайплайн); новые получают cycle_uid EWS.
        row_uid = (dt_refresh._text(record["fields"].get("cycle_uid")) if record else None) or uid
        predictions.append(dt_refresh.prediction_row(
            values, row_uid, target.key, run_id, stamp, target, results, n_sim))
        n_runs = len(seen.get(row_uid, set()) | {run_id})
        fields = {**dt_refresh.cycle_facts(target.fields, clinic.name, offset),
                  **dt_refresh.cycle_snapshot(values, run_id, n_runs)}
        if record is None:
            fields.update({"ID пациента": target.key, "cycle_uid": row_uid,
                           "Имя": dt_refresh._text(target.fields.get("Имя")),
                           "Карточка": f"{cards}/{clinic.code}.html#uid={row_uid}" if cards else None})
            creates.append(dt_refresh._clean(fields))
        else:
            if not dt_refresh._text(record["fields"].get("cycle_uid")):
                fields["cycle_uid"] = row_uid
            updates.append((record["id"], fields))
        state["scored"][uid] = {"fingerprint": targets[uid]["fingerprint"], "run_id": run_id,
                                "stage": dt_refresh.stage_of(target.snapshot.patient)}

    kinds = Counter(str((rows[uid].get("values") or {}).get("clinical_probability_kind"))
                    for uid in complete)
    flags = Counter(flag for uid in complete for flag in targets[uid]["target"].snapshot.flags)
    reasons = Counter(targets[uid]["target"].reason for uid in complete)
    run_fields = dt_refresh._clean({
        "Run ID": run_id, "Дата прогона": stamp, "Клиника": clinic.name,
        "Входной файл": "зеркало OPU EWS → IVF DT 7.1 REST analytics",
        "Пациентов всего": len(complete),
        "Версия DT": "v7.1 (REST analytics)",
        "Заметки": (
            f"Автоматический прогон EWS (scripts/ews_dt_refresh.py), пакет {batch_id}, "
            f"as_entered, n_sim={n_sim}, model {str(results.get('model_fingerprint') or '')[:16]}. "
            f"Повод: {dict(reasons)}. Вид вероятности: {dict(kinds)}. "
            f"Подстановки (решение владельца 2026-10-18): {dict(flags) or 'нет'}. "
            f"Отклонено службой: {dict(invalid) or 'нет'}. Исследовательский расчёт: не "
            f"проверен врачом, без калибровки центра; PRAI/TabPFN, DNN, Regression QC, "
            f"Review, OZEMA этим шагом не создаются."),
    })
    client.create(tables["Runs"], run_fields)
    # Сначала Cycles: «Пациентка» в Predictions — ссылка по ключу цикла, и на ещё
    # не созданную строку она молча остаётся пустой.
    made = client.create_records(tables["Cycles"], creates)
    client.update_records(tables["Cycles"], updates)
    if predictions:
        client.create_records(tables["Predictions"], predictions)
    return {"run_id": run_id, "рассчитано": len(complete), "новых строк Cycles": len(made),
            "обновлено Cycles": len(updates), "отклонено": dict(invalid),
            "вид": dict(kinds), "подстановки": dict(flags)}


def rebuild_viewer(cfg, clinic) -> None:
    """Пересобрать вьюер карточек базы DT, чтобы новые прогнозы были видны сразу.

    Сборщик — чужой проект (Teable-пайплайн клиники, путь — в локальном конфиге): запускается его
    скриптом, из его каталога, со своим конфигом. Мягко: прогноз уже в базе, и
    несобранная страница его не отменяет. Из вывода печатается только итоговая
    строка — на странице ФИО, но в строке итога их нет.
    """
    script = str((cfg.machine.get("digital_twin_api") or {}).get("cards_build") or "").strip()
    if not script:
        return
    path = Path(script)
    if not path.exists():
        print(f"[warn] вьюер карточек: нет сборщика {path}")
        return
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        done = subprocess.run([sys.executable, path.name, "--clinic", clinic.code],
                              cwd=path.parent, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[warn] вьюер карточек не пересобран: {exc}")
        return
    summary = [line.strip() for line in done.stdout.splitlines() if line.strip().startswith("cards:")]
    if done.returncode != 0 or not summary:
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["нет вывода"]
        print(f"[warn] вьюер карточек не пересобран (код {done.returncode}): {tail[0][:200]}")
        return
    print(f"вьюер карточек: {summary[-1]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", help="код профиля из config/clinics/")
    parser.add_argument("--dry-run", action="store_true",
                        help="показать и проверить пакет, ничего не записывая")
    args = parser.parse_args()

    try:
        cfg = load(args.clinic)
        clinic = load_clinic(cfg, args.clinic)
    except ConfigError as exc:
        print(f"[!] {exc}")
        return 2
    settings = cfg.settings.get("dt_refresh") or {}
    if not settings.get("enabled", False):
        print("прогноз DT: выключен (settings.dt_refresh.enabled)")
        return 0

    teable_cfg = cfg.external("teable")
    dt_base = str(teable_cfg.get("dt_base_id") or "").strip()
    dt_cycles = str(teable_cfg.get("dt_cycles_table_id") or "").strip()
    if not dt_base or not dt_cycles:
        print("[!] teable.dt_base_id / dt_cycles_table_id не заданы в machine.local.yaml")
        return 2
    offset = float(cfg.settings["sync"]["dt_teable"]["date_utc_offset_hours"])
    rules = dt_refresh.Rules.from_profile(clinic.profile, clinic.columns)

    try:
        api = dt_refresh.connect_api(cfg)
        health = api.health()
    except (ConfigError, dt_refresh.ApiError, OSError) as exc:
        print(f"[!] служба DT недоступна: {exc}")
        return 3
    if health.get("status") != "ready":
        print(f"[!] служба DT не готова: {health.get('status')} (лицензия?)")
        return 3

    client = teable_client.connect(cfg, dry_run=args.dry_run)
    tables = teable_client.resolve_tables(cfg, client, dt_base)
    missing = [name for name in ("Runs", "Cycles", "Predictions") if name not in tables]
    if missing:
        print(f"[!] в базе DT нет таблиц {missing}")
        return 2

    state_path = _state_path(cfg, clinic)
    state = _load_state(state_path)
    state.setdefault("scored", {})

    mirror = [r["fields"] for r in client.all_records(
        teable_cfg["opu_table_id"], projection=dt_refresh.MIRROR_FIELDS)]
    dt_rows = {}
    for record in client.all_records(dt_cycles):
        key = (dt_refresh._text(record["fields"].get("ID пациента")) or "").upper().replace(" ", "")
        if key:
            dt_rows[key] = record

    targets, counts = dt_refresh.select(mirror, dt_rows, state, rules, offset)
    print(f"окружение: {cfg.environment}  служба DT: {health.get('status')}")
    print("циклы: " + " · ".join(f"{k} {v}" for k, v in counts.items() if v))
    by_uid = {t.uid: {"target": t, "fingerprint": t.snapshot.fingerprint} for t in targets}

    pending = state.get("pending_batch")
    if pending and not args.dry_run:
        # Пакет прошлого прогона: забрать, если досчитан. Цели пересобираются по
        # текущему зеркалу — в базу пишется то, что посчитано, а отпечаток берётся
        # тот, с которым пакет уходил.
        results = api.results(pending["id"])
        if results is None:
            print(f"пакет {pending['id'][:8]} прошлого прогона ещё считается — ждём его, "
                  f"новый не ставим")
            _save_state(state_path, state)
            return 0
        sent = {uid: fp for uid, fp in pending.get("fingerprints", {}).items()}
        lookup = {t.uid: t for t in targets}
        for uid, fingerprint in sent.items():
            if uid in lookup:
                by_uid[uid] = {"target": lookup[uid], "fingerprint": fingerprint}
        summary = write_results(cfg, clinic, client, tables, results, by_uid, state,
                                settings, offset)
        api.delete(pending["id"])
        state.pop("pending_batch", None)
        _save_state(state_path, state)
        print(f"забран пакет прошлого прогона: {summary}")
        rebuild_viewer(cfg, clinic)
        return 0

    if not targets:
        _save_state(state_path, state) if not args.dry_run else None
        print("считать нечего: у всех циклов прогноз актуален")
        return 0

    reasons = Counter(t.reason for t in targets)
    flags = Counter(f for t in targets for f in t.snapshot.flags)
    stages = Counter(dt_refresh.stage_of(t.snapshot.patient) for t in targets)
    print(f"в расчёт: {len(targets)} — {dict(reasons)}")
    print(f"  этап: {dict(stages)}")
    print(f"  подстановки: {dict(flags) or 'нет'}")

    n_sim = int(settings.get("n_sim", 2000))
    cycles = [{"cycle_id": t.uid, "patient": t.snapshot.patient} for t in targets]
    if args.dry_run:
        check = api.validate(cycles, n_sim)
        rows = check.get("rows") or check.get("cycles") or []
        verdict = Counter(r.get("status") or ("valid" if r.get("valid") else "invalid") for r in rows)
        codes = Counter((r.get("error") or {}).get("code") for r in rows if r.get("error"))
        print(f"проверка службой: {dict(verdict) or check}  {dict(codes) if codes else ''}")
        print("(dry-run: ничего не записано и не поставлено в очередь)")
        return 0

    batch = api.submit(cycles[:500], n_sim)
    batch_id = batch["id"]
    state["pending_batch"] = {"id": batch_id, "submitted": datetime.now().isoformat(timespec="seconds"),
                              "fingerprints": {uid: v["fingerprint"] for uid, v in by_uid.items()}}
    _save_state(state_path, state)
    print(f"пакет {batch_id[:8]} поставлен: {batch.get('total', len(cycles))} циклов, "
          f"ждём до {settings.get('max_wait_minutes', 30)} мин")

    results = api.wait(batch_id, float(settings.get("max_wait_minutes", 30)),
                       float(settings.get("poll_seconds", 20)))
    if results is None:
        print("пакет ещё считается — заберёт следующий прогон; EWS идёт на прежних вероятностях")
        return 0
    summary = write_results(cfg, clinic, client, tables, results, by_uid, state, settings, offset)
    api.delete(batch_id)
    state.pop("pending_batch", None)
    _save_state(state_path, state)
    print(f"записано в базу DT: {summary}")
    rebuild_viewer(cfg, clinic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
