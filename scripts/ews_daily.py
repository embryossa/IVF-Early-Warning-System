"""Оркестратор прогона: один вызов, который делает всё нужное для новых циклов.

    python scripts/ews_daily.py --clinic <code> [--weekly] [--dry-run]

Порядок шагов не произволен — каждый следующий опирается на предыдущий:

1. **Обновление экспорта листа** — забрать свежий CSV мастер-таблицы OPU.
2. **Синхронизация OPU** — забрать новые и изменённые строки в локальный Teable.
3. **Регистрация конверта** — для циклов, которые СЕГОДНЯ стоят на T0 или T+1.
   Конверт замораживается в свой момент: завтра он был бы уже ретроспективным.
4. **Добор конвертов** (`--weekly`) — для циклов, прошедших чекпойнт между
   прогонами; строки помечаются `retrospective`, см. ниже.
5. **Прогон** — оценка закрывшихся циклов, тревоги, панель.
6. **Обогащение средой** (`--weekly`) — MIRI, лоты, пересчёт границ «необычного».

## Расписание — суточное, с недельным понедельником

Решение владельца (2026-09-03): цепочка идёт **каждый день**. Понедельник —
единственный день с ключом `--weekly`; в остальные дни задача идёт без него.
Расписание ставит `ews_schedule.py` двумя задачами, разведёнными по дням недели.

Суточная каденция возвращена ради двух вещей сразу. Первая — оплодотворение: T+1
и без того регистрируется в свой день, но до карт результат доходил только с
оценкой на T+6, а она при недельном расписании ждала понедельника. Вторая —
конверты: цикл с пункцией во вторник к понедельнику уже закрыт, на своём
чекпойнте не стоит, и проспективный путь (шаг 3) его не видит — `envelope.pending`
смотрит только на текущий чекпойнт. Суточная задача застаёт циклы на чекпойнтах, и
конверт снова замораживается вовремя, то есть остаётся прогнозом.

**Почему шаг 4 при этом остаётся.** Пропущенный день — выключенная машина, нет
сети, упавший Teable — снова оставил бы циклы вовсе без конверта, а слой QA «факт
против P10–P90» без части когорты. Недельный добор подбирает такие хвосты. Он
честен и ранним предупреждением не притворяется: строки уходят с
`registration_mode=retrospective`, и в них модель обучалась в том числе на этих
циклах. Повторной записи не будет — `pending` отсеивает уже зарегистрированные
чекпойнты.

**Почему среда осталась недельной.** Выгрузка логов инкубатора приходит понедельной
нарезкой: чаще её читать нечего. Проход по всей когорте занимает минуты, и держать
его в суточной цепочке — платить их каждый день за одни и те же файлы. Суточному
прогону логи и не нужны: confounders он собирает только для сработавших циклов, а
границы «необычного» берёт из базы, которую пересчитал последний понедельник.

**Мягкие шаги не роняют прогон.** Если модель конверта недоступна или логов ещё нет,
соответствующий шаг пропускается с записью причины, а оценка закрывшихся циклов
выполняется как обычно. Расписание не должно останавливаться из-за того, чего в этот
день не случилось.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Сколько журналов прогонов держать на диске. Суточное расписание даёт 7 файлов в
# неделю; месяца с лишним истории хватает, чтобы разобрать «когда это началось».
KEEP_LOGS = 40


class Tee:
    """Вывод одновременно на экран и в файл.

    Без этого текст суточного прогона existовал только в консоли, которой у
    задачи планировщика нет: `daily_last_run.json` хранит коды возврата и секунды,
    но не то, ЧТО именно сказал упавший шаг. Разбирать ночной сбой было нечем.
    """

    def __init__(self, stream, handle) -> None:
        self.stream, self.handle = stream, handle

    def write(self, text: str) -> int:
        self.handle.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.handle.flush()
        self.stream.flush()


def prune_logs(log_dir: Path) -> None:
    logs = sorted(log_dir.glob("daily_*.log"))
    for stale in logs[:-KEEP_LOGS]:
        stale.unlink(missing_ok=True)


def child_env() -> dict:
    """Окружение для шагов: UTF-8 принудительно.

    У задачи планировщика консоли нет, и Python шага кодирует свой вывод в ANSI
    кодовой странице системы (здесь cp1251), тогда как читаем мы его как UTF-8.
    Результат — журнал прогона, целиком состоящий из «?????????» ровно в том
    случае, ради которого он и заводился. Задавать переменную в самой задаче
    ненадёжно: её унаследует не всякий способ запуска. Проще не зависеть от
    окружения вовсе.
    """
    import os  # noqa: PLC0415

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_step(label: str, command: list[str], soft: bool) -> dict:
    """Один шаг. Мягкий сбой записывается, но не прерывает цепочку."""
    started = datetime.now()
    print(f"\n=== {label} ===", flush=True)
    completed = subprocess.run(command, cwd=str(ROOT), text=True, env=child_env(),
                               capture_output=True, encoding="utf-8", errors="replace")
    output = (completed.stdout or "") + (completed.stderr or "")
    print(output.strip()[-4000:], flush=True)
    seconds = (datetime.now() - started).total_seconds()

    ok = completed.returncode == 0
    if not ok and not soft:
        print(f"[!] {label}: код {completed.returncode} — цепочка остановлена")
    elif not ok:
        print(f"[warn] {label}: код {completed.returncode} — мягкий шаг, продолжаем")
    return {"step": label, "returncode": completed.returncode, "ok": ok,
            "soft": soft, "seconds": round(seconds, 1)}


def build_steps(python: str, clinic: str, *, weekly: bool,
                dry_run: bool = False) -> list[tuple[str, list[str], bool]]:
    """Состав цепочки: (название, команда, мягкий ли шаг).

    Вынесено из `main` отдельной функцией не ради красоты: порядок шагов и то,
    какие из них мягкие, — единственное содержательное решение этого файла, и
    проверять его надо, не запуская конвейер целиком. Ошибка здесь молчалива:
    жёсткий шаг, ставший мягким, превращает сбой оценки в «прогон прошёл».
    """
    clinic_args = ["--clinic", clinic]
    dry = ["--dry-run"] if dry_run else []
    steps: list[tuple[str, list[str], bool]] = [
        # Без обновления экспорта суточная задача синхронизировала бы вчерашний
        # файл и каждый день сообщала «без изменений». Шаг мягкий: нет сети —
        # идём по прошлому экспорту, но молча этого не делаем.
        ("обновление экспорта листа",
         [python, "scripts/ews_fetch_opu_sheet.py", *clinic_args, *dry], True),
        ("синхронизация OPU",
         [python, "scripts/ews_sync_opu.py", *clinic_args, *dry], True),
        # Циклы без вероятности DT и с новым этапом воронки считаются службой DT 7.1
        # и пишутся в базу DT (решение владельца 2026-10-18). Шаг мягкий: служба
        # недоступна или пакет не досчитан — EWS идёт на прежних вероятностях, пакет
        # заберёт следующий прогон.
        ("прогноз DT",
         [python, "scripts/ews_dt_refresh.py", *clinic_args, *dry], True),
        # Повтор синхронизации забирает только что посчитанные вероятности в зеркало
        # EWS в тот же день. Идемпотентен: неизменённые строки не переписываются.
        ("вероятности DT в зеркало",
         [python, "scripts/ews_sync_opu.py", *clinic_args, *dry], True),
        ("регистрация конверта",
         [python, "scripts/ews_register_envelope.py", *clinic_args, *dry], True),
    ]
    if weekly:
        # Добор конвертов за неделю. При суточной задаче основной путь — шаг 3, и
        # сюда попадают только хвосты пропущенных дней: цикл, чей чекпойнт пришёлся
        # на день без прогона, на своём чекпойнте уже не стоит, и проспективный путь
        # его не видит (envelope.pending смотрит только на текущий чекпойнт). Без
        # этого шага такие циклы остались бы вовсе без конверта, а слой QA «факт
        # против P10-P90» — без части когорты.
        #
        # Пометка честная: строки уходят с registration_mode=retrospective и
        # доказательством раннего обнаружения не являются — модель обучалась в том
        # числе на этих циклах. Повторной записи не будет: pending отсеивает уже
        # зарегистрированные чекпойнты.
        steps.append(("добор конвертов за неделю",
                      [python, "scripts/ews_register_envelope.py", *clinic_args,
                       "--retrospective", *dry], True))
    # Единственный жёсткий шаг: всё остальное — обстановка, а это сама оценка.
    steps.append(("прогон", [python, "scripts/ews_run.py", *clinic_args, *dry], False))
    if weekly:
        steps.append(("недельное обогащение",
                      [python, "scripts/ews_enrich_miri.py", *clinic_args, *dry], True))
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", required=True, help="код профиля из config/clinics/")
    parser.add_argument("--weekly", action="store_true",
                        help="добор конвертов за неделю + обогащение средой и лотами; "
                             "по расписанию идёт только в понедельник")
    parser.add_argument("--dry-run", action="store_true")
    # Флаг ставит задача планировщика. Различие не косметическое: ручной прогон не
    # доказывает, что расписание живо, поэтому «пропущен ли плановый прогон»
    # считается только по записям `по расписанию`.
    parser.add_argument("--triggered-by", choices=["schedule", "manual"],
                        default="manual", help="кто запустил цепочку")
    args = parser.parse_args()

    steps = build_steps(sys.executable, args.clinic, weekly=args.weekly,
                        dry_run=args.dry_run)

    # Свой вывод — тоже UTF-8: без консоли он ушёл бы в ANSI и в файле журнала
    # разошёлся бы с выводом шагов.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    started_at = datetime.now()
    run_id = started_at.strftime("%Y%m%dT%H%M%S")
    log_dir = ROOT / "data" / args.clinic / "state"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Текст прогона пишется на диск с первой строки, а не в конце: если цепочка
    # оборвётся жёстко, в файле останется то, что успело произойти. Ради этого же
    # файл открывается построчно-буферизованным.
    log_path = log_dir / f"daily_{run_id}.log"
    handle = None if args.dry_run else log_path.open("w", encoding="utf-8", buffering=1)
    stdout = sys.stdout
    if handle is not None:
        sys.stdout = Tee(stdout, handle)

    try:
        print(f"EWS daily — клиника {args.clinic}, {started_at:%Y-%m-%d %H:%M}, "
              f"запуск {args.triggered_by}, прогон {run_id}")
        results = []
        interrupted = False
        # Прерывание с клавиатуры ловится здесь не ради красивого выхода, а ради
        # журнала. 2026-09-03 11:24 ручной прогон был остановлен Ctrl+C на мягком
        # шаге; он успел поднять тревогу (watch FR), но до записи в Teable
        # управление не дошло, и в журнале прогонов этой работы нет вовсе. Прогон,
        # который что-то сделал и не отчитался, хуже непрошедшего: его молчание
        # неотличимо от «задача не сработала».
        try:
            for label, command, soft in steps:
                result = run_step(label, command, soft)
                results.append(result)
                if not result["ok"] and not soft:
                    break
        except KeyboardInterrupt:
            interrupted = True
            print()
            print("[!] прервано с клавиатуры — итог записывается как есть")

        failed = [r for r in results if not r["ok"] and not r["soft"]]
        skipped = [r for r in results if not r["ok"] and r["soft"]]

        print(f"\n=== итог {datetime.now():%H:%M} ===")
        for result in results:
            mark = "ok" if result["ok"] else ("пропущен" if result["soft"] else "СБОЙ")
            print(f"  [{mark:8s}] {result['step']:24s} {result['seconds']:6.1f} с")

        if args.dry_run:
            return 1 if failed else 0
        notes = (["прогон прерван с клавиатуры: цепочка выполнена не полностью"]
                 if interrupted else [])

        (log_dir / "daily_last_run.json").write_text(json.dumps({
            "run_id": run_id,
            "started_at": started_at.isoformat(timespec="seconds"),
            "triggered_by": args.triggered_by,
            "weekly": args.weekly, "steps": results,
            "soft_skipped": [r["step"] for r in skipped],
            "log": log_path.name,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        prune_logs(log_dir)

        journal(args, run_id, started_at, results, notes=notes)
        return 1 if failed or interrupted else 0
    finally:
        sys.stdout = stdout
        if handle is not None:
            handle.close()


def journal(args, run_id: str, started_at: datetime, results: list[dict],
            *, notes: list[str] | None = None) -> None:
    """Записать итог прогона в журнал Teable. Мягко: прогон уже состоялся."""
    from ews import runlog  # noqa: PLC0415
    from ews.config import ConfigError, load  # noqa: PLC0415

    manifest_path = ROOT / "data" / args.clinic / "run_manifest.json"
    manifest = None
    notes = list(notes or [])
    try:
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # Манифеста нет, когда цепочка не дошла до прогона. Это само по себе
        # содержание записи, а не повод её не делать.
        notes.append(f"манифест прогона не прочитан ({exc.__class__.__name__}) — "
                     f"оценка, вероятно, не выполнялась")
    else:
        # Манифест на диске мог остаться от ПРОШЛОГО прогона: шаг оценки мягким не
        # является, но упасть может, и тогда файл не обновится. Записать его числа
        # как сегодняшние — значит отчитаться о работе, которой не было.
        try:
            produced = datetime.fromisoformat(str(candidate.get("started_at")))
        except (TypeError, ValueError):
            produced = None
        if produced is not None and produced >= started_at:
            manifest = candidate
        else:
            when = f"{produced:%Y-%m-%d %H:%M}" if produced else "неизвестно когда"
            notes.append(f"манифест прогона от {when} старше этой цепочки — "
                         f"оценка в ней не отработала, числа не берутся")

    try:
        cfg = load(args.clinic)
    except ConfigError as exc:
        print(f"[warn] журнал прогонов: конфигурация не загрузилась — {exc}")
        return

    trigger = (runlog.TRIGGER_SCHEDULE if args.triggered_by == "schedule"
               else runlog.TRIGGER_MANUAL)
    record = runlog.build_record(cfg, run_id=run_id, started_at=started_at,
                                 weekly=args.weekly, triggered_by=trigger,
                                 results=results, manifest=manifest, notes=notes)
    ok, detail = runlog.write(cfg, record)
    print(f"журнал прогонов: {'записан' if ok else 'НЕ записан — ' + detail} "
          f"(итог «{record['Итог']}»)")


if __name__ == "__main__":
    raise SystemExit(main())
