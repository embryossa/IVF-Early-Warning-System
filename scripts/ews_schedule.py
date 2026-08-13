"""Расписание суточного прогона через планировщик Windows.

    python scripts/ews_schedule.py --clinic <code> --install [--time 15:45]
    python scripts/ews_schedule.py --clinic <code> --status
    python scripts/ews_schedule.py --clinic <code> --remove

Создаёт две задачи:

* **ежедневную** — синхронизация, регистрация конверта, оценка. Время по умолчанию
  15:45: мастер-таблица и база OPU обновляются в 15:15, и получасовой запас
  покрывает задержку синхронизации. Раньше запускать нельзя — прогон прочитает
  вчерашние данные и решит, что новых циклов нет.
* **недельную** — обогащение средой и лотами, пересчёт границ «необычного».
  По понедельникам, через час после суточной: разбор логов всей когорты занимает
  минуты, и мешать ему с ежедневной цепочкой незачем.

Задачи запускаются `python`, а не `.bat`: обёртка добавила бы ещё один слой, в
котором теряются коды возврата и кодировка вывода.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

DAILY_TIME = "15:45"      # OPU обновляется в 15:15
WEEKLY_TIME = "16:45"
WEEKLY_DAY = "MON"


def task_names(clinic: str) -> tuple[str, str]:
    return f"EWS_{clinic}_daily", f"EWS_{clinic}_weekly"


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True,
                          encoding="cp866", errors="replace")


def install(clinic: str, daily_time: str, weekly_time: str) -> int:
    python = sys.executable
    # pythonw не используется намеренно: нужен код возврата и текстовый вывод в лог.
    daily_name, weekly_name = task_names(clinic)
    log_dir = ROOT / "data" / clinic / "state"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (daily_name, ["/SC", "DAILY", "/ST", daily_time],
         f'"{python}" "{ROOT / "scripts" / "ews_daily.py"}" --clinic {clinic}'),
        (weekly_name, ["/SC", "WEEKLY", "/D", WEEKLY_DAY, "/ST", weekly_time],
         f'"{python}" "{ROOT / "scripts" / "ews_daily.py"}" --clinic {clinic} --weekly'),
    ]

    for name, schedule, action in jobs:
        result = _run(["schtasks", "/Create", "/TN", name, *schedule,
                       "/TR", action, "/F", "/RL", "LIMITED"])
        if result.returncode != 0:
            print(f"[!] {name}: {result.stdout.strip()} {result.stderr.strip()}")
            return 2
        print(f"  + {name}: {' '.join(schedule)}")
    print(f"\nежедневная задача в {daily_time} (OPU обновляется в 15:15), "
          f"недельная — {WEEKLY_DAY} в {weekly_time}")
    print("логи каждого прогона: data/<clinic>/state/daily_last_run.json")
    return 0


def status(clinic: str) -> int:
    for name in task_names(clinic):
        result = _run(["schtasks", "/Query", "/TN", name, "/FO", "LIST"])
        if result.returncode != 0:
            print(f"  [нет] {name}")
            continue
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        wanted = ("Состояние", "Status", "Время следующего", "Next Run Time",
                  "Последний запуск", "Last Run Time", "Последний результат",
                  "Last Result")
        print(f"  [есть] {name}")
        for line in lines:
            if any(line.startswith(w) for w in wanted):
                print(f"         {line}")
    return 0


def remove(clinic: str) -> int:
    for name in task_names(clinic):
        result = _run(["schtasks", "/Delete", "/TN", name, "/F"])
        print(f"  {'- ' + name if result.returncode == 0 else '[нет] ' + name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", required=True)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--time", default=DAILY_TIME, help="время суточной задачи")
    parser.add_argument("--weekly-time", default=WEEKLY_TIME)
    args = parser.parse_args()

    if args.install:
        return install(args.clinic, args.time, args.weekly_time)
    if args.remove:
        return remove(args.clinic)
    return status(args.clinic)


if __name__ == "__main__":
    raise SystemExit(main())
