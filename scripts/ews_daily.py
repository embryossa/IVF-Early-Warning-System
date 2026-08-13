"""Суточный оркестратор: один вызов, который делает всё нужное для новых циклов.

    python scripts/ews_daily.py --clinic <code> [--weekly] [--dry-run]

Порядок шагов не произволен — каждый следующий опирается на предыдущий:

1. **Синхронизация OPU** — забрать новые и изменённые строки в локальный Teable.
2. **Регистрация конверта** — для циклов, которые СЕГОДНЯ стоят на T0 или T+1.
   Пропустить этот шаг нельзя: конверт замораживается в свой момент, и завтра он
   был бы уже ретроспективным.
3. **Прогон** — оценка закрывшихся циклов, тревоги, панель.
4. **Недельное обогащение** (`--weekly`) — среда, лоты, пересчёт границ «необычного».

Шаги 1-3 идут ежедневно, шаг 4 — раз в неделю: разбор логов всей когорты занимает
минуты, а границы так часто не меняются.

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


def run_step(label: str, command: list[str], soft: bool) -> dict:
    """Один шаг. Мягкий сбой записывается, но не прерывает цепочку."""
    started = datetime.now()
    print(f"\n=== {label} ===", flush=True)
    completed = subprocess.run(command, cwd=str(ROOT), text=True,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinic", required=True, help="код профиля из config/clinics/")
    parser.add_argument("--weekly", action="store_true",
                        help="добавить недельное обогащение средой и лотами")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    clinic = ["--clinic", args.clinic]
    dry = ["--dry-run"] if args.dry_run else []

    steps = [
        ("синхронизация OPU", [python, "scripts/ews_sync_opu.py", *clinic, *dry], True),
        ("регистрация конверта", [python, "scripts/ews_register_envelope.py",
                                  *clinic, *dry], True),
        ("прогон", [python, "scripts/ews_run.py", *clinic, *dry], False),
    ]
    if args.weekly:
        steps.append(("недельное обогащение",
                      [python, "scripts/ews_enrich_miri.py", *clinic, *dry], True))

    print(f"EWS daily — клиника {args.clinic}, {datetime.now():%Y-%m-%d %H:%M}")
    results = []
    for label, command, soft in steps:
        result = run_step(label, command, soft)
        results.append(result)
        if not result["ok"] and not soft:
            break

    failed = [r for r in results if not r["ok"] and not r["soft"]]
    skipped = [r for r in results if not r["ok"] and r["soft"]]

    print(f"\n=== итог {datetime.now():%H:%M} ===")
    for result in results:
        mark = "ok" if result["ok"] else ("пропущен" if result["soft"] else "СБОЙ")
        print(f"  [{mark:8s}] {result['step']:24s} {result['seconds']:6.1f} с")

    if not args.dry_run:
        log_dir = ROOT / "data" / args.clinic / "state"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "daily_last_run.json").write_text(json.dumps({
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "weekly": args.weekly, "steps": results,
            "soft_skipped": [r["step"] for r in skipped],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
