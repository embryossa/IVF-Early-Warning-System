"""Расписание прогона через планировщик Windows.

    python scripts/ews_schedule.py --clinic <code> --install [--time 15:45]
    python scripts/ews_schedule.py --clinic <code> --status
    python scripts/ews_schedule.py --clinic <code> --remove

Создаются **две** задачи, и вместе они дают ровно один прогон в сутки:

| задача | когда | чем отличается |
|---|---|---|
| `EWS_<clinic>_daily` | вторник–воскресенье | без `--weekly`: конверты, оценка, панель |
| `EWS_<clinic>_weekly` | понедельник | плюс добор конвертов и логи инкубатора |

Дни разведены намеренно. Две задачи, идущие в один день, писали бы в журнал две
строки «по расписанию» за понедельник и дважды прокручивали бы одну и ту же
синхронизацию; разнесение по дням недели оставляет прежнее «одна задача — один
ответ», но уже на суточной сетке.

Решение владельца (2026-09-03): суточная каденция возвращена. Недельная сетка
задерживала обнаружение сдвига по оплодотворению (T+1 регистрируется в свой день,
но оценка на T+6 доходила до карт только в понедельник) и делала конверты
ретроспективными: цикл с пункцией во вторник к понедельнику уже закрыт, на своём
чекпойнте не стоит и проспективным путём не регистрируется. Суточная задача
застаёт циклы на чекпойнтах — конверт снова становится прогнозом. Разбор логов
MIRI остаётся недельным: выгрузка приходит понедельной нарезкой, чаще её читать
нечего, а проход по всей когорте занимает минуты.

Время по умолчанию 15:45: мастер-таблица и база OPU обновляются в 15:15, и
получасовой запас покрывает задержку синхронизации. Раньше запускать нельзя —
прогон прочитает вчерашние данные и решит, что новых циклов нет.

Задача запускается `python`, а не `.bat`: обёртка добавила бы ещё один слой, в
котором теряются коды возврата и кодировка вывода. **Claude Code для прогона не
нужен**: планировщик исполняет интерпретатор проекта напрямую, и ничего, кроме
Windows, Python и поднятого Teable, для этого не требуется.

## Почему задача регистрируется через XML

`schtasks /Create` с ключами даёт задачу с умолчаниями, три из которых для
лабораторной машины неверны, и все три отказывают молча — задача просто не
запускается, а «Последний запуск» остаётся прошлым:

1. **`LogonType=InteractiveToken`** — задача идёт только когда этот пользователь
   залогинен интерактивно. В 15:45 в лаборатории за машиной может никого не быть,
   сеанс может быть заблокирован или завершён. Здесь ставится `S4U`: прогон идёт
   независимо от того, вошёл ли кто-нибудь, и пароль для этого не нужен.
2. **`DisallowStartIfOnBatteries`/`StopIfGoingOnBatteries=true`** — машина за ИБП,
   отчитавшимся как батарея, прогон пропустит или оборвёт на середине. Выключено.
3. **Нет `StartWhenAvailable`** — прогон, пропущенный из-за выключенной машины, не
   догоняется. Для суточной задачи это потеря дня оценки, для недельной —
   пропущенный разбор логов и добор конвертов. Включено для обеих.

Ключ `--triggered-by schedule` в действии задачи — не украшение: по нему журнал
прогонов отличает плановый запуск от ручного, и только по плановым видно, что
расписание живо.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN_TIME = "15:45"        # OPU обновляется в 15:15
WEEKLY_DAY = "понедельник"
DAILY_DAYS = "вторник–воскресенье"

# Сколько дней без планового прогона считать пропуском. Суточное расписание: сутки
# допуска сверх одних покрывают перенос из-за выключенной машины. Меньше нельзя —
# прогон, догнанный `StartWhenAvailable` на следующее утро, считался бы пропуском.
STALE_AFTER_DAYS = 2

# Потолок на прогон. Самая долгая часть — регистрация конверта (TabPFN обучается в
# контексте на 16k строк) и недельный разбор логов; обе укладываются в минуты.
# Двух часов хватает с запасом, а зависшая задача не должна висеть сутками и
# блокировать следующую (MultipleInstancesPolicy=IgnoreNew).
EXECUTION_TIME_LIMIT = "PT2H"

TASK_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{description}</Description>
    <URI>\\{name}</URI>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <UserId>{sid}</UserId>
      <LogonType>S4U</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowHardTerminate>true</AllowHardTerminate>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>{limit}</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{start}</StartBoundary>
      <Enabled>true</Enabled>
      {schedule}
    </CalendarTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>{command}</Command>
      <Arguments>{arguments}</Arguments>
      <WorkingDirectory>{workdir}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""

WEEKLY_SCHEDULE = ("<ScheduleByWeek><WeeksInterval>1</WeeksInterval>"
                   "<DaysOfWeek><Monday /></DaysOfWeek></ScheduleByWeek>")

# Все дни, кроме понедельника: понедельник занят недельным прогоном, который делает
# то же самое и сверх того. Календарный триггер по дням недели, а не ScheduleByDay,
# именно ради этого исключения — иначе в понедельник цепочка шла бы дважды.
DAILY_SCHEDULE = ("<ScheduleByWeek><WeeksInterval>1</WeeksInterval><DaysOfWeek>"
                  "<Tuesday /><Wednesday /><Thursday /><Friday /><Saturday />"
                  "<Sunday /></DaysOfWeek></ScheduleByWeek>")


def daily_task_name(clinic: str) -> str:
    return f"EWS_{clinic}_daily"


def weekly_task_name(clinic: str) -> str:
    return f"EWS_{clinic}_weekly"


def task_names(clinic: str) -> list[str]:
    return [daily_task_name(clinic), weekly_task_name(clinic)]


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True,
                          encoding="cp866", errors="replace")


def _current_sid() -> str:
    """SID текущего пользователя — от него задача и будет работать.

    Через `whoami /user`, а не через имя: имя пользователя локализуется и
    переименовывается, SID — нет.
    """
    result = _run(["whoami", "/user", "/fo", "csv", "/nh"])
    if result.returncode == 0 and "," in result.stdout:
        return result.stdout.strip().rsplit(",", 1)[-1].strip().strip('"')
    raise RuntimeError(f"не удалось определить SID: {result.stdout} {result.stderr}")


def _register(name: str, description: str, start: str, schedule: str,
              arguments: str) -> tuple[bool, str]:
    xml = TASK_XML.format(
        name=name, description=description, sid=_current_sid(),
        limit=EXECUTION_TIME_LIMIT, start=start, schedule=schedule,
        command=f'"{sys.executable}"', arguments=arguments, workdir=str(ROOT),
    )
    # UTF-16 обязателен: планировщик читает объявление кодировки в первой строке и
    # на UTF-8 отвечает невнятной ошибкой разбора.
    handle = tempfile.NamedTemporaryFile("wb", suffix=".xml", delete=False)
    try:
        handle.write(xml.encode("utf-16"))
        handle.close()
        result = _run(["schtasks", "/Create", "/TN", name, "/XML", handle.name, "/F"])
    finally:
        Path(handle.name).unlink(missing_ok=True)
    if result.returncode != 0:
        return False, f"{result.stdout.strip()} {result.stderr.strip()}".strip()
    return True, ""


def install(clinic: str, run_time: str) -> int:
    script = ROOT / "scripts" / "ews_daily.py"
    if not script.exists():
        print(f"[!] нет {script} — ставить в расписание нечего")
        return 2
    (ROOT / "data" / clinic / "state").mkdir(parents=True, exist_ok=True)

    base_args = f'"{script}" --clinic {clinic}'
    trigger = "--triggered-by schedule"
    # 2026-01-05 — понедельник. Для суточной задачи это тоже годная точка отсчёта:
    # дни отбирает DaysOfWeek, а не StartBoundary.
    start = f"2026-01-05T{run_time}:00"
    plan = [
        (daily_task_name(clinic), f"EWS {clinic}: суточный прогон",
         DAILY_SCHEDULE, f"{base_args} {trigger}"),
        (weekly_task_name(clinic), f"EWS {clinic}: недельный прогон (среда и лоты)",
         WEEKLY_SCHEDULE, f"{base_args} --weekly {trigger}"),
    ]
    for name, description, schedule, arguments in plan:
        ok, detail = _register(name, description, start, schedule, arguments)
        if not ok:
            print(f"[!] {name}: {detail}")
            return 2
        print(f"  + {name}")

    print(f"\nдве задачи, вместе — один прогон в сутки в {run_time} "
          f"(мастер-таблица OPU обновляется в 15:15):")
    print(f"  {DAILY_DAYS}: экспорт листа, синхронизация, конверты, оценка, панель")
    print(f"  {WEEKLY_DAY}: то же плюс добор конвертов, среда и лоты, "
          f"границы «необычного»")
    print("работают без входа пользователя (S4U), пропуск догоняется, "
          "батарея прогон не отменяет")
    print(f"интерпретатор: {sys.executable}")
    print("логи прогонов: data/<clinic>/state/daily_<прогон>.log, "
          "журнал в Teable: таблица Runs")
    return 0


def _query(name: str) -> dict[str, str] | None:
    result = _run(["schtasks", "/Query", "/TN", name, "/FO", "LIST", "/V"])
    if result.returncode != 0:
        return None
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def _pick(fields: dict[str, str], *names: str) -> str:
    """Значение по любому из вариантов имени — вывод schtasks локализован."""
    for name in names:
        if name in fields and fields[name]:
            return fields[name]
    return "—"


# SCHED_S_TASK_HAS_NOT_RUN. Планировщик отдаёт его вместе с датой 30.11.1999, и
# без перевода это выглядит как код ошибки у исправной задачи.
NEVER_RAN_CODE = "267011"


def _last_run(fields: dict[str, str]) -> str:
    # Русская локаль бывает в двух редакциях: «последнего/Последний» и
    # «прошлого/Прошлый» (Windows 11 26200). Не узнав имя, --status печатал
    # «— КОД —» у исправной задачи.
    when = _pick(fields, "Last Run Time", "Время последнего запуска",
                 "Время прошлого запуска")
    code = _pick(fields, "Last Result", "Последний результат", "Прошлый результат")
    if code == NEVER_RAN_CODE or when.startswith("30.11.1999"):
        return "ещё не запускалась"
    if code == "—":
        return f"{when}  результат не прочитан из вывода schtasks"
    verdict = "ok" if code == "0" else f"КОД {code}"
    return f"{when}  {verdict}"


def status(clinic: str) -> int:
    problems = 0
    for name in task_names(clinic):
        fields = _query(name)
        if fields is None:
            print(f"  [нет] {name} — выполните --install")
            problems += 1
            continue
        print(f"  [есть] {name}")
        print(f"         состояние:  {_pick(fields, 'Scheduled Task State', 'Состояние')}")
        print(f"         следующий:  {_pick(fields, 'Next Run Time', 'Время следующего запуска')}")
        print(f"         последний:  {_last_run(fields)}")

        # Задача, оставшаяся от прошлого расположения проекта или от системного
        # интерпретатора, выглядит совершенно здоровой и не работает. Проверяем то,
        # на что она реально указывает.
        action = _pick(fields, "Task To Run", "Задача для выполнения")
        for token in ("python.exe", "ews_daily.py"):
            piece = next((p for p in action.replace('"', " ").split()
                          if p.endswith(token)), None)
            if piece is None:
                print(f"         [!] в команде задачи нет {token}: {action}")
                problems += 1
            elif not Path(piece).exists():
                print(f"         [!] путь из задачи не существует: {piece}")
                problems += 1
        # Перепутанный ключ `--weekly` не ломает ничего заметного: суточная задача с
        # ним каждый день пятью минутами разбирает логи, недельная без него никогда
        # не пересчитывает границы «необычного» — и подтверждение по среде тихо
        # остаётся недоступным. Обе ошибки видны только здесь.
        wants_weekly = name == weekly_task_name(clinic)
        if ("--weekly" in action) is not wants_weekly:
            print(f"         [!] ключ --weekly {'потерян' if wants_weekly else 'лишний'}"
                  f" в команде задачи: {action}")
            problems += 1
        logon = _pick(fields, "Logon Mode", "Режим входа в систему", "Режим входа")
        if "nteractive only" in logon or "олько интерактив" in logon:
            print(f"         [!] режим входа «{logon}»: задача пойдёт только при "
                  f"вошедшем пользователе — переустановите --install")
            problems += 1
    return problems


def journal_status(clinic: str) -> int:
    """Что о прогонах говорит база. Планировщик знает, что ЗАПУСКАЛ, а не что вышло."""
    from ews import runlog  # noqa: PLC0415
    from ews.config import ConfigError, load  # noqa: PLC0415

    try:
        cfg = load(clinic)
    except ConfigError as exc:
        print(f"  [!] конфигурация не загрузилась: {exc}")
        return 1
    if not cfg.gate("teable"):
        print("  enabled.teable=false — журнал прогонов не ведётся")
        return 0
    try:
        rows = runlog.recent(cfg, limit=5)
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] журнал недоступен: {exc.__class__.__name__}: {exc}")
        return 1
    if not rows:
        print("  журнал пуст — ни одного прогона ещё не записано")
        return 1

    for fields in rows:
        print(f"  {str(fields.get('Прогон') or '—'):16s} "
              f"{str(fields.get('Запуск') or '—'):14s} "
              f"{str(fields.get('Итог') or '—'):13s} "
              f"оценено {fields.get('Циклов оценено') or 0:>3} · "
              f"watch {fields.get('watch') or 0} · alert {fields.get('alert') or 0}")

    behind = runlog.missing_since(rows, days=STALE_AFTER_DAYS)
    if behind:
        print(f"  [!] {behind}")
        return 1
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
    parser.add_argument("--time", default=RUN_TIME,
                        help="время прогона, одно на обе задачи")
    args = parser.parse_args()

    if args.install:
        return install(args.clinic, args.time)
    if args.remove:
        return remove(args.clinic)

    print("Планировщик:")
    problems = status(args.clinic)
    print("\nЖурнал прогонов (Teable):")
    problems += journal_status(args.clinic)
    print()
    if problems:
        print(f"НЕ В ПОРЯДКЕ: {problems} замечаний выше")
        return 1
    print("расписание и журнал в порядке")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
