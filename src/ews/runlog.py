"""Журнал прогонов в Teable: жив ли конвейер, видно из базы.

Зачем это в базе, а не только в `state/daily_last_run.json`. Локальный JSON
отвечает на вопрос «как прошёл последний прогон» и только сидящему за этой
машиной. Он не отвечает на вопрос, который на самом деле задают: **работает ли
система вообще**. Задача планировщика, которая не сработала, ничего не пишет — ни
в JSON, ни куда-либо ещё, — и её молчание неотличимо от «сегодня нечего было
делать». Запись в базе делает отсутствие записи заметным: за каждый рабочий день
в журнале должна стоять строка «по расписанию».

Запись — **мягкий шаг по построению**. Прогон уже состоялся к моменту, когда сюда
дошло управление; уронить его из-за того, что не записался журнал, было бы ровно
тем обменом, который делать нельзя. Любая ошибка возвращается текстом и печатается
вызывающим кодом, но кода возврата не меняет.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import socket
from typing import Any

from .config import Config

TABLE_KEY = "Runs"

RESULT_OK = "успех"
RESULT_SOFT = "с пропусками"
RESULT_FAILED = "сбой"

TRIGGER_SCHEDULE = "по расписанию"
TRIGGER_MANUAL = "вручную"

MODE_DAILY = "суточный"
MODE_WEEKLY = "недельный"


def result_of(failed: list, skipped: list) -> str:
    """Итог цепочки одним словом.

    «с пропусками» — не синоним успеха: мягкий шаг пропущен, значит чего-то из
    обстановки в этом прогоне не было. Слипание этих двух исходов в один сделало
    бы журнал бесполезным именно там, где он нужен.
    """
    if failed:
        return RESULT_FAILED
    return RESULT_SOFT if skipped else RESULT_OK


def build_record(cfg: Config, *, run_id: str, started_at: datetime,
                 weekly: bool, triggered_by: str, results: list[dict],
                 manifest: dict | None, notes: list[str]) -> dict:
    """Строка журнала из того, что уже посчитано цепочкой и прогоном."""
    failed = [r for r in results if not r["ok"] and not r["soft"]]
    skipped = [r for r in results if not r["ok"] and r["soft"]]
    levels = (manifest or {}).get("levels") or {}

    lines = list(notes)
    for item in skipped:
        lines.append(f"мягко пропущен: {item['step']} (код {item['returncode']})")
    for item in failed:
        lines.append(f"СБОЙ: {item['step']} (код {item['returncode']})")

    return {
        "Прогон": run_id,
        "Начат": started_at.isoformat(timespec="seconds"),
        "Режим": MODE_WEEKLY if weekly else MODE_DAILY,
        "Запуск": triggered_by,
        "Итог": result_of(failed, skipped),
        "Длительность, с": round(sum(r["seconds"] for r in results), 1),
        "Шагов успешно": sum(1 for r in results if r["ok"]),
        "Мягко пропущено": len(skipped),
        "Циклов оценено": int((manifest or {}).get("cycles_evaluated") or 0),
        "watch": int(levels.get("watch") or 0),
        "alert": int(levels.get("alert") or 0),
        # Гейт пишется как есть: 'open' либо причина блокировки. Читающий журнал
        # обязан видеть, мог ли этот прогон вообще поднять alert.
        "Гейт alert": str((manifest or {}).get("alert_gate") or "—"),
        "Площадка": f"{cfg.site}@{socket.gethostname()}",
        "Окружение": cfg.environment,
        "Замечания": "\n".join(lines),
    }


def _table_id(cfg: Config, client: Any) -> str:
    from . import teable_client  # noqa: PLC0415

    base_id = str(cfg.external("teable").get("base_id") or "").strip()
    if not base_id:
        raise RuntimeError("teable.base_id пуст — выполните ews_bootstrap_opu.py")
    tables = teable_client.resolve_tables(cfg, client, base_id)
    if TABLE_KEY not in tables:
        raise RuntimeError("в базе нет таблицы Runs — выполните ews_bootstrap_opu.py "
                           "(идемпотентен, создаст только недостающее)")
    return tables[TABLE_KEY]


def write(cfg: Config, record: dict) -> tuple[bool, str]:
    """Записать строку журнала. Возвращает (получилось, пояснение)."""
    if not cfg.gate("teable"):
        return False, "enabled.teable=false — журнал прогонов не ведётся"
    try:
        from . import teable_client  # noqa: PLC0415

        client = teable_client.connect(cfg)
        client.create_records(_table_id(cfg, client), [record])
    except Exception as exc:  # noqa: BLE001 — мягкий по построению
        return False, f"{exc.__class__.__name__}: {exc}"
    return True, "записан"


def recent(cfg: Config, limit: int = 5) -> list[dict]:
    """Последние прогоны из журнала, свежие первыми.

    Сортировка по `Прогон`: run id — это метка времени `YYYYmmddTHHMMSS`, поэтому
    лексикографический порядок совпадает с хронологическим, и отдельного поля для
    сортировки не нужно.
    """
    from . import teable_client  # noqa: PLC0415

    client = teable_client.connect(cfg)
    records = client.all_records(_table_id(cfg, client))
    rows = [r.get("fields", {}) for r in records]
    rows.sort(key=lambda f: str(f.get("Прогон") or ""), reverse=True)
    return rows[:limit]


def missing_since(rows: list[dict], *, days: int = 1,
                  now: datetime | None = None) -> str | None:
    """Не пропущен ли плановый прогон.

    Смотрит только на запуски `по расписанию`: ручные прогоны не доказывают, что
    задача планировщика жива, — а именно это здесь и проверяется.
    """
    now = now or datetime.now()
    scheduled = [f for f in rows if str(f.get("Запуск") or "") == TRIGGER_SCHEDULE]
    if not scheduled:
        return "прогонов по расписанию в журнале нет вовсе"
    stamp = str(scheduled[0].get("Прогон") or "")
    try:
        last = datetime.strptime(stamp[:15], "%Y%m%dT%H%M%S")
    except ValueError:
        return f"последний плановый прогон не разобран: {stamp!r}"
    behind = now - last
    if behind > timedelta(days=days, hours=2):
        return (f"последний прогон по расписанию был {last:%Y-%m-%d %H:%M} — "
                f"{behind.days} дн. назад")
    return None
