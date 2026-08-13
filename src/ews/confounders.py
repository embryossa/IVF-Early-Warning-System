"""Confounders окна культивирования: среда инкубатора и лоты расходников.

Это не причины и не отдельные тревоги, а то, что карточка обязана показать рядом с
сработавшей картой: что ещё происходило в окне `Date OPU..+N` этого цикла. Оба
источника переиспользуются из `embryo-lab-data` целиком — своя реализация разошлась
бы с ней при первом же изменении правил.

Два неподвижных ограничения, перенесённые оттуда без смягчения:

* **Пустое поле среды не означает «нарушений не было».** Недельная выгрузка приходит
  с задержкой, и окно цикла может быть ею не покрыто. Отсюда `miri_status`:
  `pending` — логов пока нет, `unavailable` — окно уже не будет покрыто,
  `enriched` — разобрано.
* **Смена лота — кандидат-confounder, не причина и не оценка качества лота.** Событие
  возникает только когда внутри окна реально пересекается более одного периода лота.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .clinic import Clinic
from .config import Config

STATUS_PENDING = "pending"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ENRICHED = "enriched"
STATUS_DISABLED = "disabled"


@dataclass
class WindowContext:
    """Что происходило в окне культивирования одного цикла."""
    cycle_uid: str
    miri_status: str = STATUS_DISABLED
    miri_events: list = field(default_factory=list)
    lot_events: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    # Нагрузка средой за окно. Именно минуты, а не число эпизодов: распределение
    # по минутам скошено в 19 раз (медиана 103, верхний дециль 1916), а по числу
    # эпизодов — почти нет (22 против 33). Считать надо ту величину, которая
    # различает циклы.
    @property
    def critical_minutes(self) -> float:
        total = 0.0
        for event in self.miri_events:
            if event.get("severity") != "critical":
                continue
            try:
                total += float(event.get("duration_minutes") or 0.0)
            except (TypeError, ValueError):
                continue
        return total

    @property
    def lot_changes(self) -> int:
        return sum(1 for e in self.lot_events if e.get("event_type") == "lot_change")

    def is_unusual(self, thresholds: dict) -> bool:
        """Необычна ли обстановка окна НА ФОНЕ ДРУГИХ ЦИКЛОВ.

        Первая версия считала подтверждением сам факт критического эпизода — и
        срабатывала у 49 циклов из 49. Источник, присутствующий всегда, не
        подтверждает ничего: быстрый маршрут watch -> alert повышал бы уровень
        каждой тревоге. Причина не в разборе логов: за шесть суток на шести камерах
        открытие крышки — рутинная работа эмбриолога, а экскурсия O2 после него —
        физика. Поэтому подтверждение сравнивается с распределением по когорте, как
        и всё остальное в этой системе (docs/METHODOLOGY.md §6.1).
        """
        minutes_limit = thresholds.get("critical_minutes")
        lots_limit = thresholds.get("lot_changes")
        by_environment = (minutes_limit is not None
                          and self.critical_minutes > minutes_limit)
        by_lots = lots_limit is not None and self.lot_changes > lots_limit
        return bool(by_environment or by_lots)


def _log_coverage(base_dir: Path, incubator_id: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Период, реально покрытый скачанными логами. Нужен, чтобы отличить
    «логов ещё нет» от «окно не будет покрыто никогда»."""
    import re

    root = base_dir / str(incubator_id)
    root = root if root.exists() else base_dir
    dates = []
    for path in root.glob("**/*.csv"):
        match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", path.name)
        if match:
            dates.append(pd.Timestamp(f"{match.group(1)}-{match.group(2)}-{match.group(3)}"))
    return (min(dates), max(dates)) if dates else None


BASELINE_FILE = "confounder_baseline.json"


def baseline_path(cfg: Config, clinic: Clinic) -> Path:
    return clinic.data_dir(cfg) / "state" / BASELINE_FILE


def save_baseline(cfg: Config, clinic: Clinic, thresholds: dict[str, float],
                  n_cycles: int) -> None:
    """Сохранить границы «необычного», чтобы суточный прогон не пересчитывал их.

    Разбор логов всей когорты занимает минуты: недельные выгрузки перекрываются, и
    даже с кэшем это десятки файлов на цикл. Суточному прогону нужны confounders
    только у сработавших циклов — а чтобы понять, необычны ли они, достаточно
    границ, посчитанных последним недельным проходом.
    """
    import json
    from datetime import date

    path = baseline_path(cfg, clinic)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "computed_at": date.today().isoformat(),
        "n_cycles": n_cycles,
        "quantile": 0.9,
        "thresholds": thresholds,
        "note": "верхний дециль нагрузки средой и сменами лотов по когорте; "
                "абсолютных порогов быть не может — они зависят от того, как "
                "лаборатория работает",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_baseline(cfg: Config, clinic: Clinic) -> tuple[dict[str, float], str]:
    """Границы и пояснение к ним. Пустые границы -> подтверждение недоступно."""
    import json

    path = baseline_path(cfg, clinic)
    if not path.exists():
        return {}, ("базы для сравнения нет — подтверждение по среде и лотам "
                    "недоступно, пока не отработает недельный проход")
    data = json.loads(path.read_text(encoding="utf-8"))
    return (data.get("thresholds") or {},
            f"границы от {data.get('computed_at')} по {data.get('n_cycles')} циклам")


def cohort_thresholds(contexts: dict[str, WindowContext],
                      quantile: float = 0.9) -> dict[str, float]:
    """Границы «необычного» по когорте: верхний дециль нагрузки средой и лотами.

    Абсолютных порогов здесь быть не может — они зависят от того, как лаборатория
    работает, сколько раз в сутки открывают инкубатор и как часто меняют лоты.
    Пересчитываются на каждом прогоне вместе с когортой.
    """
    minutes = sorted(c.critical_minutes for c in contexts.values() if c.miri_events)
    changes = sorted(c.lot_changes for c in contexts.values() if c.lot_events)
    out: dict[str, float] = {}
    if len(minutes) >= 10:
        out["critical_minutes"] = float(pd.Series(minutes).quantile(quantile))
    if len(changes) >= 10:
        out["lot_changes"] = float(pd.Series(changes).quantile(quantile))
    return out


def collect(cfg: Config, clinic: Clinic, frame: pd.DataFrame) -> dict[str, WindowContext]:
    """Собрать контекст окна для каждого закрытого цикла."""
    window_days = int(cfg.settings["soft_steps"]["miri"]["cultivation_window_days"])
    contexts = {uid: WindowContext(cycle_uid=uid) for uid in frame["cycle_uid"]}

    _collect_miri(cfg, clinic, frame, contexts, window_days)
    _collect_lots(cfg, clinic, frame, contexts, window_days)
    return contexts


def _collect_miri(cfg: Config, clinic: Clinic, frame: pd.DataFrame,
                  contexts: dict[str, WindowContext], window_days: int) -> None:
    if not (clinic.source_enabled("miri")
            and cfg.settings["soft_steps"]["miri"].get("enabled")):
        return
    section = cfg.machine.get("miri_logs") or {}
    base_dir = str(section.get("dump_dir") or "").strip()
    if not base_dir or not Path(base_dir).exists():
        for context in contexts.values():
            context.miri_status = STATUS_PENDING
        return

    from .vendor import incubator  # noqa: PLC0415

    base = Path(base_dir)
    incubator_id = str(section.get("incubator") or "6391")
    coverage = _log_coverage(base, incubator_id)
    cache: dict[tuple, pd.DataFrame] = {}

    for _, row in frame.iterrows():
        context = contexts[row["cycle_uid"]]
        date_opu = row.get("date_opu")
        if pd.isna(date_opu):
            context.miri_status = STATUS_UNAVAILABLE
            continue
        start = pd.Timestamp(date_opu)
        end = start + timedelta(days=window_days)

        if coverage is None or start < coverage[0] or end > coverage[1]:
            # Окно вне покрытия выгрузок. Если оно в прошлом относительно
            # последнего лога — покрытия уже не будет; если в будущем — ждём.
            context.miri_status = (STATUS_PENDING if coverage and end > coverage[1]
                                   else STATUS_UNAVAILABLE)
            context.notes.append(
                f"окно {start:%Y-%m-%d}..{end:%Y-%m-%d} вне покрытия логов "
                f"({coverage[0]:%Y-%m-%d}..{coverage[1]:%Y-%m-%d})" if coverage
                else "логов нет вовсе")
            continue

        try:
            files = incubator.find_log_files(str(base), incubator_id, start, end)
            if not files:
                context.miri_status = STATUS_UNAVAILABLE
                continue
            # Кэш разобранных файлов. Недельные выгрузки перекрываются между
            # соседними циклами, и без кэша один и тот же CSV читается десятки раз:
            # на 87 циклах это превращало разбор из секунд в минуты. В суточном
            # режиме разница не видна (1-3 новых цикла), при пересборе истории —
            # решающая.
            key = tuple(sorted(str(f) for f in files))
            logs = cache.get(key)
            if logs is None:
                logs = incubator.load_files(files, sep=str(section.get("sep", ";")))
                cache[key] = logs
            _, events = incubator.analyze_window_events(
                logs, start, days=window_days,
                setpoints=section.get("setpoints") or {},
                tolerances=section.get("tolerances") or {},
                cycle_uid=row["cycle_uid"],
            )
        except Exception as exc:  # noqa: BLE001 — мягкий шаг
            context.miri_status = STATUS_PENDING
            context.notes.append(f"разбор логов не удался: {exc}")
            continue
        context.miri_status = STATUS_ENRICHED
        context.miri_events = (events.to_dict("records")
                               if isinstance(events, pd.DataFrame) and not events.empty
                               else [])


def _collect_lots(cfg: Config, clinic: Clinic, frame: pd.DataFrame,
                  contexts: dict[str, WindowContext], window_days: int) -> None:
    if not (clinic.source_enabled("lots") and cfg.gate("airtable")):
        return
    airtable = cfg.machine.get("airtable") or {}
    try:
        from .vendor import lots  # noqa: PLC0415

        # Конфиг в том же виде, что ждёт соседний проект: свою обёртку не пишем,
        # чтобы правила сбора лотов не разошлись между проектами.
        items, changes = lots.fetch_consumable_lots({"consumable_lots": {"airtable": {
            "token": cfg.secret("airtable"),
            "base_id": str(airtable.get("base_id") or ""),
            "items_table_id": str(airtable.get("items_table_id") or ""),
            "changes_table_id": str(airtable.get("changes_table_id") or ""),
        }}})
        timeline = lots.build_lot_timeline(items, changes)
    except Exception as exc:  # noqa: BLE001 — мягкий шаг
        for context in contexts.values():
            context.notes.append(f"лоты недоступны: {exc}")
        return

    for _, row in frame.iterrows():
        context = contexts[row["cycle_uid"]]
        date_opu = row.get("date_opu")
        if pd.isna(date_opu):
            continue
        start = pd.Timestamp(date_opu)
        try:
            _, events = lots.analyze_window_events(timeline, start, days=window_days,
                                                   cycle_uid=row["cycle_uid"])
        except Exception as exc:  # noqa: BLE001
            context.notes.append(f"разбор лотов не удался: {exc}")
            continue
        context.lot_events = (events.to_dict("records")
                              if isinstance(events, pd.DataFrame) and not events.empty
                              else [])
