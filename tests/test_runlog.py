"""Журнал прогонов: итог цепочки и обнаружение пропущенного расписания."""
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import runlog
from ews.config import Config


def _cfg():
    return Config(settings={}, machine={"environment": "production", "site": "lab"},
                  secrets={}, clinic_code="two_line")


def _step(name, ok, soft, seconds=1.0):
    return {"step": name, "returncode": 0 if ok else 1, "ok": ok, "soft": soft,
            "seconds": seconds}


def test_soft_skip_is_not_success():
    """«с пропусками» обязан отличаться от «успеха».

    Мягко пропущенный шаг означает, что чего-то из обстановки в прогоне не было.
    Слипание с успехом сделало бы журнал слепым именно там, где он нужен.
    """
    assert runlog.result_of([], []) == runlog.RESULT_OK
    assert runlog.result_of([], [_step("среда", False, True)]) == runlog.RESULT_SOFT
    assert runlog.result_of([_step("прогон", False, False)], []) == runlog.RESULT_FAILED


def test_hard_failure_wins_over_soft_skip():
    failed = [_step("прогон", False, False)]
    skipped = [_step("экспорт", False, True)]
    assert runlog.result_of(failed, skipped) == runlog.RESULT_FAILED


def test_record_carries_levels_and_gate():
    record = runlog.build_record(
        _cfg(), run_id="20260814T154500", started_at=datetime(2026, 8, 14, 15, 45),
        weekly=False, triggered_by=runlog.TRIGGER_SCHEDULE,
        results=[_step("прогон", True, False, 12.0)],
        manifest={"cycles_evaluated": 3, "levels": {"ok": 1, "watch": 2},
                  "alert_gate": "open"},
        notes=[])
    assert record["Итог"] == runlog.RESULT_OK
    assert record["Циклов оценено"] == 3
    assert record["watch"] == 2
    assert record["alert"] == 0          # уровня нет -> 0, а не пропуск
    assert record["Гейт alert"] == "open"
    assert record["Режим"] == runlog.MODE_DAILY
    assert record["Окружение"] == "production"
    assert record["Площадка"].startswith("lab@")


def test_record_without_manifest_says_so():
    """Цепочка, не дошедшая до оценки, обязана быть отличима от нулевого улова."""
    record = runlog.build_record(
        _cfg(), run_id="20260814T154500", started_at=datetime(2026, 8, 14, 15, 45),
        weekly=False, triggered_by=runlog.TRIGGER_SCHEDULE,
        results=[_step("прогон", False, False)], manifest=None,
        notes=["манифест прогона не прочитан"])
    assert record["Итог"] == runlog.RESULT_FAILED
    assert record["Циклов оценено"] == 0
    assert "манифест" in record["Замечания"]
    assert "СБОЙ: прогон" in record["Замечания"]


def test_manual_runs_do_not_prove_schedule_alive():
    """Ручной прогон не доказывает, что задача планировщика работает."""
    rows = [{"Прогон": "20260814T110000", "Запуск": runlog.TRIGGER_MANUAL}]
    assert runlog.missing_since(rows, now=datetime(2026, 8, 14, 12, 0)) is not None


def test_fresh_scheduled_run_is_not_flagged():
    rows = [{"Прогон": "20260814T154500", "Запуск": runlog.TRIGGER_SCHEDULE}]
    assert runlog.missing_since(rows, now=datetime(2026, 8, 15, 12, 0)) is None


def test_stale_scheduled_run_is_flagged():
    rows = [{"Прогон": "20260810T154500", "Запуск": runlog.TRIGGER_SCHEDULE}]
    note = runlog.missing_since(rows, now=datetime(2026, 8, 14, 16, 0))
    assert note and "2026-08-10" in note


# Допуск суточного расписания — ews_schedule.STALE_AFTER_DAYS.
DAILY_TOLERANCE = 2


def test_daily_tolerance_survives_a_caught_up_run():
    """Прогон, догнанный `StartWhenAvailable` на следующий день, — не пропуск.

    Ровно суточный допуск объявлял бы пропуском каждый перенос из-за выключенной
    машины, и `--status` ругался бы там, где расписание отработало.
    """
    rows = [{"Прогон": "20260810T154500", "Запуск": runlog.TRIGGER_SCHEDULE}]
    caught_up = datetime(2026, 8, 11, 20, 0)
    assert runlog.missing_since(rows, days=DAILY_TOLERANCE, now=caught_up) is None


def test_daily_tolerance_catches_a_missed_day():
    """Суточное расписание: два дня молчания обязаны быть видны.

    Прежний недельный допуск в 8 суток пропустил бы этот ряд целиком — при суточной
    каденции он означал бы неделю без оценки.
    """
    rows = [{"Прогон": "20260810T154500", "Запуск": runlog.TRIGGER_SCHEDULE}]
    now = datetime(2026, 8, 13, 16, 0)
    assert runlog.missing_since(rows, days=DAILY_TOLERANCE, now=now) is not None
    assert runlog.missing_since(rows, days=8, now=now) is None


def test_empty_journal_is_flagged():
    assert runlog.missing_since([]) is not None
