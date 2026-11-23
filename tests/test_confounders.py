"""Обстановка окна культивирования: среда MIRI и лоты расходников.

Два неподвижных ограничения, и оба здесь проверяются:

* **Пустое поле среды не означает «нарушений не было».** Выгрузка приходит раз в
  7 дней и может не покрывать окно цикла — отсюда три различимых статуса.
* **Необычность считается ОТНОСИТЕЛЬНО когорты.** Первая версия считала
  подтверждением сам факт критического эпизода и срабатывала у 49 циклов из 49:
  за шесть суток на шести камерах открытие крышки — рутина, а экскурсия O2 после
  него — физика. Источник, присутствующий всегда, не подтверждает ничего.
"""
import json

import pytest

from ews import confounders


def _context(uid="A", minutes=(), lots=()) -> confounders.WindowContext:
    context = confounders.WindowContext(cycle_uid=uid)
    context.miri_events = [{"severity": "critical", "duration_minutes": m}
                           for m in minutes]
    context.lot_events = [{"event_type": t} for t in lots]
    return context


# --- Нагрузка окна ---------------------------------------------------------

def test_only_critical_episodes_count_toward_minutes():
    """Считается объём КРИТИЧЕСКОЙ нагрузки, а не всё подряд."""
    context = confounders.WindowContext(cycle_uid="A")
    context.miri_events = [
        {"severity": "critical", "duration_minutes": 100},
        {"severity": "warning", "duration_minutes": 900},
        {"severity": "critical", "duration_minutes": 50},
    ]
    assert context.critical_minutes == 150


def test_unparsable_duration_is_skipped_not_guessed():
    """Испорченное значение длительности не превращается в ноль минут молча —
    оно просто не участвует, а остальные события считаются как обычно."""
    context = confounders.WindowContext(cycle_uid="A")
    context.miri_events = [{"severity": "critical", "duration_minutes": "долго"},
                           {"severity": "critical", "duration_minutes": 30}]
    assert context.critical_minutes == 30


def test_missing_duration_is_zero_minutes_not_a_crash():
    context = confounders.WindowContext(cycle_uid="A")
    context.miri_events = [{"severity": "critical"}]
    assert context.critical_minutes == 0.0


def test_only_real_lot_changes_are_counted():
    """Событие lot_change возникает только при реальном пересечении периодов лота.

    Всё остальное в списке событий сменой лота не является — это правило проекта.
    """
    context = _context(lots=("lot_change", "lot_open", "lot_change"))
    assert context.lot_changes == 2


# --- Необычность -----------------------------------------------------------

def test_without_thresholds_nothing_is_unusual():
    """Нет базы для сравнения — подтверждение недоступно, а не «всё обычно».

    Разница существенная: недоступное подтверждение не должно молча превращаться
    в отрицательный ответ и разрешать эскалацию.
    """
    assert _context(minutes=(5000,)).is_unusual({}) is False


def test_load_above_the_cohort_decile_is_unusual():
    context = _context(minutes=(2000,))
    assert context.is_unusual({"critical_minutes": 1900}) is True
    assert context.is_unusual({"critical_minutes": 2100}) is False


def test_lot_changes_above_the_decile_are_unusual():
    assert _context(lots=("lot_change", "lot_change")).is_unusual(
        {"lot_changes": 1}) is True


def test_either_source_is_enough():
    """Среда и лоты — независимые источники; необычность любого из них считается."""
    context = _context(minutes=(9999,), lots=("lot_change",))
    assert context.is_unusual({"critical_minutes": 100, "lot_changes": 5}) is True


# --- Границы по когорте ---------------------------------------------------

def test_cohort_thresholds_need_enough_cycles():
    """Меньше десяти циклов с разобранной обстановкой — границ нет.

    Верхний дециль по девяти наблюдениям — не дециль, а случайное число, и
    подтверждение по нему было бы подтверждением ничего.
    """
    few = {f"C{i}": _context(f"C{i}", minutes=(i * 100,)) for i in range(9)}
    assert confounders.cohort_thresholds(few) == {}


def test_cohort_thresholds_are_the_upper_decile():
    # Одиннадцать циклов по 0..1000 минут: девятый дециль приходится ровно на 900.
    contexts = {f"C{i}": _context(f"C{i}", minutes=(i * 100,)) for i in range(11)}
    thresholds = confounders.cohort_thresholds(contexts)
    assert thresholds["critical_minutes"] == pytest.approx(900.0)


def test_cycles_without_events_do_not_dilute_the_decile():
    """Цикл без разобранных логов в распределение нагрузки не входит.

    Иначе «нет выгрузки» считалось бы «нагрузки не было» и опускало бы границу.
    """
    contexts = {f"C{i}": _context(f"C{i}", minutes=(1000,)) for i in range(10)}
    contexts.update({f"E{i}": _context(f"E{i}") for i in range(50)})
    assert confounders.cohort_thresholds(contexts)["critical_minutes"] == \
        pytest.approx(1000.0)


# --- База границ на диске -------------------------------------------------

def test_baseline_round_trip(cfg, clinic):
    confounders.save_baseline(cfg, clinic, {"critical_minutes": 1916.0,
                                            "lot_changes": 1.0}, n_cycles=87)
    thresholds, note = confounders.load_baseline(cfg, clinic)
    assert thresholds["critical_minutes"] == 1916.0
    assert "87" in note
    payload = json.loads(confounders.baseline_path(cfg, clinic).read_text(
        encoding="utf-8"))
    assert payload["quantile"] == 0.9


def test_missing_baseline_explains_itself(cfg, clinic):
    """Отсутствие базы обязано объяснять, что делать, а не молчать."""
    thresholds, note = confounders.load_baseline(cfg, clinic)
    assert thresholds == {}
    assert "недельный проход" in note


def test_baseline_is_written_atomically(cfg, clinic):
    """Полуфабрикатов на диске не остаётся — правило контрактов артефактов."""
    confounders.save_baseline(cfg, clinic, {"critical_minutes": 1.0}, n_cycles=10)
    directory = confounders.baseline_path(cfg, clinic).parent
    assert not list(directory.glob("*.tmp"))


# --- Статусы среды --------------------------------------------------------

def test_status_values_match_the_contract(cfg):
    """Три статуса, и они те же, что в settings.yaml и в CONTRACTS.md."""
    declared = set(cfg.settings["soft_steps"]["miri"]["status_values"])
    assert declared == {confounders.STATUS_PENDING, confounders.STATUS_UNAVAILABLE,
                        confounders.STATUS_ENRICHED}


def test_default_status_is_disabled_not_enriched():
    """Умолчание — «источник выключен», а не «разобрано».

    Обратное означало бы, что цикл без единого обращения к логам выглядит как цикл
    с чистой средой.
    """
    assert confounders.WindowContext(cycle_uid="A").miri_status == \
        confounders.STATUS_DISABLED


def test_miri_is_skipped_when_the_source_is_off(cfg, clinic, tmp_path):
    """Выключенный в профиле источник не трогается вовсе — и это не ошибка."""
    import pandas as pd

    cfg.settings["soft_steps"]["miri"]["enabled"] = False
    cfg.settings["soft_steps"]["lots"]["enabled"] = False
    frame = pd.DataFrame([{"cycle_uid": "A", "date_opu": pd.Timestamp("2026-06-01")}])
    contexts = confounders.collect(cfg, clinic, frame)
    assert contexts["A"].miri_status == confounders.STATUS_DISABLED
