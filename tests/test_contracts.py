"""Стыки между слоями: имена, по которым они друг друга находят.

Все проверки здесь — про ОДИН класс ошибки. Слои общаются строковыми ключами:
конвейер пишет в Teable поле «Режим» значением «суточный», читает из листа колонку
`Bl`, кладёт в карточку тревоги колонку `corroboration_rank`. Опечатка или
переименование с одной стороны не даёт ни исключения, ни предупреждения: поле молча
не записывается, метрика молча пропадает, колонка молча становится пустой.

Отдельно проверяется, что настоящий `config/settings.yaml` содержит все ключи,
которые код индексирует ЖЁСТКО. Такой ключ, исчезнув, роняет прогон посреди работы —
уже после того, как задача планировщика отчиталась о запуске.
"""
from datetime import datetime

import pandas as pd
import pytest

from conftest import load_script

from ews import cycles, envelope, envelope_qc, evaluate, opu_schema, runlog, sync_opu


run = load_script("ews_run")

SCHEMA = {table["key"]: {field["name"] for field in table["fields"]}
          for table in opu_schema.schema()}


def _choices(spec) -> set[str]:
    return {choice["name"] for choice in spec}


# --- Журнал прогонов ------------------------------------------------------

def test_every_journal_field_exists_in_the_table(cfg):
    """Строка журнала не содержит полей, которых нет в таблице.

    Лишнее поле Teable не создаёт и об этом не сообщает — запись просто теряет
    часть содержимого, а журнал существует именно для того, чтобы по нему судить о
    работе системы.
    """
    record = runlog.build_record(
        cfg, run_id="20260903T154500", started_at=datetime(2026, 9, 3, 15, 45),
        weekly=False, triggered_by=runlog.TRIGGER_SCHEDULE,
        results=[{"step": "прогон", "returncode": 0, "ok": True, "soft": False,
                  "seconds": 1.0}],
        manifest={"cycles_evaluated": 1, "levels": {"ok": 1}, "alert_gate": "open"},
        notes=[])
    assert set(record) <= SCHEMA["Runs"], set(record) - SCHEMA["Runs"]


@pytest.mark.parametrize("value, field", [
    (runlog.MODE_DAILY, "RUN_MODE_CHOICES"),
    (runlog.MODE_WEEKLY, "RUN_MODE_CHOICES"),
    (runlog.TRIGGER_SCHEDULE, "RUN_TRIGGER_CHOICES"),
    (runlog.TRIGGER_MANUAL, "RUN_TRIGGER_CHOICES"),
    (runlog.RESULT_OK, "RUN_RESULT_CHOICES"),
    (runlog.RESULT_SOFT, "RUN_RESULT_CHOICES"),
    (runlog.RESULT_FAILED, "RUN_RESULT_CHOICES"),
])
def test_journal_values_are_known_to_the_single_select(value, field):
    """Значение, которого нет в списке singleSelect, Teable не примет.

    Отказ придёт из сети в самом конце цепочки — когда прогон уже состоялся и
    единственное свидетельство о нём как раз и не записалось.
    """
    assert value in _choices(getattr(opu_schema, field))


# --- Журнал синхронизаций -------------------------------------------------

def test_sync_report_matches_its_table():
    report = sync_opu.SyncReport(run_id="r", started_at=datetime(2026, 9, 3),
                                 source="лист", environment="production")
    assert set(report.as_record()) <= SCHEMA["Sync"]


def test_opu_row_matches_its_table():
    """Все поля строки цикла объявлены в схеме таблицы OPU."""
    fields = sync_opu.row_to_fields(pd.Series({"ID": "A1", "Date OPU": "01.06.2026"}),
                                    cycle_uid="uid", environment="production")
    assert set(fields) <= SCHEMA["OPU"], set(fields) - SCHEMA["OPU"]


def test_checkpoint_values_are_known_to_the_single_select():
    known = _choices(opu_schema.CHECKPOINT_CHOICES)
    for row in ({}, {"Дата OPU": "2026-06-01"},
                {"Дата OPU": "2026-06-01", "2pN": 8},
                {"Дата OPU": "2026-06-01", "Бласт": 5}):
        assert sync_opu.checkpoint_of(row) in known


def test_environment_values_are_known_to_the_single_select():
    from ews.config import DRAFT, PRODUCTION

    assert {DRAFT, PRODUCTION} <= _choices(opu_schema.ENV_CHOICES)


# --- Лист -> Teable -> канонический кадр ----------------------------------

def test_canonical_map_reads_only_fields_that_exist():
    """Кадр циклов читает из Teable ровно те поля, которые туда пишет синхронизация.

    Односторонняя карта листа рвётся тихо: неизвестное имя даёт колонку пропусков,
    а метрика на ней — «наблюдений нет», что неотличимо от честного отсутствия
    данных.
    """
    written = set(opu_schema.SHEET_TO_FIELD.values())
    read = set(cycles.TEABLE_TO_CANONICAL)
    assert read <= written, read - written


def test_numeric_canonical_names_are_all_declared():
    assert cycles.NUMERIC_CANONICAL <= set(cycles.TEABLE_TO_CANONICAL.values())


def test_every_metric_part_has_a_source_column():
    """У каждой метрики есть, откуда взять числитель и знаменатель."""
    known = set(cycles.TEABLE_TO_CANONICAL.values()) | set(cycles.DERIVED_COLUMNS)
    for metric, parts in cycles.METRIC_PARTS.items():
        assert set(parts) <= known, (metric, set(parts) - known)


def test_funnel_fields_match_the_checkpoint_columns(clinic):
    """Поля, двигающие цикл по чекпойнтам, совпадают в зеркале и в кадре.

    Синхронизация считает чекпойнт по русским именам полей Teable, кадр — по
    каноническим. Разойдись они, цикл в базе значился бы закрытым, а в оценку не
    попадал бы — и наоборот.
    """
    to_field = opu_schema.SHEET_TO_FIELD
    canonical_to_sheet = clinic.columns["canonical"]

    for checkpoint, funnel in (("fertilisation", opu_schema.FUNNEL_T1),
                               ("closure", opu_schema.FUNNEL_T6)):
        expected = {to_field[canonical_to_sheet[name]]
                    for name in clinic.checkpoint_fields(checkpoint)}
        assert set(funnel) == expected, (checkpoint, set(funnel) ^ expected)


# --- Конверт --------------------------------------------------------------

def test_every_risk_adjusted_stage_is_actually_registered():
    """Стадия, которой карта ждёт ожидания, обязана где-то регистрироваться."""
    registered = {stage for stages in envelope.STAGES_AT.values() for stage in stages}
    assert set(evaluate.ENVELOPE_STAGE.values()) <= registered


def test_qc_covers_exactly_the_registered_stages():
    registered = {stage for stages in envelope.STAGES_AT.values() for stage in stages}
    assert set(envelope_qc.STAGE_ACTUAL) == registered
    assert set(envelope_qc.STAGE_BASELINE_KEY) == registered


def test_every_qc_stage_has_a_measured_baseline(cfg):
    """Покрытие сравнивается с ИЗМЕРЕННЫМ опорным, а не с номинальными 0.80.

    У стадий оно разное (0.759 у 2PN, 0.823 у бластоцист), и сравнение с номиналом
    давало бы постоянный ложный сигнал на одной стадии и слепоту на другой.
    """
    measured = set(cfg.settings["envelope"]["baseline_coverage"])
    assert set(envelope_qc.STAGE_BASELINE_KEY.values()) <= measured


def test_qc_actual_columns_exist_in_the_frame():
    assert set(envelope_qc.STAGE_ACTUAL.values()) <= set(
        cycles.TEABLE_TO_CANONICAL.values())


def test_good_and_frozen_are_registered_nowhere():
    """`good` и `frozen` обусловлены числом бластоцист — той величиной, которую мы
    и мониторим. Конвертом они быть не могут ни на одном чекпойнте."""
    registered = {stage for stages in envelope.STAGES_AT.values() for stage in stages}
    assert "good" not in registered and "frozen" not in registered


# --- Карточка тревоги -----------------------------------------------------

REQUIRED_BY_CONTRACT = {"alert_id", "cycle_uid", "raised_at", "level", "chart",
                        "contour", "corroboration", "corroboration_rank",
                        "miri_status", "environment"}


def test_alert_card_carries_every_contracted_field():
    assert REQUIRED_BY_CONTRACT <= set(run.ALERT_COLUMNS)


def test_alert_card_carries_no_phi():
    """В карточке тревоги нет ни одного идентификатора пациентки.

    Разрешён `cycle_uid` и аналитические признаки; всё остальное — нарушение
    границы приватности, наследуемой без послаблений.
    """
    forbidden = {"patient_id", "patient_name", "dob", "ID", "_key", "_join"}
    assert not (set(run.ALERT_COLUMNS) & forbidden)


# --- Настройки метода -----------------------------------------------------

HARD_KEYS = [
    ("checkpoints", "t6", "offset_days"),
    ("charts", "cusum", "odds_ratio_deterioration"),
    ("charts", "cusum", "odds_ratio_upper_control"),
    ("localisation", "contours"),
    ("thresholds", "target_false_alarms_per_year", "watch"),
    ("thresholds", "target_false_alarms_per_year", "alert"),
    ("soft_steps", "miri", "cultivation_window_days"),
    ("soft_steps", "miri", "enabled"),
    ("envelope", "min_interval_width"),
    ("envelope", "iqr_to_sigma_divisor"),
    ("envelope", "baseline_coverage"),
    ("envelope", "risk_adjustment_enabled"),
]


@pytest.mark.parametrize("path", HARD_KEYS, ids=lambda p: ".".join(p))
def test_settings_contain_every_hard_indexed_key(shipped_settings, path):
    """Ключ, который код берёт без `.get`, обязан существовать в settings.yaml.

    Его исчезновение роняет прогон посреди работы — после того, как задача
    планировщика уже отчиталась о запуске, и до того, как что-либо записано.
    """
    node = shipped_settings
    for key in path:
        assert key in node, f"нет {'.'.join(path)}"
        node = node[key]


def test_every_contour_metric_is_a_known_metric(shipped_settings):
    """Контуры перечисляют метрики, которые система умеет считать."""
    for contour, metrics in shipped_settings["localisation"]["contours"].items():
        unknown = set(metrics) - set(cycles.METRIC_PARTS)
        assert not unknown, (contour, unknown)


def test_no_metric_belongs_to_two_contours(shipped_settings):
    """Локализация не имеет смысла, если метрика числится в двух контурах сразу."""
    seen: set[str] = set()
    for metrics in shipped_settings["localisation"]["contours"].values():
        assert not (seen & set(metrics))
        seen |= set(metrics)


def test_targets_match_the_declared_decision(shipped_settings):
    """Целевые частоты — решение владельца, а не подобранное число.

    alert — не чаще одной ложной на 6 месяцев по системе; watch — не чаще двух в
    месяц. Изменение этих чисел меняет смысл всех порогов сразу.
    """
    targets = shipped_settings["thresholds"]["target_false_alarms_per_year"]
    assert targets["watch"] == 24 and targets["alert"] == 2
