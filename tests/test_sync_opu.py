"""Синхронизация листа в Teable: то место, где данные входят в систему.

Направление одностороннее: лист — источник, Teable — зеркало. Проверяется главным
образом то, чего делать НЕЛЬЗЯ: подменять пропуск нулём, сливать дубли, удалять
строки по расхождению выгрузок и трогать запись, в которой ничего не изменилось.
Все четыре ошибки молчаливы — по числу строк в журнале их не видно.
"""
from datetime import datetime

import pandas as pd
import pytest

from conftest import FakeTeable

from ews import sync_opu


# --- Разбор ячеек ----------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("12", 12.0),
    ("2,5", 2.5),          # запятая как десятичный разделитель — норма
    ("1 200", 1200.0),     # неразрывный пробел из листа
    ("0", 0.0),            # ноль — известное наблюдение
    ("-3", -3.0),
])
def test_numbers_are_read_as_written(raw, expected):
    assert sync_opu._to_number(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, "н/д", "12a", "1.2.3"])
def test_non_numbers_become_missing_not_zero(raw):
    """Пустая ячейка — пропуск, а НЕ ноль.

    Ноль здесь известное наблюдение (например, 0 бластоцист), и подменять им
    пропуск запрещено правилами проекта: доля посчиталась бы по циклу, которого
    в знаменателе быть не должно.
    """
    assert sync_opu._to_number(raw) is None


@pytest.mark.parametrize("raw, expected", [
    ("01.06.2026", "2026-06-01"),
    ("2026-06-01", "2026-06-01"),
    ("01/06/2026", "2026-06-01"),
])
def test_dates_accept_the_formats_of_the_sheet(raw, expected):
    assert sync_opu._to_date(raw) == expected


def test_unknown_date_format_is_not_guessed():
    assert sync_opu._to_date("июнь 2026") is None
    assert sync_opu._to_date("") is None


# --- Ключ цикла ------------------------------------------------------------

def test_key_joins_patient_and_opu_date():
    """Ключ — ID#дата пункции. По одному ID нельзя: у повторных попыток одной
    пациентки он общий, и join схлопнул бы разные циклы в один."""
    row = pd.Series({"ID": "ab123", "Date OPU": "01.06.2026"})
    assert sync_opu.build_key(row) == "AB123#01.06.2026"


def test_key_uses_the_date_as_written():
    """Дата в ключе — как в листе, а не как её сегодня распарсили.

    Иначе ключ поехал бы вместе с любой сменой формата, и та же строка приехала бы
    в Teable второй раз как новая.
    """
    row = pd.Series({"ID": "AB123", "Date OPU": "2026-06-01"})
    assert sync_opu.build_key(row).endswith("#2026-06-01")


@pytest.mark.parametrize("row", [
    {"ID": "", "Date OPU": "01.06.2026"},
    {"ID": "AB123", "Date OPU": ""},
])
def test_row_without_key_parts_has_no_key(row):
    assert sync_opu.build_key(pd.Series(row)) == ""


# --- Чекпойнт зеркала ------------------------------------------------------

def _fields(**values) -> dict:
    base = {"Дата OPU": "2026-06-01"}
    base.update(values)
    return base


def test_checkpoint_needs_both_data_and_calendar():
    early = sync_opu.checkpoint_of(_fields(**{"Бласт": 5, "2pN": 8}),
                                   today=datetime(2026, 6, 3))
    due = sync_opu.checkpoint_of(_fields(**{"Бласт": 5, "2pN": 8}),
                                 today=datetime(2026, 6, 8))
    assert early == "T+1 оплодотворение"
    assert due == "T+6 закрытие"


def test_checkpoint_without_date_has_not_started():
    assert sync_opu.checkpoint_of({}) == "— не наступил"


def test_checkpoint_at_opu_when_nothing_is_filled():
    assert sync_opu.checkpoint_of(_fields()) == "T0 пункция"


# --- Хеш содержимого -------------------------------------------------------

def test_hash_ignores_service_fields():
    """Служебные поля из хеша исключены намеренно.

    Иначе «Обновлено» менялось бы каждый прогон само от себя, и журнал перестал бы
    отличать реальную правку от холостого хода.
    """
    base = {"MII": 10, "Обновлено": "2026-06-01T10:00:00", "Окружение": "production",
            "Чекпойнт": "T0 пункция", "Хеш строки": "старый"}
    moved = {**base, "Обновлено": "2026-06-02T10:00:00", "Окружение": "draft",
             "Чекпойнт": "T+6 закрытие", "Хеш строки": "новый"}
    assert sync_opu.content_hash(base) == sync_opu.content_hash(moved)


def test_hash_changes_when_the_data_changes():
    assert sync_opu.content_hash({"MII": 10}) != sync_opu.content_hash({"MII": 11})


def test_hash_distinguishes_missing_from_zero():
    assert sync_opu.content_hash({"Бласт": None}) != sync_opu.content_hash({"Бласт": 0})


# --- Upsert ----------------------------------------------------------------

SHEET_COLUMNS = ["ID", "Patient Full Name", "Date OPU", "MII", "#"]


def _sheet(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=SHEET_COLUMNS).fillna("")


def _existing(key: str, row_hash: str, record_id="rec1") -> dict:
    return {"id": record_id, "fields": {"Ключ": key, "Хеш строки": row_hash}}


def _sync(cfg, client, frame, dry_run=False):
    return sync_opu.sync(cfg, client, {"OPU": "tblOPU", "Sync": "tblSync"},
                         frame, secret=b"secret", source="лист", dry_run=dry_run)


def test_new_rows_are_created(cfg):
    client = FakeTeable([])
    report = _sync(cfg, client, _sheet([
        {"ID": "A1", "Patient Full Name": "П", "Date OPU": "01.06.2026", "MII": "10",
         "#": "1"}]))
    assert report.created == 1 and report.updated == 0
    created = client.created[0][1][0]
    assert created["Ключ"] == "A1#01.06.2026"
    assert created["cycle_uid"] and created["cycle_uid"] != "A1"


def test_unchanged_rows_are_left_alone(cfg):
    """Строка без изменений не переписывается: иначе «Обновлено» лжёт."""
    client = FakeTeable([])
    row = {"ID": "A1", "Patient Full Name": "П", "Date OPU": "01.06.2026",
           "MII": "10", "#": "1"}
    _sync(cfg, client, _sheet([row]))
    stored = client.created[0][1][0]

    again = FakeTeable([_existing(stored["Ключ"], stored["Хеш строки"])])
    report = _sync(cfg, again, _sheet([row]))
    assert report.unchanged == 1 and report.created == 0 and report.updated == 0
    assert again.updated == []


def test_changed_rows_are_updated_in_place(cfg):
    client = FakeTeable([])
    row = {"ID": "A1", "Patient Full Name": "П", "Date OPU": "01.06.2026",
           "MII": "10", "#": "1"}
    _sync(cfg, client, _sheet([row]))
    stored = client.created[0][1][0]

    again = FakeTeable([_existing(stored["Ключ"], stored["Хеш строки"])])
    report = _sync(cfg, again, _sheet([{**row, "MII": "11"}]))
    assert report.updated == 1 and report.created == 0
    assert again.updated[0][1][0][0] == "rec1", "правка идёт в ту же запись"


def test_duplicate_key_is_skipped_not_merged(cfg):
    """Дубль ключа в самом листе не сливается и не выбирается «полная» строка.

    Какая из них верна, решает владелец: молчаливый выбор здесь означал бы, что
    система тихо потеряла данные цикла.
    """
    client = FakeTeable([])
    row = {"ID": "A1", "Patient Full Name": "П", "Date OPU": "01.06.2026",
           "MII": "10", "#": "1"}
    report = _sync(cfg, client, _sheet([row, {**row, "MII": "11", "#": "2"}]))
    assert report.skipped == 1 and report.created == 1
    assert any("дубль ключа" in note for note in report.notes)


def test_row_without_key_is_skipped_with_a_note(cfg):
    client = FakeTeable([])
    report = _sync(cfg, client, _sheet([
        {"ID": "", "Patient Full Name": "П", "Date OPU": "", "MII": "10", "#": "7"}]))
    assert report.created == 0 and report.skipped == 1
    assert any("без ID или даты" in note for note in report.notes)


def test_rows_missing_from_the_sheet_are_never_deleted(cfg):
    """Строка исчезла из листа — она остаётся в зеркале, с записью в журнале.

    Удаление данных пациентки по расхождению выгрузок — не то решение, которое
    принимает скрипт.
    """
    client = FakeTeable([_existing("OLD#01.01.2026", "хеш")])
    report = _sync(cfg, client, _sheet([
        {"ID": "A1", "Patient Full Name": "П", "Date OPU": "01.06.2026", "MII": "10",
         "#": "1"}]))
    assert any("отсутствуют в листе" in note for note in report.notes)
    assert client.updated == []


def test_dry_run_writes_nothing_but_reports_everything(cfg):
    """`--dry-run` честный: считает всё, не записывает ничего — включая журнал."""
    client = FakeTeable([])
    report = _sync(cfg, client, _sheet([
        {"ID": "A1", "Patient Full Name": "П", "Date OPU": "01.06.2026", "MII": "10",
         "#": "1"}]), dry_run=True)
    assert report.created == 1
    assert client.created == [] and client.updated == []


def test_sync_writes_its_own_journal_entry(cfg):
    client = FakeTeable([])
    _sync(cfg, client, _sheet([
        {"ID": "A1", "Patient Full Name": "П", "Date OPU": "01.06.2026", "MII": "10",
         "#": "1"}]))
    tables = [table for table, _ in client.created]
    assert "tblSync" in tables


def test_pseudonym_is_stable_for_the_same_cycle(cfg):
    """Один и тот же цикл всегда получает один и тот же псевдоним.

    Иначе связь между прогонами рвалась бы, а «уже оценённые циклы» переставали бы
    узнаваться.
    """
    row = {"ID": "A1", "Patient Full Name": "П", "Date OPU": "01.06.2026",
           "MII": "10", "#": "1"}
    first, second = FakeTeable([]), FakeTeable([])
    _sync(cfg, first, _sheet([row]))
    _sync(cfg, second, _sheet([row]))
    assert first.created[0][1][0]["cycle_uid"] == second.created[0][1][0]["cycle_uid"]


# --- Чтение экспорта -------------------------------------------------------

def test_export_drops_trailing_empty_rows(tmp_path):
    """Хвост пустых строк листа циклами не является."""
    path = tmp_path / "opu.csv"
    path.write_text("ID,Patient Full Name,Date OPU\n"
                    "A1,П,01.06.2026\n"
                    ",,\n"
                    ",,\n", encoding="utf-8")
    frame = sync_opu.load_sheet_export(path)
    assert len(frame) == 1


def test_export_keeps_a_row_with_only_a_name(tmp_path):
    """Строка без ID, но с пациенткой — это данные; отбросит её уже upsert, с записью."""
    path = tmp_path / "opu.csv"
    path.write_text("ID,Patient Full Name,Date OPU\n,П,01.06.2026\n", encoding="utf-8")
    assert len(sync_opu.load_sheet_export(path)) == 1
