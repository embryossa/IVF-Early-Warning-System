"""Ingest: чекпойнты, приватность на границе, пригодность наблюдения для карты.

Здесь проходит граница приватности всего проекта (`CLAUDE.md`: псевдонимизация на
ingest, дальше PHI не уходит вообще), и здесь же решается, что вообще считается
наблюдением. Обе вещи проверяются кодом, потому что нарушение любой из них не видно
по выходным числам: отчёт с PHI выглядит как обычный отчёт, а карта, съевшая долю
больше единицы, — как обычная карта.
"""
from datetime import datetime

import pandas as pd
import pytest

from conftest import FakeTeable, teable_record

from ews import cycles


# --- Чекпойнт --------------------------------------------------------------

def _row(**values) -> pd.Series:
    base = {"date_opu": pd.Timestamp("2026-06-01"), "cleavage_d3": None,
            "bl_total": None, "bl_good": None, "frozen": None, "two_pn": None,
            "three_pn": None, "one_pn": None, "zero_pn": None, "degrade": None}
    base.update(values)
    return pd.Series(base)


def test_no_opu_date_means_no_checkpoint():
    """Без даты пункции цикл не стоит нигде: считать его закрытым не от чего."""
    assert cycles.assign_checkpoint(_row(date_opu=pd.NaT),
                                    datetime(2026, 6, 20), 6) == cycles.CHECKPOINT_NONE


def test_closure_needs_both_data_and_calendar():
    """Заполненная воронка до шестого дня закрытием не является.

    Данные могут появиться раньше срока (внесли заранее, ошиблись строкой). Оценка
    на T+6 — неподвижное решение владельца, и календарь здесь ограничение, а не
    украшение: иначе тревога поднималась бы на неполном окне культивирования.
    """
    row = _row(bl_total=5.0, two_pn=8.0)
    early = cycles.assign_checkpoint(row, datetime(2026, 6, 4), 6)
    due = cycles.assign_checkpoint(row, datetime(2026, 6, 7), 6)
    assert early == cycles.CHECKPOINT_FERTILISATION
    assert due == cycles.CHECKPOINT_CLOSURE


def test_late_filled_row_still_reaches_closure():
    """Строку заполнили с опозданием — цикл всё равно закрыт.

    Календарь один не годится: стадия определяется наличием значений, а дата лишь
    отсекает то, что физически не могло дойти до закрытия.
    """
    row = _row(bl_total=5.0, two_pn=8.0)
    assert cycles.assign_checkpoint(row, datetime(2026, 7, 30), 6) == cycles.CHECKPOINT_CLOSURE


def test_fertilisation_recognised_by_any_pn_field():
    """T+1 наступил, если заполнено ЛЮБОЕ поле проверки оплодотворения.

    В том числе нулевое: 0pN — известное наблюдение, а не пропуск.
    """
    for field in ("two_pn", "three_pn", "one_pn", "zero_pn", "degrade"):
        row = _row(**{field: 0.0})
        assert cycles.assign_checkpoint(row, datetime(2026, 6, 20), 6) == \
            cycles.CHECKPOINT_FERTILISATION, field


def test_only_opu_data_stays_at_opu():
    assert cycles.assign_checkpoint(_row(), datetime(2026, 6, 20), 6) == cycles.CHECKPOINT_OPU


# --- Пригодность наблюдения ------------------------------------------------

def test_zero_numerator_is_an_observation():
    """Ноль бластоцист — наблюдение, а не пропуск. Правило проекта, не оптимизация."""
    row = pd.Series({"bl_total": 0.0, "two_pn": 8.0})
    assert cycles.metric_counts(row, "BlR") == (0.0, 8.0)


def test_missing_value_is_not_zero():
    row = pd.Series({"bl_total": None, "two_pn": 8.0})
    assert cycles.metric_counts(row, "BlR") is None


def test_zero_denominator_is_no_observation():
    """Знаменатель 0 — не «доля 0», а отсутствие наблюдения."""
    row = pd.Series({"bl_total": 0.0, "two_pn": 0.0})
    assert cycles.metric_counts(row, "BlR") is None


def test_ratio_above_one_is_excluded_from_charts():
    """ClR>100% и MII>ОКК — штатные ситуации, но для карты непригодны.

    Правило проекта запрещает называть это порчей данных; доля вне [0,1] при этом
    не является вероятностью Бернулли, и подавать её в карту нельзя.
    """
    assert cycles.metric_counts(pd.Series({"cleavage_d3": 9.0, "two_pn": 8.0}), "ClR") is None
    assert cycles.metric_counts(pd.Series({"mii": 13.0, "occ": 12.0}), "MIIR") is None


def test_nan_denominator_is_no_observation():
    row = pd.Series({"bl_total": 5.0, "two_pn": float("nan")})
    assert cycles.metric_counts(row, "BlR") is None


def test_empty_cell_is_missing_not_zero():
    assert cycles._number("") is None
    assert cycles._number(None) is None
    assert cycles._number("0") == 0.0
    assert cycles._number("не число") is None


# --- Загрузка из Teable ----------------------------------------------------

def _load(cfg, clinic, records, today=datetime(2026, 6, 20)):
    client = FakeTeable(records)
    frame = cycles.load_from_teable(cfg, clinic, client, "tblOPU", today=today)
    return frame, client


def test_ingest_never_asks_teable_for_phi(cfg, clinic):
    """Проекция запроса не содержит ни одного поля с PHI.

    Граница приватности проходит здесь. Проверять её по готовому кадру бесполезно:
    ФИО могло приехать по сети и быть отброшено уже после — а утечь оно способно
    из любого места между этими двумя точками.
    """
    _, client = _load(cfg, clinic, [teable_record("CYC-1")])
    asked = set(client.projections[0])
    forbidden = {"Имя", "ДР", "ID", "Ключ", "Качество эмбриона"}
    assert not (asked & forbidden), sorted(asked & forbidden)
    assert "cycle_uid" in asked


def test_rows_without_cycle_uid_are_dropped(cfg, clinic):
    """Строка без псевдонима в конвейер не идёт: связать её не с чем."""
    frame, _ = _load(cfg, clinic, [teable_record("CYC-1"), teable_record("")])
    assert list(frame.data["cycle_uid"]) == ["CYC-1"]


def test_opu_date_becomes_clinic_calendar_date(cfg, clinic):
    """Teable отдаёт UTC, а дата пункции — календарная дата клиники.

    Без перевода в часовой пояс лаборатории пункция позднего вечера уезжала бы на
    сутки, а вместе с ней и граница окна культивирования.
    """
    # Asia/Tashkent — UTC+5: 19:00 UTC это уже следующие сутки в лаборатории.
    record = {"id": "rec1", "fields": {"cycle_uid": "CYC-1",
                                       "Дата OPU": "2026-06-01T19:00:00.000Z"}}
    frame, _ = _load(cfg, clinic, [record])
    stamp = frame.data["date_opu"].iloc[0]
    assert stamp == pd.Timestamp("2026-06-02"), stamp
    assert stamp.tz is None, "дальше по конвейеру дата обязана быть наивной"


def test_checkpoint_is_recomputed_not_read_from_teable(cfg, clinic):
    """Чекпойнт считается по данным заново, а не берётся из зеркала.

    В Teable поле «Чекпойнт» не входит в хеш строки и потому может отстать: цикл,
    у которого шестой день наступил без единой правки, там так и останется «T+1».
    Конвейер обязан не зависеть от этого поля — и не зависит.
    """
    record = teable_record("CYC-1", **{"Бласт": 5, "2pN": 8, "Чекпойнт": "T+1 оплодотворение"})
    frame, client = _load(cfg, clinic, [record])
    assert frame.data["checkpoint"].iloc[0] == cycles.CHECKPOINT_CLOSURE
    assert "Чекпойнт" not in client.projections[0]


def test_frame_is_sorted_chronologically(cfg, clinic):
    """Порядок обязателен: карта секвенциальна, перестановка меняет её траекторию."""
    frame, _ = _load(cfg, clinic, [
        teable_record("B", date_opu="2026-06-05"),
        teable_record("A", date_opu="2026-06-01"),
    ])
    assert list(frame.data["cycle_uid"]) == ["A", "B"]


def test_closure_due_is_opu_plus_window(cfg, clinic):
    frame, _ = _load(cfg, clinic, [teable_record("CYC-1", date_opu="2026-06-01")])
    assert frame.data["closure_due"].iloc[0] == pd.Timestamp("2026-06-07")


def test_empty_table_yields_usable_empty_frame(cfg, clinic):
    """Пустое зеркало — это результат «циклов нет», а не сбой.

    Первый прогон новой клиники выглядит именно так, и упасть здесь значит не дать
    системе стартовать вовсе.
    """
    frame, _ = _load(cfg, clinic, [])
    assert frame.data.empty
    assert "checkpoint" in frame.data.columns


# --- Безопасное чтение колонок ---------------------------------------------

def test_numeric_column_survives_missing_column():
    """Отсутствующая колонка даёт РЯД пропусков, а не скаляр.

    `pd.to_numeric(frame.get(name))` на отсутствующей колонке возвращает скаляр nan,
    и падает потом — в чужом месте, с сообщением про numpy.float64. Кадр без колонок
    приходит с пустой таблицы, то есть в самом обычном случае.
    """
    frame = pd.DataFrame(columns=["cycle_uid"])
    series = cycles.numeric_column(frame, "age")
    assert isinstance(series, pd.Series)
    assert series.empty

    with_rows = pd.DataFrame({"cycle_uid": ["A", "B"]})
    series = cycles.numeric_column(with_rows, "age")
    assert len(series) == 2 and series.isna().all()


def test_datetime_column_survives_missing_column():
    frame = pd.DataFrame({"cycle_uid": ["A"]})
    series = cycles.datetime_column(frame, "date_opu")
    assert isinstance(series, pd.Series) and series.isna().all()


def test_numeric_column_coerces_text():
    frame = pd.DataFrame({"age": ["33", "н/д", None]})
    series = cycles.numeric_column(frame, "age")
    assert series.iloc[0] == 33.0
    assert series.iloc[1:].isna().all()


@pytest.mark.parametrize("metric", sorted(cycles.METRIC_PARTS))
def test_every_metric_has_both_parts(metric):
    numerator, denominator = cycles.METRIC_PARTS[metric]
    assert numerator and denominator and numerator != denominator
