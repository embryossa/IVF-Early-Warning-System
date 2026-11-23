"""Профиль лаборатории и приведение выгрузки к каноническому кадру.

Разделение, на котором стоит воспроизводимость: `settings.yaml` — метод, одинаковый
везде; `config/clinics/<code>/` — лаборатория. Запуск на новой лаборатории не требует
правок кода, и именно поэтому профиль обязан проверяться: ошибка в нём выглядит как
ошибка метода.
"""
import pandas as pd
import pytest

from ews import clinic as clinic_mod
from ews.clinic import Clinic
from ews.config import ConfigError


def _clinic(profile=None, columns=None) -> Clinic:
    return Clinic(code="test", profile=profile or {}, columns=columns or {})


# --- Набор карт ------------------------------------------------------------

def test_subjective_denominator_cannot_get_a_chart():
    """Метрика с субъективным знаменателем не попадает на карту по недосмотру.

    Проверка, а не соглашение: карта на OCCR ловила бы изменения в том, КАК считают
    фолликулы, а не в том, как работают (METHODOLOGY §5.5), и профиль копируют.
    """
    profile = {"charts": {"cusum": ["BlR", "OCCR"],
                          "subjective_denominator": {"OCCR": "счёт врача"}}}
    with pytest.raises(ConfigError, match="subjective_denominator"):
        _clinic(profile).cusum_charts


def test_empty_chart_list_is_an_error():
    with pytest.raises(ConfigError, match="нечего калибровать"):
        _clinic({"charts": {"cusum": []}}).cusum_charts


def test_real_profile_has_one_chart_per_contour(cfg, clinic):
    """У этой лаборатории по одной карте на контур — это осознанный предел объёма.

    Каждая лишняя карта ужесточает пороги всем остальным; при ~70 эмбрионах в месяц
    три — практический максимум.
    """
    contours = cfg.settings["localisation"]["contours"]
    placed = {metric: name for name, metrics in contours.items()
              for metric in metrics if metric in clinic.cusum_charts}
    assert sorted(placed) == sorted(clinic.cusum_charts)
    assert len(set(placed.values())) == len(clinic.cusum_charts)


# --- Опорное окно ----------------------------------------------------------

def test_missing_reference_window_is_an_error():
    """Без опорного окна нет центра относительного контура — считать не от чего."""
    with pytest.raises(ConfigError, match="reference_window.start"):
        _clinic().reference_window


def test_open_window_ends_today():
    bounds = _clinic({"reference_window": {"start": "2026-05-01"}}).window_bounds()
    assert bounds[0] == pd.Timestamp("2026-05-01")
    assert bounds[1] >= bounds[0]


def test_declared_shifts_travel_with_the_profile(clinic):
    """Объяснение сдвига живёт в профиле рядом с окном, а не в чьей-то памяти."""
    assert "OCCR" in clinic.declared_shifts
    assert clinic.declared_shifts["OCCR"]["confirmed_by_data"] is True


# --- Загрузка профиля ------------------------------------------------------

def test_unknown_clinic_lists_the_available_ones(cfg):
    with pytest.raises(ConfigError, match="нет профиля клиники"):
        clinic_mod.load_clinic(cfg, "не_существует")


def test_available_lists_real_profiles():
    assert "two_line" in clinic_mod.available()
    assert not any(name.startswith("_") for name in clinic_mod.available())


def test_clinic_data_dirs_never_overlap(cfg, clinic):
    """Артефакты клиник лежат раздельно: смешать их нельзя."""
    other = Clinic(code="другая", profile={}, columns={})
    assert clinic.data_dir(cfg) != other.data_dir(cfg)
    assert clinic.data_dir(cfg).name == clinic.code


# --- Приведение к каноническому кадру -------------------------------------

def _sheet(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_missing_column_is_reported_not_guessed(clinic):
    """Отсутствующая колонка попадает в список, а не заменяется догадкой.

    Метрика, которой она нужна, окажется недоступна явно — это и есть требуемое
    поведение: посчитать по подвернувшемуся столбцу нельзя.
    """
    frame = _sheet([{"Age": "33", "Date OPU": "01.06.2026"}])
    canonical, missing = clinic_mod.to_canonical(clinic, frame)
    assert any("bl_total (Bl)" in item for item in missing)
    assert canonical["bl_total"].isna().all()


def test_decimal_comma_is_understood(clinic):
    """Запятая как десятичный разделитель — норма этой выгрузки."""
    frame = _sheet([{"AMH": "2,45", "Date OPU": "01.06.2026"}])
    canonical, _ = clinic_mod.to_canonical(clinic, frame)
    assert canonical["amh"].iloc[0] == pytest.approx(2.45)


def test_empty_cell_becomes_missing_not_zero(clinic):
    frame = _sheet([{"Bl": "", "Date OPU": "01.06.2026"}])
    canonical, _ = clinic_mod.to_canonical(clinic, frame)
    assert pd.isna(canonical["bl_total"].iloc[0])


def test_zero_stays_zero(clinic):
    frame = _sheet([{"Bl": "0", "Date OPU": "01.06.2026"}])
    canonical, _ = clinic_mod.to_canonical(clinic, frame)
    assert canonical["bl_total"].iloc[0] == 0.0


def test_attempt_default_fills_only_the_gaps(clinic):
    """Пустой «№ попытки» означает первую попытку — доменное решение владельца.

    Подстановка выполняется на границе приведения, чтобы весь конвейер видел одно
    и то же значение, а не каждый своё.
    """
    frame = _sheet([{"# Attempt": "", "Date OPU": "01.06.2026"},
                    {"# Attempt": "3", "Date OPU": "01.06.2026"}])
    canonical, _ = clinic_mod.to_canonical(clinic, frame)
    assert list(canonical["attempt_no"]) == [1.0, 3.0]


def test_dates_are_parsed_by_every_declared_format(clinic):
    frame = _sheet([{"Date OPU": "01.06.2026"}, {"Date OPU": "2026-06-02"},
                    {"Date OPU": "03/06/2026"}, {"Date OPU": "мусор"}])
    canonical, _ = clinic_mod.to_canonical(clinic, frame)
    assert list(canonical["date_opu"][:3]) == [pd.Timestamp("2026-06-01"),
                                               pd.Timestamp("2026-06-02"),
                                               pd.Timestamp("2026-06-03")]
    assert pd.isna(canonical["date_opu"].iloc[3])


def test_raw_opu_date_is_kept_for_the_key(clinic):
    """Сырая дата пункции сохраняется: ключ цикла не должен зависеть от разбора."""
    frame = _sheet([{"Date OPU": " 01.06.2026 "}])
    canonical, _ = clinic_mod.to_canonical(clinic, frame)
    assert canonical["date_opu_raw"].iloc[0] == "01.06.2026"


# --- Ключ цикла ------------------------------------------------------------

def test_cycle_key_is_composite(clinic):
    row = pd.Series({"patient_id": "ab1", "date_opu_raw": "01.06.2026"})
    assert clinic_mod.cycle_key(clinic, row) == "AB1#01.06.2026"


def test_cycle_key_is_empty_when_a_part_is_missing(clinic):
    """Неполный ключ не достраивается: цикл лучше пропустить, чем склеить с чужим."""
    assert clinic_mod.cycle_key(clinic, pd.Series({"patient_id": None,
                                                   "date_opu_raw": "01.06.2026"})) == ""
    assert clinic_mod.cycle_key(clinic, pd.Series({"patient_id": "AB1",
                                                   "date_opu_raw": ""})) == ""


# --- Профиль как источник правды ------------------------------------------

def test_phi_columns_are_declared_in_the_profile(clinic):
    """Список PHI объявлен явно: угадывать, что является PHI, нельзя."""
    assert set(clinic.phi_columns) >= {"patient_id", "patient_name", "dob"}


def test_timezone_defaults_are_explicit():
    assert _clinic().timezone == "Asia/Tashkent"
    assert _clinic({"timezone": "UTC"}).timezone == "UTC"


def test_amh_constant_is_optional():
    assert _clinic().amh_constant is None
    assert _clinic({"imputation": {"amh_constant": 2.0}}).amh_constant == 2.0
