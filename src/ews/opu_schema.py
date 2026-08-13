"""Декларативная схема базы «EWS — OPU master» в локальном Teable.

Одна база = зеркало мастер-таблицы OPU + журнал синхронизаций. Схема применяется
идемпотентно (`scripts/ews_bootstrap_opu.py`): недостающие таблицы создаются,
недостающие поля добавляются, ничего не переименовывается и не удаляется.

Что СОЗНАТЕЛЬНО не зеркалится: данные мужа (ФИО, ДР, паспортный ID). EWS их никогда
не использует, а любая PHI, которую мы не храним, — это PHI, которую невозможно
утечь. Остальные идентификаторы пациентки нужны для обратного сопоставления и
остаются локальными: в агентский пакет уходит только cycle_uid.
"""
from __future__ import annotations

# Порядок = порядок колонок в Teable. Первое поле таблицы становится первичным.
METHOD_CHOICES = [
    {"name": "ICSI", "color": "blueLight1"},
    {"name": "IVF", "color": "cyanLight1"},
    {"name": "NO", "color": "grayLight1"},
    {"name": "Cryo", "color": "purpleLight1"},
]

ENV_CHOICES = [
    {"name": "draft", "color": "yellowLight1"},
    {"name": "production", "color": "greenLight1"},
]

CHECKPOINT_CHOICES = [
    {"name": "T0 пункция", "color": "blueLight1"},
    {"name": "T+1 оплодотворение", "color": "cyanLight1"},
    {"name": "T+6 закрытие", "color": "greenLight1"},
    {"name": "— не наступил", "color": "grayLight1"},
]


def txt():
    return {"type": "singleLineText"}


def longtxt():
    return {"type": "longText"}


def num(precision: int):
    return {"type": "number",
            "options": {"formatting": {"type": "decimal", "precision": precision}}}


def date(withtime: bool = False, tz: str = "Asia/Tashkent"):
    return {"type": "date", "options": {"formatting": {
        "date": "YYYY-MM-DD", "time": "HH:mm" if withtime else "None", "timeZone": tz}}}


def sel(choices):
    return {"type": "singleSelect", "options": {"choices": list(choices)}}


def f(name: str, spec: dict) -> dict:
    return {"name": name, **spec}


# Соответствие «колонка листа -> поле Teable». Односторонняя карта: лист —
# источник, Teable — зеркало. Имена листа берутся ровно как в экспорте, включая
# висячие пробелы ('Addition ') — угадывать их запрещено.
SHEET_TO_FIELD: dict[str, str] = {
    "ID": "ID",
    "Patient Full Name": "Имя",
    "DOB": "ДР",
    "#": "№ строки",
    "Date OPU": "Дата OPU",
    "Date ET": "Дата ET",
    "OPU time": "Время OPU",
    "Age": "Возраст",
    "BMI": "BMI",
    "AMH": "AMH",
    "FSH": "FSH",
    "Antral follicule count": "КАФ",
    "# Attempt": "№ попытки",
    "cause of infertility": "Причина бесплодия",
    "Stimulation, days": "Стимуляция, дней",
    "Start FSH dosage": "Старт ФСГ",
    "Sum stim dosage": "Сумма ФСГ",
    "N folicules OPU": "Фолликулы",
    "OCC": "ОКК",
    "MII": "MII",
    "IVF/ICSI": "Метод",
    "sperm": "Сперма",
    "Inseminated": "Инсеминировано",
    "2pN": "2pN",
    "3pN": "3pN",
    "1pN": "1pN",
    "0pN": "0pN",
    "Deg": "Deg",
    "Cleavage": "Дробление D3",
    "Bl": "Бласт",
    "Good Bl": "Хор.бласт",
    "Cryo": "Криоконсервировано",
    "ET": "Перенесено",
    "Day of ET": "День переноса",
    "Embryo grade": "Качество эмбриона",
    "Preg": "Исход",
    "Embryo implanted": "Имплантировано",
    "PGT": "ПГТ",
    "PGT REQUEST": "Запрос ПГТ",
    "PRAI": "PRAI (лист)",
    "DIGITAL TWIN": "DIGITAL TWIN (лист)",
    "Embryologist": "Эмбриолог",
    "Doctor": "Врач",
}

# Числовые поля и их точность. Всё остальное из карты выше — текст.
NUMERIC_FIELDS: dict[str, int] = {
    "№ строки": 0, "Возраст": 0, "BMI": 1, "AMH": 2, "FSH": 2, "КАФ": 0,
    "№ попытки": 0, "Стимуляция, дней": 0, "Старт ФСГ": 0, "Сумма ФСГ": 0,
    "Фолликулы": 0, "ОКК": 0, "MII": 0, "Инсеминировано": 0, "2pN": 0,
    "3pN": 0, "1pN": 0, "0pN": 0, "Deg": 0, "Дробление D3": 0, "Бласт": 0,
    "Хор.бласт": 0, "Криоконсервировано": 0, "Перенесено": 0, "День переноса": 0,
    "Имплантировано": 0, "PRAI (лист)": 3, "DIGITAL TWIN (лист)": 3,
}

DATE_FIELDS = ("Дата OPU", "Дата ET")

# Поля воронки — те, чьё появление двигает цикл по чекпойнтам.
FUNNEL_T1 = ("2pN", "3pN", "1pN", "0pN", "Deg")
FUNNEL_T6 = ("Дробление D3", "Бласт", "Хор.бласт", "Криоконсервировано")


def _opu_fields(tz: str) -> list[dict]:
    fields = [
        f("Ключ", txt()),                 # ID#Дата OPU — первичный, по нему upsert
        f("cycle_uid", txt()),
    ]
    for sheet_name, field_name in SHEET_TO_FIELD.items():
        if field_name in ("Ключ", "cycle_uid"):
            continue
        if field_name in DATE_FIELDS:
            fields.append(f(field_name, date(tz=tz)))
        elif field_name == "Метод":
            fields.append(f(field_name, sel(METHOD_CHOICES)))
        elif field_name == "Причина бесплодия":
            fields.append(f(field_name, longtxt()))
        elif field_name in NUMERIC_FIELDS:
            fields.append(f(field_name, num(NUMERIC_FIELDS[field_name])))
        else:
            fields.append(f(field_name, txt()))
    fields += [
        f("Чекпойнт", sel(CHECKPOINT_CHOICES)),
        f("Хеш строки", txt()),
        f("Обновлено", date(withtime=True, tz=tz)),
        f("Окружение", sel(ENV_CHOICES)),
    ]
    return fields


def _sync_fields(tz: str) -> list[dict]:
    return [
        f("Прогон", txt()),
        f("Начат", date(withtime=True, tz=tz)),
        f("Источник", txt()),
        f("Строк прочитано", num(0)),
        f("Создано", num(0)),
        f("Обновлено", num(0)),
        f("Без изменений", num(0)),
        f("Пропущено", num(0)),
        f("Окружение", sel(ENV_CHOICES)),
        f("Замечания", longtxt()),
    ]


def schema(tz: str = "Asia/Tashkent") -> list[dict]:
    return [
        {"key": "OPU", "name": "OPU — мастер циклов", "fields": _opu_fields(tz)},
        {"key": "Sync", "name": "Sync — журнал синхронизаций", "fields": _sync_fields(tz)},
    ]
