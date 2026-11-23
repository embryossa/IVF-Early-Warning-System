"""Скрипты: состав цепочки, расписание, приём выгрузки, запись артефактов.

Скрипты — не «обвязка вокруг настоящего кода». В них живут ровно те решения, ошибка
в которых не видна ни по числам, ни по логу: какой шаг мягкий, в какие дни идёт
задача, что считается настоящей выгрузкой листа, и переживёт ли завтрашний прогон
сегодняшний пустой файл.
"""
from xml.etree import ElementTree

import pandas as pd
import pytest

from conftest import load_script


schedule = load_script("ews_schedule")
daily = load_script("ews_daily")
fetch = load_script("ews_fetch_opu_sheet")
run = load_script("ews_run")


# --- Расписание ------------------------------------------------------------

def _days(schedule_xml: str) -> set[str]:
    fragment = ElementTree.fromstring(f"<root>{schedule_xml}</root>")
    return {child.tag for child in fragment.find(".//DaysOfWeek")}


ALL_DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
            "Sunday"}


def test_the_two_tasks_together_cover_every_day_exactly_once():
    """Ровно один прогон в сутки — ни пропущенного дня, ни двойного понедельника.

    Две задачи в один день писали бы в журнал две строки «по расписанию» за один
    понедельник и дважды прокручивали бы одну синхронизацию; пропущенный день —
    это день без оценки закрывшихся циклов.
    """
    daily_days = _days(schedule.DAILY_SCHEDULE)
    weekly_days = _days(schedule.WEEKLY_SCHEDULE)
    assert daily_days | weekly_days == ALL_DAYS
    assert daily_days & weekly_days == set()
    assert weekly_days == {"Monday"}


def test_only_the_weekly_task_carries_the_weekly_flag():
    """Ключ `--weekly` стоит ровно у понедельничной задачи.

    Перепутанный, он не ломает ничего заметного: суточная задача каждый день пять
    минут разбирала бы логи, а недельная никогда не пересчитывала бы границы
    «необычного» — и подтверждение по среде тихо оставалось бы недоступным.
    """
    registered: dict[str, str] = {}

    def spy(name, description, start, schedule_xml, arguments):
        registered[name] = arguments
        return True, ""

    # Клиника настоящая: `install` заводит каталог состояния, и на выдуманном коде
    # тест оставлял бы за собой мусор в data/.
    schedule._register, original = spy, schedule._register
    try:
        assert schedule.install("two_line", "15:45") == 0
    finally:
        schedule._register = original

    daily_args = registered[schedule.daily_task_name("two_line")]
    weekly_args = registered[schedule.weekly_task_name("two_line")]
    assert "--weekly" not in daily_args
    assert "--weekly" in weekly_args
    assert all("--triggered-by schedule" in args for args in registered.values()), (
        "по этому ключу журнал отличает плановый запуск от ручного — без него "
        "«пропущен ли прогон» считать не по чему")


def test_task_xml_survives_an_empty_lab():
    """Три умолчания планировщика неверны для лабораторной машины, и все три молчат.

    Задача просто не запускается, «Последний запуск» остаётся прошлым, и отличить
    это от «сегодня не было новых циклов» нечем.
    """
    xml = schedule.TASK_XML.format(
        name="EWS_test_daily", description="", sid="S-1-5-21",
        limit=schedule.EXECUTION_TIME_LIMIT, start="2026-01-05T15:45:00",
        schedule=schedule.DAILY_SCHEDULE, command='"python.exe"',
        arguments="--clinic test", workdir="D:\\EWS")
    tree = ElementTree.fromstring(xml.split("?>", 1)[1])
    namespace = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    assert tree.find(".//t:LogonType", namespace).text == "S4U", (
        "InteractiveToken означает «только при вошедшем пользователе»")
    assert tree.find(".//t:StartWhenAvailable", namespace).text == "true"
    assert tree.find(".//t:DisallowStartIfOnBatteries", namespace).text == "false"
    assert tree.find(".//t:StopIfGoingOnBatteries", namespace).text == "false"
    assert tree.find(".//t:MultipleInstancesPolicy", namespace).text == "IgnoreNew"


def test_stale_tolerance_matches_the_daily_cadence():
    """Допуск «пропущенного прогона» — двое суток, и это не круглое число.

    Одни сутки объявляли бы пропуском каждый перенос из-за выключенной машины
    (StartWhenAvailable догоняет его на следующий день); неделя, как при прошлом
    расписании, не заметила бы шести дней молчания.
    """
    assert schedule.STALE_AFTER_DAYS == 2


def test_task_names_are_distinct_per_clinic():
    assert schedule.task_names("a") != schedule.task_names("b")
    assert len(set(schedule.task_names("two_line"))) == 2


# --- Состав цепочки --------------------------------------------------------

def _labels(weekly: bool) -> list[str]:
    return [label for label, _, _ in daily.build_steps("py", "two_line", weekly=weekly)]


def test_daily_chain_has_no_weekly_work():
    """Суточный прогон не разбирает логи и не добирает конверты.

    Разбор всей когорты — 5 минут из 6, а выгрузка MIRI приходит понедельной
    нарезкой: читать её ежедневно нечего.
    """
    labels = _labels(weekly=False)
    assert "недельное обогащение" not in labels
    assert "добор конвертов за неделю" not in labels
    assert labels == ["обновление экспорта листа", "синхронизация OPU", "прогноз DT",
                      "вероятности DT в зеркало", "регистрация конверта", "прогон"]


def test_dt_forecast_lands_before_evaluation():
    """Прогноз DT считается после синхронизации листа и забирается в зеркало до оценки.

    Иначе новый цикл получал бы вероятность DT только на следующий день, а повторная
    синхронизация до прогноза не забирала бы ничего.
    """
    labels = _labels(weekly=False)
    assert (labels.index("синхронизация OPU") < labels.index("прогноз DT")
            < labels.index("вероятности DT в зеркало") < labels.index("прогон"))


def test_weekly_chain_adds_exactly_two_steps():
    assert set(_labels(weekly=True)) - set(_labels(weekly=False)) == {
        "добор конвертов за неделю", "недельное обогащение"}


def test_backfill_runs_before_evaluation():
    """Добор конвертов обязан идти ДО оценки, а обогащение средой — после.

    Порядок не косметика: конверт, зарегистрированный после прогона, не попадёт в
    сегодняшний слой QA, а среда, разобранная до оценки, не будет знать, что
    объяснять.
    """
    labels = _labels(weekly=True)
    assert labels.index("добор конвертов за неделю") < labels.index("прогон")
    assert labels.index("прогон") < labels.index("недельное обогащение")


def test_evaluation_is_the_only_hard_step():
    """Всё, кроме оценки, — обстановка: её отсутствие прогон не отменяет.

    Обратное тоже верно и важнее: жёсткий шаг, ставший мягким, превратил бы сбой
    оценки в «прогон прошёл с пропусками».
    """
    hard = [label for label, _, soft in daily.build_steps("py", "c", weekly=True)
            if not soft]
    assert hard == ["прогон"]


def test_dry_run_reaches_every_step():
    """`--dry-run` не должен обрываться на середине цепочки."""
    steps = daily.build_steps("py", "c", weekly=True, dry_run=True)
    assert all("--dry-run" in command for _, command, _ in steps)


def test_clinic_code_reaches_every_step():
    steps = daily.build_steps("py", "two_line", weekly=True)
    for _, command, _ in steps:
        assert command[command.index("--clinic") + 1] == "two_line"


# --- Приём выгрузки листа --------------------------------------------------

VALID_EXPORT = ("ID,Patient Full Name,Date OPU,MII\n"
                "A1,Пациентка,01.06.2026,10\n")


def test_valid_export_is_accepted():
    assert fetch.looks_like_opu_export(VALID_EXPORT) == (True, "")


def test_login_page_is_rejected():
    """Страница логина Google, записанная поверх выгрузки, читалась бы как
    «в таблице не осталось циклов» — самый тихий из возможных отказов."""
    ok, reason = fetch.looks_like_opu_export(
        "<!DOCTYPE html><html><title>Sign in</title></html>")
    assert ok is False and "не тот лист" in reason


def test_export_of_a_different_sheet_is_rejected():
    ok, reason = fetch.looks_like_opu_export("a,b,c\n1,2,3\n")
    assert ok is False and "нет колонок" in reason


def test_empty_export_is_rejected():
    """Заголовок без единой пациентки принимать нельзя: это не «циклов нет»,
    а сорванная выгрузка."""
    ok, reason = fetch.looks_like_opu_export("ID,Patient Full Name,Date OPU\n,,\n")
    assert ok is False and "ни одной строки" in reason


def test_row_with_only_a_name_still_counts_as_an_export():
    ok, _ = fetch.looks_like_opu_export(
        "ID,Patient Full Name,Date OPU\n,Пациентка,01.06.2026\n")
    assert ok is True


# --- Артефакты прогона -----------------------------------------------------

def test_alerts_file_always_has_a_header(tmp_path):
    """Пустой журнал тревог — это заголовок без строк, а не пустой файл.

    Файл без заголовка роняет следующий прогон с EmptyDataError, и происходит это
    на самом обычном пути здоровой лаборатории: сегодня тревог нет, завтра тоже.
    """
    empty = run.alerts_frame([], "production", "two_line")
    assert list(empty.columns) == run.ALERT_COLUMNS

    path = tmp_path / "ews_alerts.csv"
    empty.to_csv(path, index=False, encoding="utf-8-sig")
    assert list(run.read_alerts(path).columns) == run.ALERT_COLUMNS


def test_headerless_legacy_file_is_survived(tmp_path):
    """Файл, оставленный прошлой версией, не должен ронять сегодняшний прогон.

    Потерять оценку из-за вчерашнего пустого файла — обмен, которого делать нельзя.
    """
    path = tmp_path / "ews_alerts.csv"
    path.write_bytes(b"\xef\xbb\xbf\r\n")
    assert run.read_alerts(path).empty


def test_missing_alerts_file_reads_as_no_alerts(tmp_path):
    assert run.read_alerts(tmp_path / "нет.csv").empty


def test_checkpoints_frame_survives_an_empty_cohort():
    """У кадра без циклов нет колонки closure_due вовсе.

    Обычная выборка списком роняла бы запись артефакта на первом же прогоне новой
    клиники — там, где правильный ответ «циклов нет».
    """
    empty = pd.DataFrame(columns=["cycle_uid", "date_opu", "checkpoint"])
    out = run.checkpoints_frame(empty, "production")
    assert list(out.columns)[:4] == run.CHECKPOINT_COLUMNS
    assert out.empty


def test_checkpoints_frame_stamps_the_environment():
    """Каждый артефакт несёт окружение: черновой отчёт обязан быть отличим."""
    frame = pd.DataFrame([{"cycle_uid": "A", "checkpoint": "closure",
                           "date_opu": pd.Timestamp("2026-06-01"),
                           "closure_due": pd.Timestamp("2026-06-07")}])
    out = run.checkpoints_frame(frame, "draft")
    assert set(out["environment"]) == {"draft"}
    assert out["recorded_at"].notna().all()


def test_ok_cycles_never_reach_the_alert_journal():
    """В журнал тревог попадает только то, что сработало."""

    class _Evaluation:
        def __init__(self, level):
            self.level = level
            self.cycle_uid = "A"
            self.closure_date = pd.Timestamp("2026-06-07")
            self.charts = {"BlR": {"crossed": "watch", "statistic": 2.0,
                                   "expected_source": "baseline"}}
            self.contour = "culture"
            self.corroboration = []
            self.corroboration_rank = "none"
            self.miri_status = "pending"
            self.action = "casemix"
            self.verdict_summary = ""
            self.notes = []

    frame = run.alerts_frame([_Evaluation("ok"), _Evaluation("watch")],
                             "production", "two_line")
    assert len(frame) == 1 and frame["level"].iloc[0] == "watch"
    assert frame["alert_id"].iloc[0].startswith("two_line-")


# --- Опорные измерения -----------------------------------------------------

baseline = load_script("ews_baseline")


def _export(clinic, tmp_path, rows: list[dict]):
    """Выгрузка листа с колонками ЭТОЙ клиники: имена берутся из профиля."""
    named = [{clinic.source_column(canonical): value
              for canonical, value in row.items()} for row in rows]
    path = tmp_path / "opu_master.csv"
    pd.DataFrame(named).to_csv(path, index=False, encoding="utf-8")
    return path


POPULATION = [
    {"patient_id": "P1", "date_opu": "01.06.2026", "fertilization_method": "ICSI",
     "occ": "10", "mii": "8", "two_pn": "6", "bl_total": "3", "bl_good": "2"},
    {"patient_id": "P2", "date_opu": "02.06.2026", "fertilization_method": "NO",
     "occ": "0", "mii": "0", "two_pn": "", "bl_total": "0", "bl_good": "0"},
    {"patient_id": "P3", "date_opu": "03.06.2026", "fertilization_method": "Cryo",
     "occ": "9", "mii": "7", "two_pn": "5", "bl_total": "2", "bl_good": "1"},
]


def test_baseline_reads_exclusion_tokens_from_the_clinic_profile(cfg, clinic,
                                                                 tmp_path):
    """Токены исключений живут в профиле клиники, а не в чужом проекте.

    Раньше они читались из `config/columns.yaml` проекта embryo-lab-data. На
    автономной машине его нет по замыслу, `project_dir` пуст, и скрипт падал с
    FileNotFoundError ещё до первого числа (docs/LAB_SITE.md §7) — то есть шаг
    «опорные параметры» на боевой площадке был недоступен вовсе.
    """
    cfg.settings["privacy"] = {**cfg.settings["privacy"],
                               "secret_file": str(tmp_path / "uid.key")}
    assert not cfg.external("embryo_lab_data")["project_dir"]

    population, excluded = baseline.load_lab_population(
        cfg, clinic, _export(clinic, tmp_path, POPULATION))

    # NO-retrieval и freeze-all ушли, обычный цикл остался.
    assert excluded == 2
    assert list(population["fertilization_method"]) == ["ICSI"]


def test_baseline_stops_when_the_exclusion_section_is_missing(cfg, clinic,
                                                              tmp_path):
    """Нет секции — остановка, а не пустая маска.

    Пустая маска не роняет ничего и не печатает ничего: она молча расширяет
    состав циклов в базовых KPI, а через них сдвигает опорные параметры, на
    которых центрированы карты. На боевой выгрузке это разница в 18 циклов из
    108 — ровно тот класс ошибки, который система обязана не допускать молча.
    """
    from ews.clinic import Clinic
    from ews.config import ConfigError

    stripped = Clinic(code=clinic.code, profile=clinic.profile,
                      columns={k: v for k, v in clinic.columns.items()
                               if k != "competency_exclusion"})
    cfg.settings["privacy"] = {**cfg.settings["privacy"],
                               "secret_file": str(tmp_path / "uid.key")}

    with pytest.raises(ConfigError, match="competency_exclusion"):
        baseline.load_lab_population(
            cfg, stripped, _export(stripped, tmp_path, POPULATION))


def test_baseline_dry_run_leaves_the_reference_parameters_alone(cfg, clinic,
                                                                tmp_path):
    """`--dry-run` честный и здесь — иначе посмотреть на пересчёт нельзя.

    Опорные параметры — центр относительного контура, и записанные поверх старых
    они меняют смысл уже откалиброванных порогов. Пока прогон нельзя было
    показать, не записав, единственным способом взглянуть на пересчёт оставалось
    его выполнить.
    """
    out_dir = tmp_path / "baseline"
    out_dir.mkdir()
    cfg.settings["privacy"] = {**cfg.settings["privacy"],
                               "secret_file": str(tmp_path / "uid.key")}
    export = _export(clinic, tmp_path, [
        dict(row, age="33", amh="2.0", occ=row["occ"], mii=row["mii"])
        for row in POPULATION
    ])

    baseline.write_incontrol(cfg, clinic, export, out_dir, dry_run=True)
    assert list(out_dir.iterdir()) == []

    baseline.write_incontrol(cfg, clinic, export, out_dir)
    assert (out_dir / "incontrol_parameters.json").exists()


def test_shipped_profiles_all_declare_the_exclusion_section():
    """Секция обязательна и в шаблоне: новая клиника не заводится без ответа."""
    import yaml

    from ews.clinic import CLINICS_DIR

    for directory in CLINICS_DIR.iterdir():
        if not directory.is_dir():
            continue
        with (directory / "columns.yaml").open(encoding="utf-8") as fh:
            columns = yaml.safe_load(fh) or {}
        assert "competency_exclusion" in columns, directory.name


# --- Журнал тревог: нумерация и происхождение ------------------------------
#
# Три правила, каждое из которых было нарушено на боевой площадке и каждое
# проверялось глазами по CSV, а не тестом.

def _evaluation(uid, level="watch", statistic=2.0):
    class Stub:
        pass
    item = Stub()
    item.cycle_uid = uid
    item.closure_date = pd.Timestamp("2026-09-03")
    item.level = level
    item.charts = {"BlR": {"crossed": "watch", "statistic": statistic,
                           "expected_source": "baseline", "contour": "culture"}}
    item.contour = "culture"
    item.corroboration = []
    item.corroboration_rank = "none"
    item.miri_status = "pending"
    item.action = "casemix"
    item.verdict_summary = "тренд, не провал"
    item.notes = []
    return item


def test_alert_ids_continue_from_the_journal():
    """Номер карточки продолжается от файла, а не начинается заново каждый прогон.

    Раньше номер брался из позиции цикла в списке оценок ЭТОГО прогона, и в
    инкрементальном режиме счётчик сбрасывался на 1: в боевом журнале лежали две
    разные карточки `two_line-00001`. Идентификатор, на который ссылается разбор,
    обязан быть уникален.
    """
    previous = pd.DataFrame([{"alert_id": "two_line-00007"},
                             {"alert_id": "two_line-00088"}])
    assert run.next_alert_number(previous, "two_line") == 89

    fresh = run.alerts_frame([_evaluation("CYC-NEW1"), _evaluation("CYC-NEW2")],
                             "production", "two_line",
                             start=run.next_alert_number(previous, "two_line"))
    assert list(fresh["alert_id"]) == ["two_line-00089", "two_line-00090"]


def test_empty_journal_starts_from_one():
    assert run.next_alert_number(pd.DataFrame(), "two_line") == 1
    assert run.next_alert_number(pd.DataFrame(columns=["alert_id"]), "two_line") == 1


def test_card_records_where_it_came_from():
    """Реконструкция `--replay` и реальное срабатывание не смешиваются.

    Журнал ведут ради одного — судить о результативности ПРОСПЕКТИВНОГО
    отслеживания. Карточка, восстановленная прогоном по всей истории, говорит, что
    система сказала БЫ, зная будущее; ранним обнаружением она не является и стоять
    рядом с реальной без пометки не может.
    """
    live = run.alerts_frame([_evaluation("CYC-A")], "production", "two_line")
    replayed = run.alerts_frame([_evaluation("CYC-A")], "production", "two_line",
                                raised_in=run.RAISED_IN_REPLAY)
    assert live.iloc[0]["raised_in"] == run.RAISED_IN_RUN
    assert replayed.iloc[0]["raised_in"] == run.RAISED_IN_REPLAY


def test_ok_cycles_do_not_consume_numbers():
    """Спокойный цикл номер не тратит: иначе в журнале зияли бы дыры."""
    frame = run.alerts_frame([_evaluation("CYC-OK", level="ok"),
                              _evaluation("CYC-RAISED")],
                             "production", "two_line", start=5)
    assert list(frame["alert_id"]) == ["two_line-00005"]


def test_replay_does_not_launder_real_alerts_into_reconstructions():
    """Пересборка истории не отменяет того, что система сказала в тот день.

    `--replay` пишет журнал с нуля и пометил бы ВСЕ карточки как реконструкцию,
    включая три, поднятые живыми прогонами. Это стёрло бы единственное
    свидетельство о результативности проспективного контура. Идентификатор
    переносится по той же причине: на него ссылается разбор.
    """
    existing = pd.DataFrame([
        {"alert_id": "two_line-00015", "cycle_uid": "CYC-LIVE", "chart": "BlR",
         "raised_at": "2026-08-18", "raised_in": "прогон", "upgraded_at": ""},
        {"alert_id": "two_line-00014", "cycle_uid": "CYC-BACK", "chart": "FR",
         "raised_at": "2026-08-14", "raised_in": "replay", "upgraded_at": ""},
    ])
    rebuilt = pd.DataFrame([
        {"alert_id": "two_line-00018", "cycle_uid": "CYC-LIVE", "chart": "BlR",
         "raised_at": "2026-08-18", "raised_in": "replay", "upgraded_at": ""},
        {"alert_id": "two_line-00019", "cycle_uid": "CYC-BACK", "chart": "FR",
         "raised_at": "2026-08-14", "raised_in": "replay", "upgraded_at": ""},
        {"alert_id": "two_line-00020", "cycle_uid": "CYC-NEW", "chart": "MIIR",
         "raised_at": "2026-09-09", "raised_in": "replay", "upgraded_at": ""},
    ])
    out = run.carry_over(rebuilt, existing).set_index("cycle_uid")

    assert out.loc["CYC-LIVE", "raised_in"] == "прогон"
    assert out.loc["CYC-LIVE", "alert_id"] == "two_line-00015"
    assert out.loc["CYC-BACK", "alert_id"] == "two_line-00014"
    # Карточки, которой в старом журнале не было, пересборка не трогает.
    assert out.loc["CYC-NEW", "alert_id"] == "two_line-00020"
    assert out.loc["CYC-NEW", "raised_in"] == "replay"


def test_carry_over_survives_an_empty_or_legacy_journal():
    """Пустой журнал и файл без новых колонок пересборку не роняют."""
    rebuilt = pd.DataFrame([{"alert_id": "two_line-00001", "cycle_uid": "C",
                             "chart": "FR", "raised_at": "2026-01-01",
                             "raised_in": "replay", "upgraded_at": ""}])
    assert run.carry_over(rebuilt, pd.DataFrame()).equals(rebuilt)
    legacy = pd.DataFrame([{"alert_id": "two_line-00007", "cycle_uid": "C",
                            "chart": "FR", "raised_at": "2026-01-01"}])
    out = run.carry_over(rebuilt, legacy)
    assert out.iloc[0]["alert_id"] == "two_line-00007"
    assert out.iloc[0]["raised_in"] == "replay"


def test_carry_over_does_not_invent_an_upgrade_from_an_empty_cell():
    """Пустое `upgraded_at` из CSV (NaN) остаётся пустым, а не становится "nan".

    2026-10-02 пересборка приписала шести перенесённым карточкам «доведено средой
    nan»: доводки не было, а панель её показывала.
    """
    rebuilt = pd.DataFrame([{"alert_id": "two_line-00030", "cycle_uid": "C",
                             "chart": "FR", "raised_at": "2026-01-01",
                             "raised_in": "replay", "upgraded_at": ""}])
    existing = pd.DataFrame([{"alert_id": "two_line-00007", "cycle_uid": "C",
                              "chart": "FR", "raised_at": "2026-01-01",
                              "raised_in": "прогон", "upgraded_at": float("nan")}])
    out = run.carry_over(rebuilt, existing)
    assert out.iloc[0]["alert_id"] == "two_line-00007"
    assert out.iloc[0]["raised_in"] == "прогон"
    assert out.iloc[0]["upgraded_at"] == ""


def test_interrupted_chain_still_reaches_the_journal(monkeypatch, tmp_path):
    """Прерванный прогон отчитывается о том, что успел сделать.

    2026-09-03 11:24 ручной прогон был остановлен Ctrl+C на мягком шаге. Он успел
    поднять тревогу (watch FR), но до записи в Teable управление не дошло, и в
    журнале прогонов этой работы нет вовсе. Прогон, который что-то сделал и не
    отчитался, хуже непрошедшего: его молчание неотличимо от «задача не сработала».
    """
    calls = []

    def fake_step(label, command, soft):
        if label == "прогон":
            return {"step": label, "returncode": 0, "ok": True, "soft": soft,
                    "seconds": 1.0}
        raise KeyboardInterrupt

    def fake_journal(args, run_id, started_at, results, *, notes=None):
        calls.append((results, list(notes or [])))

    monkeypatch.setattr(daily, "run_step", fake_step)
    monkeypatch.setattr(daily, "journal", fake_journal)
    monkeypatch.setattr(daily, "ROOT", tmp_path)
    monkeypatch.setattr(daily.sys, "argv",
                        ["ews_daily.py", "--clinic", "two_line"])

    code = daily.main()

    assert calls, "журнал не был вызван — прерванный прогон остался бы невидимым"
    _, notes = calls[0]
    assert any("прерван" in note for note in notes)
    assert code == 1


def test_last_run_reads_both_russian_editions_of_schtasks():
    """Windows 11 26200 пишет «прошлого/Прошлый», а не «последнего/Последний».

    2026-10-02 --status печатал «— КОД —» у исправных задач: имя поля не узналось.
    """
    new = {"Время прошлого запуска": "01.10.2026 15:45:00", "Прошлый результат": "0"}
    old = {"Время последнего запуска": "01.10.2026 15:45:00", "Последний результат": "0"}
    assert schedule._last_run(new) == "01.10.2026 15:45:00  ok"
    assert schedule._last_run(old) == "01.10.2026 15:45:00  ok"
    assert "не прочитан" in schedule._last_run({"Время прошлого запуска": "01.10.2026"})
