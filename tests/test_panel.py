"""HTML-панель: то, что человек прочитает вместо кода.

Проверяется не вёрстка, а три вещи, каждая из которых является правилом проекта:
PHI на панель не попадает; артефакт несёт штамп окружения и площадки; закрытый гейт
alert объявлен прямо на странице, а не подразумевается.

Панель читает занятый человек. Ошибка здесь не роняет прогон — она приводит к
неверному решению, что дороже.
"""
import pandas as pd
import pytest

from conftest import cycle_frame, cycle_row, make_config

from ews import cycles, evaluate, panel


@pytest.fixture
def rendered(cfg, clinic, baselines, tmp_path):
    """Панель, собранная по одной сработавшей и одной спокойной тревоге."""
    return _render(cfg, clinic, baselines, tmp_path)


def journal_row(cycle_uid="CYC-AAA111", *, level=evaluate.LEVEL_WATCH,
                raised_at="2026-06-07", raised_in="прогон", alert_id="two_line-00001",
                miri_status="pending", upgraded_at=""):
    """Строка накопленного журнала — то, из чего панель теперь строит таблицу."""
    return {"alert_id": alert_id, "cycle_uid": cycle_uid, "raised_at": raised_at,
            "level": level, "chart": "BlR", "action": "casemix",
            "verdict": "тренд, не провал", "miri_status": miri_status,
            "raised_in": raised_in, "upgraded_at": upgraded_at}


def _render(cfg, clinic, baselines, tmp_path, *, level=evaluate.LEVEL_WATCH,
            evaluations=None, journal=None):
    frame = cycle_frame([cycle_row("CYC-AAA111"), cycle_row("CYC-BBB222")])
    holder = cycles.CycleFrame(data=frame, source="test",
                               loaded_at=pd.Timestamp("2026-06-20").to_pydatetime())
    evaluation = evaluate.CycleEvaluation(
        cycle_uid="CYC-AAA111", closure_date=pd.Timestamp("2026-06-07"), level=level)
    evaluation.charts = {"BlR": {"crossed": "watch", "statistic": 2.0, "peak": 2.0,
                                 "expected_source": "baseline"}}
    evaluation.verdict_summary = "тренд, не провал"
    evaluation.action = "casemix"
    states = {"BlR": evaluate.ChartState(metric="BlR", statistic=2.0, observations=40,
                                         peak=3.1, watch_episodes=1)}
    states["BlR"].record("2026-06-07", 2.0)
    if journal is None:
        journal = pd.DataFrame([journal_row(level=level)])
    if evaluations is None:
        evaluations = [evaluation]
    path = tmp_path / "EWS_Panel.html"
    panel.render(cfg, clinic, holder, evaluations, states, baselines,
                 cfg.settings["thresholds"], path, alerts_journal=journal)
    return path.read_text(encoding="utf-8")


def test_panel_is_written_atomically(cfg, clinic, baselines, tmp_path):
    """Полуфабрикатов на диске не остаётся: панель читают в момент прогона."""
    _render(cfg, clinic, baselines, tmp_path)
    assert (tmp_path / "EWS_Panel.html").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_panel_carries_no_phi(rendered):
    """На панель уходит только `cycle_uid`.

    Проверять по отсутствию конкретного имени бессмысленно — проверяется, что
    идентификатор цикла есть, а полей PHI из профиля клиники нет вовсе.
    """
    assert "CYC-AAA111" in rendered
    for forbidden in ("Patient Full Name", "patient_name", "ДР", "DOB", "паспорт"):
        assert forbidden not in rendered


def test_production_panel_is_marked_as_the_lab(rendered):
    assert "лаборатория" in rendered and "площадка lab" in rendered


def test_draft_panel_is_unmistakable(settings, clinic, baselines, tmp_path):
    """Черновой стенд обязан быть отличим без чтения содержимого.

    HTML-панель черновика визуально неотличима от лабораторной — ровно тот класс
    ошибки, ради предотвращения которого система и строится.
    """
    cfg = make_config(settings, tmp_path, environment="draft", site="home")
    html = _render(cfg, clinic, baselines, tmp_path)
    assert "ЧЕРНОВОЙ СТЕНД" in html
    assert "площадка home" in html


def test_blocked_gate_is_announced_on_the_page(settings, clinic, baselines, tmp_path):
    """Закрытый гейт объявлен прямо на панели, а не подразумевается.

    Читающий обязан знать, что watch перед ним мог бы быть alert в боевом режиме.
    """
    settings["thresholds"]["calibrated"] = False
    cfg = make_config(settings, tmp_path)
    html = _render(cfg, clinic, baselines, tmp_path)
    assert "Уровень alert заблокирован" in html


def test_open_gate_says_nothing_about_blocking(rendered):
    assert "Уровень alert заблокирован" not in rendered


def test_excluded_metric_explains_itself(rendered):
    """Метрика без карты объясняет, почему её там нет.

    Иначе отсутствие OCCR на панели читается как недосмотр, а не как решение.
    """
    assert "OCCR" in rendered and "без карты" in rendered


def test_trajectory_is_shown_not_just_the_last_number(rendered):
    """Панель показывает траекторию и пик, а не только «сейчас».

    Снимок на конец периода почти всегда близок к нулю: карта обнуляется после
    сигнала и разряжается в спокойные периоды. «0.00 при пороге 1.78» читается как
    «мы далеко от беды», хотя означает лишь «прямо сейчас ничего не копится».
    """
    assert "<polyline" in rendered
    assert "пик" in rendered


def test_empty_run_says_that_silence_is_normal(cfg, clinic, baselines, tmp_path):
    """Отсутствие тревог — нормальный и самый частый исход, и так и написано.

    Система, сообщающая что-то каждый день, перестаёт читаться.
    """
    holder = cycles.CycleFrame(
        data=cycle_frame([cycle_row("CYC-AAA111")]), source="test",
        loaded_at=pd.Timestamp("2026-06-20").to_pydatetime())
    path = tmp_path / "EWS_Panel.html"
    panel.render(cfg, clinic, holder, [], {}, baselines,
                 cfg.settings["thresholds"], path)
    html = path.read_text(encoding="utf-8")
    assert "Тревог нет" in html


def test_alert_level_is_visually_distinct(cfg, clinic, baselines, tmp_path):
    html = _render(cfg, clinic, baselines, tmp_path, level=evaluate.LEVEL_ALERT)
    assert "lvl-alert" in html and "ALERT" in html


def test_environment_note_about_missing_logs_is_always_there(rendered):
    """Пустое поле среды означает «логи ещё не выгружены», а не «нарушений не было».

    Эта оговорка обязана стоять рядом с журналом тревог: без неё пустая колонка
    читается как подтверждённое отсутствие событий.
    """
    assert "не «нарушений инкубации не было»" in rendered


def test_panel_declares_itself_a_research_tool(rendered):
    """Наблюдения и поводы для разбора — не диагнозы и не оценка специалиста."""
    assert "не диагнозы" in rendered


# --- Журнал тревог: накопленный, а не «за сегодня» --------------------------

def test_journal_survives_a_quiet_run(cfg, clinic, baselines, tmp_path):
    """Ни один цикл сегодня не закрылся — журнал всё равно на месте.

    Это главная поломка отчёта, найденная на боевой площадке: раздел строился из
    `evaluations`, то есть из циклов, оценённых В ЭТОМ прогоне. При суточной
    каденции это 0-1 цикл, и панель печатала «Журнал тревог (0) — Тревог нет» при
    17 карточках в файле, три из них за последний месяц. Отчёт, каждый день
    сообщающий «всё чисто», — не мониторинг, а его имитация.
    """
    journal = pd.DataFrame([
        journal_row("CYC-OLD1", raised_at="2026-08-18", alert_id="two_line-00001"),
        journal_row("CYC-OLD2", raised_at="2026-08-26", alert_id="two_line-00002"),
    ])
    html = _render(cfg, clinic, baselines, tmp_path, evaluations=[], journal=journal)
    assert "всего в журнале 2" in html
    assert "CYC-OLD1" in html and "CYC-OLD2" in html
    assert "Тревог нет за всё время наблюдения" not in html


def test_journal_marks_reconstructed_cards(cfg, clinic, baselines, tmp_path):
    """Реконструкция `--replay` названа реконструкцией прямо в таблице."""
    journal = pd.DataFrame([
        journal_row("CYC-LIVE", raised_at="2026-09-03", raised_in="прогон"),
        journal_row("CYC-BACK", raised_at="2026-03-16", raised_in="replay"),
    ])
    html = _render(cfg, clinic, baselines, tmp_path, evaluations=[], journal=journal)
    assert "реконструкция" in html
    assert "проспективно за всё время: 1 из 2" in html


def test_journal_shows_deferred_upgrade(cfg, clinic, baselines, tmp_path):
    """Доводка средой видна на карточке: когда именно её подняли."""
    journal = pd.DataFrame([journal_row("CYC-UP", level="alert",
                                        miri_status="enriched",
                                        upgraded_at="2026-09-14T15:48:00")])
    html = _render(cfg, clinic, baselines, tmp_path, evaluations=[], journal=journal)
    assert "доведено средой 2026-09-14" in html


def test_journal_reads_a_file_without_the_new_columns(cfg, clinic, baselines, tmp_path):
    """Журнал прошлого поколения не роняет панель.

    Один устаревший файл на диске не должен лишать лабораторию отчёта целиком.
    """
    old = pd.DataFrame([{"alert_id": "two_line-00007", "cycle_uid": "CYC-OLD",
                         "raised_at": "2026-01-21", "level": "watch", "chart": "FR",
                         "action": "casemix", "verdict": "тренд, не провал",
                         "miri_status": "unavailable"}])
    html = _render(cfg, clinic, baselines, tmp_path, evaluations=[], journal=old)
    assert "всего в журнале 1" in html and "CYC-OLD" in html


def test_empty_journal_says_so_honestly(cfg, clinic, baselines, tmp_path):
    html = _render(cfg, clinic, baselines, tmp_path, evaluations=[],
                   journal=pd.DataFrame())
    assert "Тревог нет за всё время наблюдения" in html
