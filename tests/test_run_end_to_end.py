"""Сквозной прогон `ews_run.py`: от записи Teable до артефактов на диске.

Единственный тест, который проходит весь суточный путь целиком, и потому
единственный, который ловит отказы на стыках — там, где каждый слой по отдельности
исправен. Все четыре падения, найденные при разборе (пустой кадр в приоре, в
медленном контуре, в сортировке закрытых циклов и в чтении вчерашнего пустого
журнала тревог), лежали именно на стыках и по отдельным модульным тестам не
воспроизводились.

Внешние границы заменены заглушками — Teable и модель конверта, — но НЕ логика:
считает, решает и пишет настоящий код прогона.
"""
import json
import sys

import pandas as pd
import pytest

from conftest import FakeTeable, load_script, teable_record

run = load_script("ews_run")

# Цикл, закрывшийся с нулевым выходом бластоцист: 0 из 8 зигот при базовой доле 0.60.
FAILING = {"OCC": 12, "MII": 10, "2pN": 8, "Бласт": 0, "Хор.бласт": 0,
           "Дробление D3": 8, "Возраст": 33, "AMH": 2.0, "Криоконсервировано": 0}
HEALTHY = {**FAILING, "Бласт": 6, "Хор.бласт": 4, "Криоконсервировано": 4}


def _write_baseline(cfg, clinic) -> None:
    """Опорные параметры на диске: без них прогон отказывается стартовать."""
    path = clinic.data_dir(cfg) / "baseline" / "incontrol_parameters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "reference_window": {"start": "2026-05-01", "end": "2026-09-01"},
        "metrics": [
            {"metric": "MIIR", "rate": 0.80, "direction": "lower", "status": "benchmark",
             "competency": 0.75, "benchmark": 0.85, "ci_low": 0.75, "ci_high": 0.85},
            {"metric": "FR", "rate": 0.80, "direction": "lower", "status": "benchmark",
             "competency": 0.65, "benchmark": 0.75, "ci_low": 0.75, "ci_high": 0.85},
            {"metric": "BlR", "rate": 0.60, "direction": "lower", "status": "benchmark",
             "competency": 0.40, "benchmark": 0.60, "ci_low": 0.55, "ci_high": 0.65},
        ],
    }, ensure_ascii=False), encoding="utf-8")


def _invoke(monkeypatch, cfg, clinic, records, *argv) -> int:
    monkeypatch.setattr(run, "load", lambda code=None: cfg)
    monkeypatch.setattr(run, "load_clinic", lambda config, code=None: clinic)
    monkeypatch.setattr(run, "embryo_lab_data_on_path", lambda config: None)
    monkeypatch.setattr(run.teable_client, "connect",
                        lambda config: FakeTeable(records))
    monkeypatch.setattr(sys, "argv", ["ews_run.py", "--clinic", clinic.code, *argv])
    return run.main()


def _artifacts(cfg, clinic) -> dict:
    directory = clinic.data_dir(cfg)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    return {"dir": directory, "manifest": manifest,
            "alerts": pd.read_csv(directory / "ews_alerts.csv", encoding="utf-8-sig"),
            "checkpoints": pd.read_csv(directory / "ews_checkpoints.csv",
                                       encoding="utf-8-sig")}


# --- Пустая когорта --------------------------------------------------------

def test_run_on_an_empty_mirror_reports_nothing_to_do(monkeypatch, cfg, clinic, capsys):
    """Первый прогон новой клиники — это «циклов нет», а не падение.

    Пустое зеркало выглядит именно так: до первой синхронизации, при сорванной
    выгрузке, на только что заведённой базе. Прогон обязан дойти до конца и
    записать артефакты, иначе система не может стартовать вовсе.
    """
    _write_baseline(cfg, clinic)
    assert _invoke(monkeypatch, cfg, clinic, []) == 0
    out = _artifacts(cfg, clinic)
    assert out["manifest"]["cycles_loaded"] == 0
    assert out["manifest"]["cycles_evaluated"] == 0
    assert out["alerts"].empty and list(out["alerts"].columns) == run.ALERT_COLUMNS
    assert (out["dir"] / "EWS_Panel.html").exists()
    assert "циклов загружено: 0" in capsys.readouterr().out


def test_second_empty_run_survives_the_first_ones_files(monkeypatch, cfg, clinic):
    """Вчерашний пустой журнал тревог не роняет сегодняшний прогон.

    Самый обычный путь здоровой лаборатории — два спокойных дня подряд, — и на
    нём система падала при чтении собственного файла.
    """
    _write_baseline(cfg, clinic)
    assert _invoke(monkeypatch, cfg, clinic, []) == 0
    assert _invoke(monkeypatch, cfg, clinic, []) == 0


# --- Обычный прогон --------------------------------------------------------

def test_closed_cycles_are_evaluated_and_stamped(monkeypatch, cfg, clinic):
    """Оцениваются только закрытые циклы, и каждый артефакт несёт окружение."""
    _write_baseline(cfg, clinic)
    records = [
        teable_record("CYC-OPU", date_opu="2026-08-25", **{"Возраст": 33, "ОКК": 12}),
        teable_record("CYC-CLOSED", date_opu="2026-08-01", **HEALTHY),
    ]
    assert _invoke(monkeypatch, cfg, clinic, records) == 0
    out = _artifacts(cfg, clinic)
    assert out["manifest"]["cycles_loaded"] == 2
    assert out["manifest"]["cycles_evaluated"] == 1
    assert out["manifest"]["environment"] == "production"
    assert out["manifest"]["site"] == "lab"
    assert set(out["checkpoints"]["environment"]) == {"production"}


def test_alert_journal_is_appended_not_rewritten(monkeypatch, cfg, clinic):
    """Инкрементальный прогон дописывает тревоги, а не переписывает журнал.

    Оцениваются только новые циклы; перезапись стёрла бы всю историю разбора.
    """
    _write_baseline(cfg, clinic)
    first = [teable_record(f"CYC-{i}", date_opu=f"2026-08-0{i + 1}", **FAILING)
             for i in range(3)]
    assert _invoke(monkeypatch, cfg, clinic, first) == 0
    raised = len(_artifacts(cfg, clinic)["alerts"])
    assert raised >= 1, "три подряд провальных цикла обязаны сдвинуть карту"

    second = first + [teable_record("CYC-NEW", date_opu="2026-08-10", **FAILING)]
    assert _invoke(monkeypatch, cfg, clinic, second) == 0
    assert len(_artifacts(cfg, clinic)["alerts"]) >= raised


def test_evaluated_cycles_are_not_counted_twice(monkeypatch, cfg, clinic):
    """Повторный прогон по тем же данным ничего не оценивает заново.

    Иначе каждая суточная задача прокручивала бы всю историю, и число наблюдений
    карты росло бы вдвое за прогон.
    """
    _write_baseline(cfg, clinic)
    records = [teable_record(f"CYC-{i}", date_opu=f"2026-08-0{i + 1}", **HEALTHY)
               for i in range(3)]
    _invoke(monkeypatch, cfg, clinic, records)
    state_path = clinic.data_dir(cfg) / "state" / "ews_charts_state.json"
    after_first = json.loads(state_path.read_text(encoding="utf-8"))

    assert _invoke(monkeypatch, cfg, clinic, records) == 0
    after_second = json.loads(state_path.read_text(encoding="utf-8"))
    assert (after_first["charts"]["BlR"]["observations"]
            == after_second["charts"]["BlR"]["observations"])
    assert _artifacts(cfg, clinic)["manifest"]["cycles_evaluated"] == 0


def test_replay_rebuilds_the_charts_from_scratch(monkeypatch, cfg, clinic):
    """`--replay` пересобирает историю с нуля, а не добавляет её к прежней."""
    _write_baseline(cfg, clinic)
    records = [teable_record(f"CYC-{i}", date_opu=f"2026-08-0{i + 1}", **HEALTHY)
               for i in range(3)]
    _invoke(monkeypatch, cfg, clinic, records)
    state_path = clinic.data_dir(cfg) / "state" / "ews_charts_state.json"
    incremental = json.loads(state_path.read_text(encoding="utf-8"))

    assert _invoke(monkeypatch, cfg, clinic, records, "--replay") == 0
    replayed = json.loads(state_path.read_text(encoding="utf-8"))
    assert (incremental["charts"]["BlR"]["observations"]
            == replayed["charts"]["BlR"]["observations"])
    assert _artifacts(cfg, clinic)["manifest"]["mode"] == "replay"


def test_dry_run_writes_nothing(monkeypatch, cfg, clinic):
    """`--dry-run` честный: возвращается до записи любого артефакта."""
    _write_baseline(cfg, clinic)
    records = [teable_record("CYC-1", date_opu="2026-08-01", **FAILING)]
    assert _invoke(monkeypatch, cfg, clinic, records, "--dry-run") == 0
    directory = clinic.data_dir(cfg)
    assert not (directory / "run_manifest.json").exists()
    assert not (directory / "ews_alerts.csv").exists()
    assert not (directory / "state" / "ews_charts_state.json").exists()


# --- Отказы, которые обязаны быть внятными --------------------------------

def test_missing_baseline_stops_the_run_with_an_instruction(monkeypatch, cfg, clinic,
                                                            capsys):
    """Без опорных параметров прогон не начинается и говорит, что запустить.

    Считать по несуществующему центру нельзя, а молча пропустить оценку — значит
    отчитаться об успехе, которого не было.
    """
    assert _invoke(monkeypatch, cfg, clinic, []) == 2
    assert "ews_baseline.py" in capsys.readouterr().out


def test_missing_table_id_stops_the_run(monkeypatch, cfg, clinic, capsys):
    _write_baseline(cfg, clinic)
    cfg.machine["teable"]["opu_table_id"] = ""
    assert _invoke(monkeypatch, cfg, clinic, []) == 2
    assert "ews_bootstrap_opu.py" in capsys.readouterr().out


# --- Гейт на сквозном пути -------------------------------------------------

def test_draft_run_cannot_write_an_alert(monkeypatch, settings, clinic, tmp_path):
    """На черновом стенде в журнале тревог не может появиться alert.

    Проверяется на сквозном пути, а не только в оценке: гейт обязан держаться до
    самого артефакта, который кто-то потом прочитает.
    """
    from conftest import make_config

    cfg = make_config(settings, tmp_path, environment="draft", site="home")
    _write_baseline(cfg, clinic)
    records = [teable_record(f"CYC-{i}", date_opu=f"2026-08-0{i + 1}", **FAILING)
               for i in range(6)]
    assert _invoke(monkeypatch, cfg, clinic, records) == 0
    out = _artifacts(cfg, clinic)
    assert "alert" not in set(out["alerts"]["level"])
    assert out["manifest"]["alert_gate"] != "open"
    assert set(out["alerts"]["environment"]) <= {"draft"}


def test_manifest_records_the_alert_gate(monkeypatch, cfg, clinic):
    """Читающий журнал обязан видеть, мог ли этот прогон вообще поднять alert."""
    _write_baseline(cfg, clinic)
    assert _invoke(monkeypatch, cfg, clinic, []) == 0
    assert _artifacts(cfg, clinic)["manifest"]["alert_gate"] == "open"


def test_manifest_counts_context_escalations(monkeypatch, cfg, clinic):
    """Маршрут «watch + сильное подтверждение» виден в манифесте отдельно.

    Частота alert по калибровке считана только для пересечений порога; тревоги,
    поднятые обстановкой, обязаны считаться отдельно, иначе сравнивать не с чем.
    """
    _write_baseline(cfg, clinic)
    assert _invoke(monkeypatch, cfg, clinic, []) == 0
    assert _artifacts(cfg, clinic)["manifest"]["escalated_by_context"] == 0
