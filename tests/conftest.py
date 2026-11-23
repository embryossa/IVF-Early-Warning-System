"""Общие приспособления тестов EWS.

Путь к `src` добавляется здесь один раз: пакет `ews` нигде не устанавливается
(правило проекта), и без этого каждый файл повторял бы одну и ту же вставку.

**Настройки берутся из настоящего `config/settings.yaml`**, а не из выдуманного
словаря. Причина не в лени: это файл МЕТОДА — он одинаков для всех клиник, едет в
git как есть и при переезде не правится. Тест на выдуманных настройках продолжил бы
проходить и после того, как из settings.yaml исчез ключ, который код индексирует
жёстко (`cfg.settings["localisation"]["contours"]` и подобные), то есть проверял бы
не ту систему, которая работает.

Машинный слой, наоборот, синтетический и всегда во временном каталоге:
`machine.local.yaml` в git не попадает, на чужой машине его нет, а тест, зависящий
от него, был бы непереносимым — ровно то, против чего написан `docs/PORTING.md`.
"""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ews.clinic import Clinic, load_clinic  # noqa: E402
from ews.config import CONFIG_DIR, Config  # noqa: E402

CLINIC_CODE = "two_line"


def load_script(name: str):
    """Скрипт из `scripts/` как импортируемый модуль.

    Скрипты пакетом не являются и обычным импортом не берутся — а именно в них
    живут решения, ошибиться в которых дороже всего: состав цепочки прогона, дни
    расписания, признание ответа сервера настоящей выгрузкой. Оставлять их без
    проверки только потому, что это «скрипты», значит не проверять систему.
    """
    import importlib.util

    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"ews_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def shipped_settings() -> dict:
    """Метод как он есть в репозитории. Только для чтения."""
    with (CONFIG_DIR / "settings.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@pytest.fixture
def settings(shipped_settings) -> dict:
    """Копия метода на один тест плюс откалиброванные пороги.

    Пороги подставляются здесь, а не читаются с диска: `thresholds.<clinic>.local.yaml`
    генерируется бэкфитом и в git не едет — тест, зависящий от него, проходил бы
    только на машине, где бэкфит уже отработал.
    """
    out = copy.deepcopy(shipped_settings)
    out["thresholds"] = {
        **out.get("thresholds", {}),
        "calibrated": True,
        "calibrated_environment": "production",
        "h_watch": {"FR": 1.2, "BlR": 1.8, "MIIR": 1.7},
        "h_alert": {"FR": 7.4, "BlR": 10.8, "MIIR": 10.0},
    }
    return out


def make_config(settings: dict, data_dir: Path, *, environment: str = "production",
                site: str = "lab", secrets: dict | None = None) -> Config:
    return Config(
        settings=settings,
        machine={
            "environment": environment, "site": site,
            "paths": {"data_dir": str(data_dir)},
            # Секция машины синтетическая, но по форме настоящая: скрипты читают
            # её через cfg.external() и падают на пустой, а падать они должны на
            # незаполненной боевой, а не на тестовой.
            "teable": {"url": "http://127.0.0.1:3000", "base_id": "bseTEST",
                       "opu_table_id": "tblOPU"},
            "google_sheet": {"sheet_id": "sheet", "gid": "0",
                             "export_path": str(data_dir / "opu_master.csv")},
            "miri_logs": {"dump_dir": "", "incubator": "6391"},
            "embryo_lab_data": {"project_dir": "", "clinic": CLINIC_CODE},
        },
        secrets=secrets or {},
        clinic_code=CLINIC_CODE,
    )


@pytest.fixture
def cfg(settings, tmp_path) -> Config:
    """Боевое окружение с собственной калибровкой: гейт alert открыт."""
    return make_config(settings, tmp_path)


@pytest.fixture
def draft_cfg(settings, tmp_path) -> Config:
    """Черновой стенд: alert недоступен ни при каких обстоятельствах."""
    return make_config(settings, tmp_path, environment="draft", site="home")


@pytest.fixture
def clinic(cfg) -> Clinic:
    """Настоящий профиль `two_line`: набор карт и опорное окно — как в бою."""
    return load_clinic(cfg, CLINIC_CODE)


# --- Заглушка Teable -------------------------------------------------------

class FakeTeable:
    """Клиент Teable, который помнит, о чём его спрашивали.

    Проекция запоминается не для полноты картины: именно по ней проверяется, что
    ingest не запрашивает PHI. По одному лишь отсутствию ФИО в готовом кадре этого
    не увидеть — колонка могла приехать по сети и быть отброшенной уже после.
    """

    def __init__(self, records: list[dict] | None = None) -> None:
        self.records = list(records or [])
        self.projections: list[list[str]] = []
        self.created: list[tuple[str, list[dict]]] = []
        self.updated: list[tuple[str, list[tuple[str, dict]]]] = []

    def all_records(self, table_id: str, projection=None) -> list[dict]:
        self.projections.append(list(projection or []))
        return self.records

    def create_records(self, table_id: str, rows: list[dict]) -> None:
        self.created.append((table_id, list(rows)))

    def update_records(self, table_id: str, pairs) -> None:
        self.updated.append((table_id, list(pairs)))


def teable_record(cycle_uid: str, date_opu: str | None = "2026-06-01",
                  record_id: str | None = None, **fields) -> dict:
    """Запись таблицы OPU в том виде, в каком её отдаёт Teable (UTC-даты)."""
    payload: dict = {"cycle_uid": cycle_uid}
    if date_opu is not None:
        payload["Дата OPU"] = f"{date_opu}T00:00:00.000Z"
    payload.update(fields)
    return {"id": record_id or f"rec{cycle_uid}", "fields": payload}


@pytest.fixture
def fake_teable() -> type[FakeTeable]:
    return FakeTeable


# --- Канонический кадр -----------------------------------------------------

def cycle_row(uid: str, *, date_opu: str = "2026-06-01", checkpoint: str = "closure",
              closure_offset_days: int = 6, **values) -> dict:
    """Строка канонического кадра: то, что видит слой оценки (PHI здесь уже нет)."""
    opu = pd.Timestamp(date_opu)
    row = {
        "cycle_uid": uid,
        "date_opu": opu,
        "closure_due": opu + pd.Timedelta(days=closure_offset_days),
        "checkpoint": checkpoint,
        "age": 33.0, "amh": 2.0, "attempt_no": 1.0,
        "follicles": 15.0, "occ": 12.0, "mii": 10.0,
        "two_pn": 8.0, "three_pn": 0.0, "one_pn": 0.0, "zero_pn": 0.0,
        "degrade": 0.0, "cleavage_d3": 8.0, "bl_total": 5.0, "bl_good": 3.0,
        "frozen": 3.0, "transferred": 1.0, "transfer_outcome": None,
        "prai": None, "digital_twin": None,
    }
    row.update(values)
    return row


def cycle_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def baselines() -> dict[str, dict]:
    """Опорные параметры в том виде, в каком их отдаёт incontrol_parameters.json."""
    return {
        "MIIR": {"metric": "MIIR", "rate": 0.80, "direction": "lower",
                 "competency": 0.75, "benchmark": 0.85, "status": "competency"},
        "FR": {"metric": "FR", "rate": 0.80, "direction": "lower",
               "competency": 0.65, "benchmark": 0.75, "status": "benchmark"},
        "BlR": {"metric": "BlR", "rate": 0.60, "direction": "lower",
                "competency": 0.40, "benchmark": 0.60, "status": "benchmark"},
    }


@pytest.fixture
def now() -> datetime:
    """Фиксированное «сегодня»: чекпойнты зависят от календаря."""
    return datetime(2026, 6, 20, 15, 45)
