"""Гарантии конверта: заморозка и запрет заглядывать в будущее.

Обе проверяются кодом, а не соглашением. Это ровно те два свойства, потеря которых
превращает раннее предупреждение в подгонку задним числом, и заметить такую потерю
по выходным числам невозможно.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import envelope  # noqa: E402
from ews.clinic import Clinic  # noqa: E402
from ews.config import Config, ConfigError  # noqa: E402


@pytest.fixture
def clinic():
    return Clinic(code="test", profile={"name": "Test"}, columns={})


@pytest.fixture
def cfg(tmp_path):
    return Config(settings={}, machine={"environment": "draft",
                                        "paths": {"data_dir": str(tmp_path)}},
                  secrets={}, clinic_code="test")


def _row(uid="CYC-1", stage="mii", p50=8.0):
    return pd.DataFrame([{
        "cycle_uid": uid,
        f"TPFN3_{stage}_p10": p50 - 2, f"TPFN3_{stage}_p50": p50,
        f"TPFN3_{stage}_p90": p50 + 2,
    }])


def test_envelope_is_frozen(cfg, clinic):
    """Повторная регистрация той же стадии — ошибка, а не обновление."""
    envelope.append(cfg, clinic, _row(), "opu", "prospective", "m", "r1")
    with pytest.raises(ConfigError, match="перезаписать замороженный конверт"):
        envelope.append(cfg, clinic, _row(p50=99.0), "opu", "prospective", "m", "r2")


def test_registration_mode_is_stored_not_remembered(cfg, clinic):
    """Проспективный и ретроспективный конверт различимы в самих данных."""
    envelope.append(cfg, clinic, _row("CYC-1"), "opu", "prospective", "m", "r1")
    envelope.append(cfg, clinic, _row("CYC-2"), "opu", "retrospective", "m", "r2")
    lookup = envelope.as_lookup(envelope.load_envelopes(envelope.envelope_path(cfg, clinic)))
    assert lookup["CYC-1"]["mii_p50__mode"] == "prospective"
    assert lookup["CYC-2"]["mii_p50__mode"] == "retrospective"


def test_model_input_hides_unknown_columns(clinic):
    """На T0 колонка 2PN обязана быть пустой, даже если в кадре она заполнена.

    Гарантия должна лежать в данных: полагаться на то, что чужой скрипт «наверняка
    не использует» лишнюю колонку, нельзя.
    """
    frame = pd.DataFrame([{"cycle_uid": "CYC-1", "age": 34, "occ": 12, "mii": 10,
                           "two_pn": 7, "follicles": 15, "attempt_no": 1,
                           "amh": 2.0, "bmi": 24.0}])
    at_t0 = envelope.build_model_input(clinic, frame, "opu")
    assert at_t0["2pN"].isna().all(), "на T0 модель не должна видеть 2PN"
    assert at_t0["MII"].iloc[0] == 10, "MII на T0 известен и должен быть передан"

    at_t1 = envelope.build_model_input(clinic, frame, "fertilisation")
    assert at_t1["2pN"].iloc[0] == 7, "на T+1 2PN известен"


def test_stages_are_limited_to_their_checkpoint():
    """Стадия, обусловленная ещё не известной величиной, в конверт не попадает."""
    result = pd.DataFrame([{
        "cycle_uid": "CYC-1",
        "TPFN3_mii_p50": 10.0,
        "TPFN3_blasts_after_fert_p50": 4.0,   # требует 2PN — на T0 недопустима
        "TPFN3_good_p50": 3.0,                # требует Bl — недопустима нигде
    }])
    at_t0 = envelope.extract_stages(result, "opu")
    assert "TPFN3_mii_p50" in at_t0.columns
    assert "TPFN3_blasts_after_fert_p50" not in at_t0.columns
    assert "TPFN3_good_p50" not in at_t0.columns

    at_t1 = envelope.extract_stages(result, "fertilisation")
    assert "TPFN3_blasts_after_fert_p50" in at_t1.columns
    assert "TPFN3_good_p50" not in at_t1.columns, (
        "good обусловлен числом бластоцист — той величиной, которую мы мониторим; "
        "конвертом он быть не может ни на одном чекпойнте"
    )


def test_good_and_frozen_have_no_envelope_stage():
    """TGBDR риск-коррекции не получает — и это должно быть видно в коде."""
    from ews.evaluate import ENVELOPE_STAGE  # noqa: PLC0415

    assert "TGBDR" not in ENVELOPE_STAGE
    assert set(ENVELOPE_STAGE) == {"MIIR", "FR", "BlR"}
    assert ENVELOPE_STAGE["BlR"] == "blasts_after_fert"
