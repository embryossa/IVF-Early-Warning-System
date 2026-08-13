"""Гейт alert обязан быть закрыт во всех трёх случаях, а не в одном."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews.config import Config


def _cfg(environment, calibrated, calibrated_env):
    return Config(
        settings={"thresholds": {"calibrated": calibrated,
                                 "calibrated_environment": calibrated_env}},
        machine={"environment": environment}, secrets={})


def test_draft_never_alerts():
    cfg = _cfg("draft", True, "draft")
    assert not cfg.alert_level_available
    assert "draft" in cfg.alert_block_reason()


def test_uncalibrated_never_alerts():
    cfg = _cfg("production", False, None)
    assert not cfg.alert_level_available
    assert "calibrated" in cfg.alert_block_reason()


def test_thresholds_do_not_travel_between_environments():
    """Пороги чернового стенда не должны открывать alert в лаборатории."""
    cfg = _cfg("production", True, "draft")
    assert not cfg.alert_level_available
    assert "бэкфит" in cfg.alert_block_reason()


def test_production_with_own_calibration_alerts():
    cfg = _cfg("production", True, "production")
    assert cfg.alert_level_available
    assert cfg.alert_block_reason() is None
