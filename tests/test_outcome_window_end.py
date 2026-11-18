"""Медленный контур при явном конце опорного окна (окно с первого дня данных)."""
import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import outcome, outcome_ra  # noqa: E402


def _frame(n_window, n_after, p_window, p_after, expected=0.4):
    rng = np.random.default_rng(1)
    d1 = pd.date_range("2023-01-01", "2025-12-31", periods=n_window)
    d2 = pd.date_range("2026-01-02", "2026-09-30", periods=n_after)
    return pd.DataFrame({
        "date_opu": list(d1) + list(d2),
        "transfer_outcome": list(rng.random(n_window) < p_window) + list(rng.random(n_after) < p_after),
        "digital_twin": expected,
    }).assign(transfer_outcome=lambda f: f.transfer_outcome.astype(float))


def test_without_end_behaviour_unchanged():
    f = _frame(300, 100, 0.4, 0.4)
    w = outcome.compare_periods(f, pd.Timestamp("2023-01-01"))
    assert [x.label for x in w] == ["до окна", "опорное окно"]
    verdict, note = outcome.confirms_process_trend(w)
    assert verdict is None and "не созрел" in note


def test_with_end_compares_window_to_after():
    f = _frame(600, 300, 0.45, 0.20)
    w = outcome.compare_periods(f, pd.Timestamp("2023-01-01"), end=pd.Timestamp("2025-12-31"))
    assert [x.label for x in w] == ["до окна", "опорное окно", "после окна"]
    assert w[0].n_known == 0 and w[1].n_known == 600 and w[2].n_known == 300
    verdict, note = outcome.confirms_process_trend(w)
    assert verdict is True and "снизился" in note


def _transfers(n_window, n_after, y_window, y_after, p=0.4):
    d = list(pd.date_range("2023-01-01", "2025-12-31", periods=n_window)) + \
        list(pd.date_range("2026-01-02", "2026-09-30", periods=n_after))
    y = [1.0] * int(n_window * y_window) + [0.0] * (n_window - int(n_window * y_window)) + \
        [1.0] * int(n_after * y_after) + [0.0] * (n_after - int(n_after * y_after))
    return pd.DataFrame({"date_opu": d, "y": y, "p_expected": p})


CFG = {"min_known_period": 20, "inner_limit": 0.95, "outer_limit": 0.998}


def test_ra_verdict_uses_after_window_when_end_given():
    t = _transfers(600, 200, 0.40, 0.20)
    v_old, note_old = outcome_ra.period_verdict(t, pd.Timestamp("2023-01-01"), CFG)
    v_new, note_new = outcome_ra.period_verdict(t, pd.Timestamp("2023-01-01"), CFG,
                                                end=pd.Timestamp("2025-12-31"))
    assert "опорного окна" in note_old
    assert v_new is True and "периода после окна" in note_new and "для сравнения O/E опорного окна" in note_new


def test_ra_verdict_falls_back_to_window_when_after_too_small():
    t = _transfers(600, 5, 0.40, 0.0)
    v, note = outcome_ra.period_verdict(t, pd.Timestamp("2023-01-01"), CFG, end=pd.Timestamp("2025-12-31"))
    assert "опорного окна" in note and "после окна" not in note
