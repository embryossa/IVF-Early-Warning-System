"""Риск-скорректированный медленный контур: свойства, которые обязаны держаться."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import outcome_ra  # noqa: E402

SETTINGS = {"outcome_ra": {"min_known_calibration": 40, "min_known_group": 10,
                           "min_known_period": 20,
                           "cusum": {"replicates": 300}},
            "charts": {"funnel": {"inner_limit": 0.95, "outer_limit": 0.998}}}


def synthetic(n: int, delta: float = 0.0, seed: int = 0, start: str = "2026-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.7, n)
    y = (rng.random(n) < expit(logit(p) + delta)).astype(int)
    return pd.DataFrame({
        "cycle_uid": [f"CYC-{i:04d}" for i in range(n)],
        "date_opu": pd.date_range(start, periods=n, freq="2D"),
        "checkpoint": "closure",
        "transfer_outcome": y,
        "digital_twin": p,
        "embryologist": rng.choice(["Иванова", "Петрова"], n),
    })


def test_outcome_frame_reads_percent_and_drops_unknown():
    frame = synthetic(5)
    frame.loc[0, "digital_twin"] = 45.0          # проценты
    frame.loc[1, "transfer_outcome"] = np.nan     # исход неизвестен
    frame.loc[2, "digital_twin"] = np.nan         # нет ожидания
    out = outcome_ra.outcome_frame(frame)
    assert len(out) == 3
    assert out.loc[out["cycle_uid"] == "CYC-0000", "p_model"].iloc[0] == 0.45
    assert out.attrs["known_without_expected"] == 1


def test_fit_delta_recovers_shift():
    frame = synthetic(4000, delta=0.6, seed=1)
    delta, se = outcome_ra.fit_delta(frame["transfer_outcome"], frame["digital_twin"])
    assert abs(delta - 0.6) < 3 * se


def test_oe_positions():
    rng = np.random.default_rng(2)
    p = np.full(300, 0.5)
    inside = outcome_ra.oe_row((rng.random(300) < 0.5).astype(float), p, 0.95, 0.998, 10)
    assert inside["position"] in (outcome_ra.POSITION_INSIDE, outcome_ra.POSITION_LOW_INNER,
                                  outcome_ra.POSITION_HIGH_INNER)
    low = outcome_ra.oe_row((rng.random(300) < 0.3).astype(float), p, 0.95, 0.998, 10)
    assert low["position"] == outcome_ra.POSITION_LOW_OUTER
    few = outcome_ra.oe_row(np.array([0.0, 0.0]), np.array([0.5, 0.5]), 0.95, 0.998, 10)
    assert few["position"] == outcome_ra.POSITION_FEW


def test_calibration_waits_then_freezes_and_stays(tmp_path):
    split = pd.Timestamp("2026-06-01")
    early = outcome_ra.build(synthetic(30), SETTINGS, tmp_path, split)
    assert not early.calibration.frozen and early.calibration.delta == 0.0
    assert not outcome_ra.calibration_path(tmp_path).exists()

    first = outcome_ra.build(synthetic(120, delta=0.8, seed=3), SETTINGS, tmp_path, split)
    assert first.calibration.frozen
    assert outcome_ra.calibration_path(tmp_path).exists()

    # Новые факты не двигают замороженный сдвиг.
    later = outcome_ra.build(synthetic(200, delta=-0.8, seed=4), SETTINGS, tmp_path, split)
    assert later.calibration.delta == first.calibration.delta


def test_dry_run_does_not_freeze(tmp_path):
    outcome_ra.build(synthetic(80), SETTINGS, tmp_path, pd.Timestamp("2026-03-01"), freeze=False)
    assert not outcome_ra.calibration_path(tmp_path).exists()


def test_staff_names_never_leave(tmp_path):
    result = outcome_ra.build(synthetic(60), SETTINGS, tmp_path, pd.Timestamp("2026-03-01"))
    groups = set(result.strata.loc[result.strata["dimension"] == "эмбриолог", "group"])
    assert groups <= {"Эмбриолог A", "Эмбриолог B"}
    assert not result.transfers["embryologist"].isin(["Иванова", "Петрова"]).any()


def test_lot_change_inside_window_is_not_attributed():
    timeline = pd.DataFrame({
        "item_record_id": ["r1", "r1"], "item_name": ["CULTURE MEDIUM"] * 2,
        "attribute": [None, None], "manufacturer": [None, None],
        "lot": ["LOT-A", "LOT-B"],
        "period_start": pd.to_datetime(["2026-01-01", "2026-03-04"]),
        "period_end": pd.to_datetime(["2026-03-04", pd.NaT]),
    })
    transfers = pd.DataFrame({"cycle_uid": ["c1", "c2", "c3"],
                              "date_opu": pd.to_datetime(["2026-02-01", "2026-03-01",
                                                          "2026-04-01"])})
    assigned = outcome_ra.lot_assignments(transfers, timeline, window_days=6)
    lots = dict(zip(assigned["cycle_uid"], assigned["lot"]))
    assert lots == {"c1": "LOTA", "c2": "смена лота в окне", "c3": "LOTB"}


def test_gs1_lot_is_application_identifier_10():
    # Две коробки одного лота: разный GTIN, общий лот HTFH-2501.
    a = "\x1d01054119670046141726083110HTFH-2501"
    b = "\x1d01054119670046211726083110HTFH-2501"
    assert outcome_ra.gs1_lot(a) == outcome_ra.gs1_lot(b) == "HTFH-2501"
    full = ("\x1d100000050949\x1d24090166-100ML\x1d17270331\x1d916312000085"
            "\x1d0100810195240279\x1d2101479")
    assert outcome_ra.gs1_lot(full) == "0000050949"
    assert outcome_ra.gs1_lot("\x1d0107350025910468112512241726061710514035") == "514035"
    assert outcome_ra.gs1_lot("(01)05412345678908(10)01082601260210") == "01082601260210"
    assert outcome_ra.gs1_lot("48616") is None


def test_short_scan_and_full_datamatrix_are_one_lot():
    full = ("\x1d100000050949\x1d24090166-100ML\x1d17270331\x1d916312000085"
            "\x1d0100810195240279\x1d2101479")
    assigned = pd.DataFrame({"cycle_uid": ["c1", "c2", "c3"], "item": ["OIL"] * 3,
                             "lot": [outcome_ra.normalize_lot("50949"),
                                     outcome_ra.normalize_lot(full),
                                     outcome_ra.normalize_lot("77777")]})
    merged = outcome_ra.merge_scans(assigned)
    assert merged["lot"].nunique() == 2
    assert merged.loc[0, "lot"] == merged.loc[1, "lot"]


def test_cusum_threshold_meets_target_in_control():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.3, 0.7, 200)
    h = outcome_ra.solve_cusum_h(p, 0.5, 0.5, 100, 1500, rng)
    # Независимая проверка частоты на новых розыгрышах.
    check = np.random.default_rng(6)
    signals = []
    for _ in range(1500):
        draws = check.choice(p, 100)
        y = (check.random(100) < draws).astype(float)
        path = outcome_ra.trajectories(pd.DataFrame({"y": y, "p_expected": draws}), 0.5, h)
        signals.append(path["cusum_signal"].sum())
    assert 0.3 < float(np.mean(signals)) < 0.75
