"""Знаменатель FR: цикл без инсеминации в FR не входит."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ews import cycles


def test_fr_denominator_excludes_not_inseminated():
    """Инсеминировано = 0 (Cryo / NO) — FR не определён, а не равен нулю."""
    frame = pd.DataFrame({
        "mii":         [12.0, 6.0, 0.0,    np.nan, 8.0],
        "inseminated": [0.0,  6.0, 5.0,    7.0,    np.nan],
        "two_pn":      [0.0,  4.0, 3.0,    5.0,    6.0],
    })
    denominator = cycles.fr_denominator(frame)
    assert pd.isna(denominator[0]), "витрификация ооцитов не должна давать FR = 0"
    assert denominator[1] == 6.0
    assert denominator[2] == 5.0, "MII = 0 — знаменатель из инсеминированных, как в embryo.kpi"
    assert denominator[3] == 7.0
    assert denominator[4] == 8.0, "пустое «инсеминировано» не исключает цикл"

    frame["fr_denominator"] = denominator
    assert cycles.metric_counts(frame.iloc[0], "FR") is None
    assert cycles.metric_counts(frame.iloc[0], "DegR") is None
    assert cycles.metric_counts(frame.iloc[1], "FR") == (4.0, 6.0)


def test_envelope_qc_skips_not_inseminated_after_fertilisation():
    """Cryo / NO: MII оценивается, 2PN и бластоцисты — нет."""
    from ews import envelope_qc

    frame = pd.DataFrame({"cycle_uid": ["A"], "mii": [10.0], "inseminated": [0.0],
                          "two_pn": [0.0], "bl_total": [0.0]})
    entry = {f"{stage}_{q}": v for stage in envelope_qc.STAGE_ACTUAL
             for q, v in (("p10", 2.0), ("p50", 5.0), ("p90", 9.0))}
    per_cycle, _ = envelope_qc.build(frame, {"A": entry}, {})
    assert set(per_cycle["stage"]) == {"mii"}
