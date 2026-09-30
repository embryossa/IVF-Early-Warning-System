"""Демо-контур: свойства, которые обязаны держаться на чужих выгрузках."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ews.demo import ingest, report, stats as st  # noqa: E402


def _write(tmp_path, frame, name="clinic.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
    return path


def _synthetic(n=80, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-10-01", periods=n, freq="3D")
    occ = rng.integers(4, 15, n)
    mii = np.minimum(occ, rng.binomial(occ, 0.8) + 1)
    two_pn = rng.binomial(mii, 0.75)
    bl = rng.binomial(two_pn, 0.5)
    return pd.DataFrame({
        "Patient ID": [f"P{i}" for i in range(n)], "ФИО": [f"Иванова {i}" for i in range(n)],
        "Дата пункции": dates.strftime("%d.%m.%Y"), "Возраст": rng.integers(25, 42, n),
        "Follicles": occ + 2, "OCC": occ, "MII": mii, "2 pN": two_pn, "Cleavage": two_pn,
        "Blastocysts": bl, "Good Bl": rng.binomial(bl, 0.6),
        "IVF/ICSI": rng.choice(["ICSI", "IVF"], n), "Embryologist": rng.choice(["Anna", "Olga"], n),
    })


def test_columns_exact_only_and_ambiguity_not_guessed():
    aliases = ingest.load_aliases()
    rep = ingest.resolve_columns(["2 pN ", "Число ОКК", "OCC", "Bl", "Unrelated 2pN count"], aliases)
    assert rep.mapped["two_pn"] == "2 pN "               # регистр/пробелы — да
    assert "occ" in rep.ambiguous                          # две колонки -> не угадывать
    assert "occ" not in rep.mapped
    assert "Unrelated 2pN count" in rep.unmapped_source    # подстрока — не синоним


def test_method_from_flags_and_value():
    aliases = ingest.load_aliases()
    f = pd.DataFrame({"ivf_flag": ["Да", None, "Да"], "icsi_flag": [None, "Да", "Да"]})
    assert ingest.derive_method(f, aliases).tolist() == ["IVF", "ICSI", "IVF+ICSI"]
    g = pd.DataFrame({"fertilization_method": ["ICSI", "IVF/ICSI", "cIVF", "Cryo"]})
    assert ingest.derive_method(g, aliases).tolist() == ["ICSI", "IVF+ICSI", "IVF", "Cryo"]


def test_phi_removed_and_operators_pseudonymized(tmp_path):
    demo = ingest.load_demo(_write(tmp_path, _synthetic()), secret=b"k")
    cols = set(demo.frame.columns)
    assert not cols & {"patient_id", "patient_name", "dob"}
    assert demo.frame["cycle_uid"].str.startswith("CYC-").all()
    assert set(demo.frame["embryologist"].dropna()) <= {"Эмбриолог A", "Эмбриолог B"}
    assert not demo.frame.astype(str).apply(lambda c: c.str.contains("Иванова")).any().any()


def test_summary_row_excluded_but_clr_over_100_kept(tmp_path):
    f = _synthetic(30)
    f.loc[3, "Cleavage"] = f.loc[3, "2 pN"] + 2               # ClR > 100% — наблюдение
    total = f.iloc[[0]].copy()
    total["Patient ID"] = "Итого"
    for c in ("OCC", "MII", "2 pN", "Cleavage", "Blastocysts", "Good Bl", "Follicles"):
        total[c] = f[c].sum()
    demo = ingest.load_demo(_write(tmp_path, pd.concat([f, total], ignore_index=True)), secret=b"k")
    assert demo.exclusions["summary_or_implausible"] == 1
    rows = st.metric_rows(st.cycle_rates(demo.frame), "ClR")
    assert rows["over_one"].sum() == 1


def test_pooled_is_sum_over_sum_not_mean_of_rates():
    rows = pd.DataFrame({"x": [1.0, 9.0], "n": [1.0, 20.0]})
    p = st.pooled(rows)
    assert p["rate"] == pytest.approx(10 / 21)
    assert p["rate"] != pytest.approx((1 / 1 + 9 / 20) / 2)


def test_ci_undefined_when_numerator_exceeds_denominator():
    lo, hi = st.wilson_phi(5, 4, 1.5)
    assert np.isnan(lo) and np.isnan(hi)


def test_overdispersion_widens_ci():
    lo1, hi1 = st.wilson_phi(60, 100, 1.0)
    lo2, hi2 = st.wilson_phi(60, 100, 2.5)
    assert (hi2 - lo2) > (hi1 - lo1)


def test_compare_groups_uses_quasi_test_under_overdispersion():
    rng = np.random.default_rng(3)
    rows = []
    for g, p in (("A", 0.75), ("B", 0.55)):
        for _ in range(40):
            n = int(rng.integers(4, 14))
            q = rng.beta(p * 5, (1 - p) * 5)                 # сильный межцикловый разброс
            rows.append({"grp": g, "x": float(rng.binomial(n, q)), "n": float(n)})
    rows = pd.DataFrame(rows).assign(rate=lambda d: d["x"] / d["n"])
    res = st.compare_groups(rows, "grp")
    assert res["phi"] > 1.2
    assert res["test"].startswith("квазибиномиальный")
    assert res["robust"]["test"].startswith("Манн")


def test_thresholds_come_from_kpi_specs():
    from ews.vendor import kpi
    s = st.spec("BlR")
    assert (s.competency, s.benchmark) == (kpi.KPI_SPECS["BlR"].competency, kpi.KPI_SPECS["BlR"].benchmark)


def test_end_to_end_never_alert(tmp_path):
    demo = ingest.load_demo(_write(tmp_path, _synthetic(120)), secret=b"k")
    res = report.build(demo, replicates=40, seed=1)
    for c in res["cusum"].values():
        if c.get("available") and not c["events"].empty:
            assert set(c["events"]["level"]) == {"watch"}
    assert res["environment"] == "draft" and res["mode"] == "demo"

    import ews_demo
    packet = json.dumps(ews_demo.agent_packet(res), ensure_ascii=False)
    assert "CYC-" not in packet and "Anna" not in packet and "Иванова" not in packet

    from ews.demo import dashboard
    html = dashboard.render(res, plotly_js="cdn")
    assert "environment=draft" in html and "alert заблокирован" in html
    assert "Иванова" not in html and "Anna" not in html
