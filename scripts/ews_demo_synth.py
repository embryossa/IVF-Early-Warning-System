#!/usr/bin/env python3
"""Синтетическая выгрузка «чужой клиники» для демо-панели без реальных данных.

    python scripts/ews_demo_synth.py            -> data/sample/demo_clinic_synthetic.csv

Всё выдумано: идентификаторы вида SYN-0001, имён пациенток нет. Имена колонок
намеренно другие, чем у Two-Line и Fertimed (англоязычные синонимы из словаря
config/demo/column_aliases.yaml), — чтобы демо показывало сопоставление колонок.

Заложенные «события», которые панель должна увидеть (и которые можно назвать на
презентации как проверку системы):
  * провал культивирования: BlR и TGBDR снижены в шансах ×0.45 для пункций с
    середины августа по конец сентября 2026 (карта CUSUM BlR, теплокарта месяцев);
  * эмбриолог «Kim» — FR ниже в шансах ×0.45 (funnel plot, тест между эмбриологами);
  * 4 цикла с ClR > 100% (наблюдение о тайминге проверки, не ошибка).
Дата пункции — 12 месяцев, ~20 циклов в месяц.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "demo_clinic_synthetic.csv"


def shift(p: float, odds: float) -> float:
    return p * odds / (1 - p + odds * p)


def main() -> int:
    rng = np.random.default_rng(20261001)
    days = pd.date_range("2025-10-01", "2026-09-30", freq="D")
    rows = []
    embryologists = ["Petrova", "Ivanova", "Kim"]
    for i in range(240):
        date = days[rng.integers(len(days))]
        age = int(np.clip(rng.normal(34, 4.5), 22, 45))
        amh = round(float(np.exp(rng.normal(0.8 - 0.04 * (age - 34), 0.6))), 2)
        afc = int(np.clip(rng.poisson(4 + 3.5 * amh), 1, 40))
        follicles = int(np.clip(rng.poisson(afc * 0.9 + 2), 1, 45))
        occ = int(rng.binomial(follicles, 0.78))
        if occ == 0:
            occ = 1
        method = "ICSI" if rng.random() < 0.65 else "IVF"
        emb = embryologists[rng.choice(3, p=[0.45, 0.35, 0.20])]
        # межцикловая неоднородность (бета-биномиальный разброс, как в реальных данных)
        def beta(p, k=12):
            return float(rng.beta(p * k, (1 - p) * k))
        mii = int(rng.binomial(occ, beta(0.80)))
        inseminated = mii if method == "ICSI" else occ
        p_fr = 0.78 if method == "ICSI" else 0.68
        if emb == "Kim":
            p_fr = shift(p_fr, 0.45)
        deg = int(rng.binomial(mii, 0.04)) if method == "ICSI" else 0
        two_pn = int(rng.binomial(max(inseminated - deg, 0), beta(p_fr)))
        three_pn = int(rng.binomial(inseminated, 0.03 if method == "IVF" else 0.01))
        cleav = int(rng.binomial(two_pn, 0.97))
        if i % 60 == 7 and two_pn > 1:
            cleav = two_pn + 1                      # ClR > 100%: поздняя проверка PN
        p_bl = 0.55 - 0.008 * max(age - 35, 0)
        p_good = 0.72
        episode = pd.Timestamp("2026-08-15") <= date <= pd.Timestamp("2026-09-30")
        if episode:
            p_bl = shift(p_bl, 0.45)
        bl = int(rng.binomial(two_pn, beta(p_bl)))
        good = int(rng.binomial(bl, beta(p_good)))
        rows.append({
            "Cycle ID": f"SYN-{i + 1:04d}", "Retrieval date": date.strftime("%d.%m.%Y"),
            "Female age": age, "AMH": str(amh).replace(".", ","), "AFC": afc, "Follicles": follicles,
            "Oocytes retrieved": occ, "MII oocytes": mii, "Fertilization method": method,
            "Inseminated oocytes": inseminated, "2PN zygotes": two_pn, "3PN": three_pn,
            "Degenerated": deg, "Cleavage D3": cleav, "Blastocysts": bl, "Good blastocysts": good,
            "Embryologist": emb, "Physician": rng.choice(["Dr. A", "Dr. B"]),
        })
    frame = pd.DataFrame(rows).sort_values("Retrieval date", key=lambda s: pd.to_datetime(s, dayfirst=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, sep=";", encoding="utf-8-sig")
    print(f"[ok] {OUT.relative_to(ROOT).as_posix()}: {len(frame)} синтетических циклов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
