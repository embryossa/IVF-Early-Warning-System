"""QA-слой конверта: факт против P10–P90.

Единственное, для чего конверт пригоден на этих данных: модель хорошо предсказывает
СЧЁТ (corr 0.86–0.97) и не предсказывает ДОЛЮ (0.03–0.16), поэтому риск-коррекция
карт закрыта, а интервал вокруг счёта остаётся осмысленной опорой.

Слой тревог не поднимает. Единственное, на что он влияет, — блокировка эскалации:
просевшее покрытие означает, что под вопросом сама опора, и поднимать по ней alert
до разбора нельзя.
"""
import pandas as pd

from ews import envelope_qc


SETTINGS = {"envelope": {"min_interval_width": 1.0, "iqr_to_sigma_divisor": 2.5631,
                         "baseline_coverage": {"mii": 0.794, "pn2": 0.759,
                                               "blasts": 0.823}}}


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _envelope(uid: str, stage: str = "pn2", p10=6.0, p50=8.0, p90=10.0,
              mode="prospective") -> dict:
    return {uid: {f"{stage}_p10": p10, f"{stage}_p50": p50, f"{stage}_p90": p90,
                  f"{stage}_p50__mode": mode}}


def test_no_envelopes_means_no_qc():
    """Конвертов нет — слоя нет. Пустые кадры, а не выдуманное покрытие."""
    per_cycle, summary = envelope_qc.build(_frame([{"cycle_uid": "A", "two_pn": 8}]),
                                           {}, SETTINGS)
    assert per_cycle.empty and summary.empty


def test_fact_inside_the_interval_is_counted_as_covered():
    per_cycle, _ = envelope_qc.build(
        _frame([{"cycle_uid": "A", "two_pn": 8.0}]), _envelope("A"), SETTINGS)
    row = per_cycle.loc[per_cycle["stage"] == "pn2"].iloc[0]
    assert row["status"] == "in80" and bool(row["in80"]) is True
    assert row["z"] == 0.0


def test_interval_boundaries_are_inclusive():
    """Факт ровно на границе считается покрытым: интервал закрытый."""
    per_cycle, _ = envelope_qc.build(
        _frame([{"cycle_uid": "A", "two_pn": 10.0}]), _envelope("A"), SETTINGS)
    assert bool(per_cycle.loc[per_cycle["stage"] == "pn2", "in80"].iloc[0]) is True


def test_fact_outside_the_interval_is_not_an_alarm_by_itself():
    per_cycle, _ = envelope_qc.build(
        _frame([{"cycle_uid": "A", "two_pn": 2.0}]), _envelope("A"), SETTINGS)
    row = per_cycle.loc[per_cycle["stage"] == "pn2"].iloc[0]
    assert row["status"] == "outside" and row["z"] < 0


def test_degenerate_interval_is_excluded_from_coverage():
    """Полоса шириной <= 1 на счётной величине не может дать 80% покрытия.

    Дискретность счёта превышает разрешение интервала: это ограничение интервала,
    а не сигнал лаборатории. Такие стадии в покрытие не идут, но их доля видна.
    """
    envelopes = {f"C{i}": _envelope(f"C{i}", p10=8.0, p50=8.0, p90=8.5)[f"C{i}"]
                 for i in range(12)}
    frame = _frame([{"cycle_uid": f"C{i}", "two_pn": 3.0} for i in range(12)])
    per_cycle, summary = envelope_qc.build(frame, envelopes, SETTINGS)
    assert (per_cycle["status"] == envelope_qc.DEGENERATE).all()
    assert per_cycle["in80"].isna().all()
    assert summary.empty, "считать покрытие не по чему — все стадии вырожденные"


def test_degenerate_share_is_reported_next_to_coverage():
    envelopes = {}
    rows = []
    for index in range(10):
        envelopes.update(_envelope(f"OK{index}"))
        rows.append({"cycle_uid": f"OK{index}", "two_pn": 8.0})
    envelopes.update(_envelope("BAD", p10=8.0, p50=8.0, p90=8.5))
    rows.append({"cycle_uid": "BAD", "two_pn": 8.0})
    _, summary = envelope_qc.build(_frame(rows), envelopes, SETTINGS)
    entry = summary.iloc[0]
    assert entry["n"] == 10, "вырожденная стадия в знаменатель покрытия не входит"
    assert 0 < entry["degenerate_share"] < 1


def test_coverage_above_baseline_never_alarms():
    """Проверяется только ПАДЕНИЕ покрытия.

    Рост означает, что интервалы шире нужного, — это потеря чувствительности, а не
    сбой лаборатории, и односторонний тест здесь принципиален.
    """
    envelopes, rows = {}, []
    for index in range(30):
        envelopes.update(_envelope(f"C{index}"))
        rows.append({"cycle_uid": f"C{index}", "two_pn": 8.0})
    _, summary = envelope_qc.build(_frame(rows), envelopes, SETTINGS)
    assert summary.iloc[0]["coverage"] == 1.0
    dropped, note = envelope_qc.coverage_alarm(summary)
    assert dropped is False and note == ""


def test_significant_drop_blocks_escalation():
    envelopes, rows = {}, []
    for index in range(30):
        envelopes.update(_envelope(f"C{index}"))
        rows.append({"cycle_uid": f"C{index}", "two_pn": 1.0})   # мимо интервала
    _, summary = envelope_qc.build(_frame(rows), envelopes, SETTINGS)
    dropped, note = envelope_qc.coverage_alarm(summary)
    assert dropped is True
    assert "эскалация до alert заблокирована" in note
    assert "не сигнал культивирования" in note


def test_small_sample_gives_no_verdict():
    """Меньше десяти наблюдений — тест не считается вовсе.

    Молчание здесь честнее вывода: покрытие на пяти циклах не отличает просадку от
    случайности.
    """
    envelopes, rows = {}, []
    for index in range(5):
        envelopes.update(_envelope(f"C{index}"))
        rows.append({"cycle_uid": f"C{index}", "two_pn": 1.0})
    _, summary = envelope_qc.build(_frame(rows), envelopes, SETTINGS)
    assert summary.iloc[0]["p_below"] is None
    dropped, note = envelope_qc.coverage_alarm(summary)
    # Эскалация не блокируется — но «проверять нечем» обязано отличаться в отчёте
    # от «проверено и чисто»: отсутствие свидетельства не есть свидетельство.
    assert dropped is False
    assert "ещё не проверялось" in note


def test_mixed_missing_p_values_do_not_break_the_alarm():
    """Часть стадий без опорного покрытия — не повод падать при сравнении.

    Колонка p_below становится смешанной (число и пропуск), и сравнение с alpha
    обязано пережить это молча.
    """
    summary = pd.DataFrame([
        {"stage": "mii", "p_below": None, "p_below_prospective": None,
         "n_prospective": 3},
        {"stage": "pn2", "p_below": 0.0001, "p_below_prospective": 0.0001,
         "n_prospective": 40},
    ])
    dropped, note = envelope_qc.coverage_alarm(summary)
    assert dropped is True and "pn2" in note and "mii" not in note


def test_empty_summary_is_silent():
    assert envelope_qc.coverage_alarm(pd.DataFrame()) == (False, "")


def test_registration_mode_reaches_the_qc_row():
    """Ретроспективный конверт остаётся отличимым и на этом слое.

    Иначе покрытие, посчитанное по конвертам, которые модель видела при обучении,
    выдавалось бы за проспективную проверку.
    """
    per_cycle, _ = envelope_qc.build(
        _frame([{"cycle_uid": "A", "two_pn": 8.0}]),
        _envelope("A", mode="retrospective"), SETTINGS)
    assert per_cycle.loc[per_cycle["stage"] == "pn2", "mode"].iloc[0] == "retrospective"


def test_stage_without_actual_column_is_skipped():
    """Нет факта — нет сравнения. Пропуск не превращается в промах."""
    per_cycle, _ = envelope_qc.build(
        _frame([{"cycle_uid": "A", "two_pn": None}]), _envelope("A"), SETTINGS)
    assert per_cycle.empty


def test_gate_judges_prospective_envelopes_only():
    """Просадка ретроспективного добора эскалацию не блокирует.

    Измерено на боевой когорте 2026-09-09: blasts_after_fert давал 0.699 покрытия на
    73 ретроспективных конвертах против 0.833 на 6 проспективных, в пуле — 0.709, и
    именно пулированное число запирало alert. Ретроспективный добор писался моделью,
    обучавшейся в том числе на этих циклах: его покрытие — свойство добора, а не
    прогноза, и запирать им тревогу значит блокировать сигнал данными, которые
    ранним предупреждением не являются (решение владельца, 2026-09-10).
    """
    envelopes, rows = {}, []
    for index in range(40):        # ретроспективные — все мимо интервала
        envelopes.update(_envelope(f"R{index}", mode="retrospective"))
        rows.append({"cycle_uid": f"R{index}", "two_pn": 99.0})
    for index in range(20):        # проспективные — все внутри
        envelopes.update(_envelope(f"P{index}", mode="prospective"))
        rows.append({"cycle_uid": f"P{index}", "two_pn": 8.0})

    _, summary = envelope_qc.build(_frame(rows), envelopes, SETTINGS)
    row = summary.iloc[0]
    assert row["coverage"] < 0.5
    assert row["coverage_prospective"] == 1.0
    dropped, _ = envelope_qc.coverage_alarm(summary)
    assert dropped is False


def test_gate_still_fires_when_prospective_coverage_drops():
    """Обратная сторона: просела проспективная часть — эскалация закрыта."""
    envelopes, rows = {}, []
    for index in range(40):
        envelopes.update(_envelope(f"P{index}", mode="prospective"))
        rows.append({"cycle_uid": f"P{index}", "two_pn": 99.0})

    _, summary = envelope_qc.build(_frame(rows), envelopes, SETTINGS)
    dropped, note = envelope_qc.coverage_alarm(summary)
    assert dropped is True
    assert "проспективного конверта" in note
