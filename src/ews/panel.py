"""HTML-панель EWS: состояние карт, контуры, журнал тревог.

Панель читает человек, который в этот момент занят: сначала что сработало, потом
что проверить, потом оговорки. PHI сюда не попадает — только cycle_uid.
Палитра — тёплая Claude.ai (--clay:#d97757), как в отчёте embryo-lab-data.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

from .clinic import Clinic
from .config import Config

STYLE = """
:root{--clay:#d97757;--ink:#1f1e1d;--muted:#6b6a67;--line:#e5e2dd;--bg:#faf9f7;
--card:#fff;--ok:#3d8b5f;--watch:#c98a1e;--alert:#c0392b;--ghost:#f2efe9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.badge{display:inline-block;padding:2px 9px;border-radius:11px;font-size:12px;
font-weight:600;letter-spacing:.02em}
.badge.draft{background:#fdf3e3;color:#8a6314;border:1px solid #f0dcb4}
.badge.prod{background:#e8f4ec;color:#1f6440;border:1px solid #bfe0cc}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px 18px;margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.tile .metric{font-size:13px;color:var(--muted);margin-bottom:6px}
.tile .value{font-size:26px;font-weight:600;line-height:1.1}
.tile .foot{font-size:12px;color:var(--muted);margin-top:6px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.lvl-ok{color:var(--ok);font-weight:600}
.lvl-watch{color:var(--watch);font-weight:600}
.lvl-alert{color:var(--alert);font-weight:600}
.bar{position:relative;height:9px;background:var(--ghost);border-radius:5px;
overflow:hidden;min-width:110px}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--clay)}
.bar u{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--alert)}
.note{background:#fdf6f2;border-left:3px solid var(--clay);padding:10px 14px;
border-radius:0 6px 6px 0;font-size:13.5px;margin:8px 0}
.muted{color:var(--muted)}
.scroll{overflow-x:auto}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ink:#f0ede8;--muted:#a5a29c;--line:#33302c;--bg:#1a1917;--card:#22201d;--ghost:#2c2a26}}
"""


def _spark(trajectory: list, watch: float | None, alert: float | None,
           width: int = 220, height: int = 34) -> str:
    """Траектория карты со линиями порогов.

    Одно число в конце периода почти всегда близко к нулю: CUSUM обнуляется после
    сигнала и разряжается в спокойные периоды. Показывать только его — значит
    предлагать читать «0.00 при пороге 1.78» как «мы далеко от беды», хотя это
    означает лишь «прямо сейчас свидетельство не копится». Смысл несёт траектория:
    когда и насколько статистика поднималась.
    """
    values = [float(point[1]) for point in trajectory if len(point) > 1]
    if not values:
        return '<span class="muted">нет данных</span>'
    ceiling = max([*values, watch or 0.0, alert or 0.0, 1e-6]) * 1.1
    step = width / max(len(values) - 1, 1)
    points = " ".join(
        f"{index * step:.1f},{height - value / ceiling * height:.1f}"
        for index, value in enumerate(values))

    lines = ""
    for threshold, colour in ((watch, "#c98a1e"), (alert, "#c0392b")):
        if threshold:
            y = height - threshold / ceiling * height
            lines += (f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" '
                      f'stroke="{colour}" stroke-width="1" stroke-dasharray="3 3"/>')
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'preserveAspectRatio="none" role="img">{lines}'
            f'<polyline points="{points}" fill="none" stroke="var(--clay)" '
            f'stroke-width="1.5"/></svg>')


def _vlad_svg(values: list[float], width: int = 640, height: int = 120) -> str:
    """VLAD: накопленная разница «факт − ожидание», ноль — пунктиром.

    Наклон читается без цифр: ровно — исход как ожидалось для этих пациенток, вниз —
    беременностей меньше ожидаемого, вверх — больше.
    """
    if len(values) < 2:
        return '<span class="muted">нет данных</span>'
    top = max(max(values), 0.0) + 0.5
    bottom = min(min(values), 0.0) - 0.5
    span = top - bottom
    step = width / (len(values) - 1)

    def y(value: float) -> float:
        return height - (value - bottom) / span * height

    points = " ".join(f"{i * step:.1f},{y(v):.1f}" for i, v in enumerate(values))
    zero = y(0.0)
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'preserveAspectRatio="none" role="img" aria-label="VLAD">'
            f'<line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" '
            f'stroke="var(--muted)" stroke-width="1" stroke-dasharray="4 4"/>'
            f'<polyline points="{points}" fill="none" stroke="var(--clay)" '
            f'stroke-width="2"/></svg>')


POSITION_CLASS = {
    "ниже внешнего предела": "lvl-alert", "ниже внутреннего предела": "lvl-watch",
    "выше внутреннего предела": "lvl-ok", "выше внешнего предела": "lvl-ok",
}


def _outcome_ra_section(result) -> list[str]:
    """Медленный контур с риск-коррекцией по DT: VLAD, CUSUM, O/E по группам."""
    summary = result.summary()
    parts = ["<h2>Медленный контур с поправкой на пациенток: исход против ожидания DT</h2>"]
    if not summary["transfers_with_outcome"]:
        parts.append('<div class="note">Переносов с известным исходом и ожиданием DT '
                     'нет — контур не построен.</div>')
        return parts
    overall = summary["overall"]
    cusum = summary["cusum"]
    oe = f"{overall['oe']:.2f}" if overall.get("oe") is not None else "—"
    h_text = f"{cusum['h']:.2f}" if cusum["h"] is not None else "—"
    parts.append('<div class="grid">')
    for label, value, foot in (
        ("Переносов с исходом", str(summary["transfers_with_outcome"]),
         f"факт {overall['observed']} · ожидание {overall['expected']}"),
        ("O/E всего", oe, overall.get("position", "")),
        ("VLAD на сегодня", f"{summary['vlad_last']:+.1f}",
         "беременностей сверх (+) или недобор (−) к ожиданию"),
        ("RA-CUSUM", f"{cusum['last']:.2f}",
         f"порог {h_text} · сигналов {cusum['signals']} · "
         f"~{cusum['transfers_per_year']:.0f} переносов/год"),
    ):
        parts.append(f'<div class="card tile"><div class="metric">{escape(label)}</div>'
                     f'<div class="value">{escape(value)}</div>'
                     f'<div class="foot">{escape(str(foot))}</div></div>')
    parts.append("</div>")
    vlad = result.transfers["vlad"].round(3).tolist()
    trajectory = [[i, v] for i, v in enumerate(result.transfers["cusum"].tolist())]
    parts.append(f'<div class="card"><div class="metric muted">VLAD по переносам в '
                 f'порядке пункции</div>{_vlad_svg(vlad)}'
                 f'<div class="metric muted" style="margin-top:10px">RA-CUSUM на снижение '
                 f'шансов беременности в {1 / result.odds_ratio:.0f} раза</div>'
                 f'{_spark(trajectory, cusum["h"], None, width=640, height=40)}</div>')

    parts.append('<div class="card scroll"><table><tr><th>разрез</th><th>группа</th>'
                 '<th class="num">n</th><th class="num">факт</th><th class="num">ожидание</th>'
                 '<th class="num">O/E</th><th class="num">пределы 95%</th>'
                 '<th class="num">пределы 99.8%</th><th>положение</th></tr>')
    for row in result.strata.to_dict("records"):
        def limits(low: str, high: str) -> str:
            if row.get(low) is None or pd.isna(row.get(low)):
                return "—"
            return f"{row[low]:.2f}–{row[high]:.2f}"
        oe_cell = "—" if row.get("oe") is None or pd.isna(row.get("oe")) else f"{row['oe']:.2f}"
        css = POSITION_CLASS.get(row["position"], "muted")
        parts.append(
            f"<tr><td>{escape(str(row['dimension']))}</td><td>{escape(str(row['group']))}</td>"
            f'<td class="num">{row["n"]}</td><td class="num">{row["observed"]}</td>'
            f'<td class="num">{row["expected"]}</td><td class="num">{oe_cell}</td>'
            f'<td class="num">{limits("inner_low", "inner_high")}</td>'
            f'<td class="num">{limits("outer_low", "outer_high")}</td>'
            f'<td class="{css}">{escape(row["position"])}</td></tr>')
    parts.append("</table></div>")
    notes = [summary["calibration"]["note"], summary["note"], *summary["notes"]]
    parts.append('<div class="note"><b>Как читать.</b> Ожидание — итоговая вероятность '
                 'Digital Twin для каждой пациентки, сдвинутая на уровень клиники '
                 '(сдвиг заморожен и не подстраивается под факт). O/E ниже предела — '
                 'беременностей меньше, чем ждали для ЭТИХ пациенток; это повод посмотреть '
                 'на группу, не вывод о лоте или специалисте. Контур тревог не поднимает.'
                 f'<br>{"<br>".join(escape(n) for n in notes if n)}</div>')
    return parts


def _level_cell(level: str) -> str:
    label = {"ok": "в норме", "watch": "watch", "alert": "ALERT"}.get(level, level)
    return f'<span class="lvl-{level}">{escape(label)}</span>'


ACTION_LABEL = {
    "protocol": ("разбор протокола", "alert"),
    "casemix": ("тренд — состав пациенток / стимуляция", "watch"),
    "standing_gap": ("давняя разница уровня", "muted"),
    "trend": ("наблюдение", "muted"),
}


def render(cfg: Config, clinic: Clinic, frame, evaluations: list,
           states: dict, baselines: dict, thresholds: dict, path: Path,
           literature=None, outcome_windows=None, outcome_note: str = "",
           envelope_qc=None, coverage_note: str = "", outcome_ra=None) -> Path:
    environment = cfg.environment
    block = cfg.alert_block_reason()
    counts = frame.data["checkpoint"].value_counts().to_dict()
    levels = pd.Series([e.level for e in evaluations]).value_counts().to_dict()

    badge = ('<span class="badge draft">ЧЕРНОВОЙ СТЕНД</span>' if environment == "draft"
             else '<span class="badge prod">лаборатория</span>')

    parts: list[str] = [
        f"<title>EWS — {escape(clinic.name)}</title>",
        f"<style>{STYLE}</style>", '<div class="wrap">',
        f"<h1>Раннее предупреждение — {escape(clinic.name)} {badge}</h1>",
        f'<div class="sub">собрано {datetime.now():%Y-%m-%d %H:%M} · '
        f'циклов {len(frame.data)} · оценено на закрытии {len(evaluations)}</div>',
    ]

    if block:
        parts.append(f'<div class="note"><b>Уровень alert заблокирован.</b> {escape(block)}. '
                     f'Всё, что вы видите ниже как watch, могло бы быть alert в боевом '
                     f'режиме с собственной калибровкой.</div>')

    # --- статус карт ---
    parts.append("<h2>Контрольные карты</h2>")
    parts.append('<div class="card scroll"><table><tr><th>карта</th><th>контур</th>'
                 '<th>траектория за период</th><th class="num">пик</th>'
                 '<th class="num">сейчас</th><th class="num">порог watch</th>'
                 '<th class="num">порог alert</th>'
                 '<th class="num">эпизодов watch</th>'
                 '<th class="num">сигналов alert</th></tr>')
    contours = cfg.settings["localisation"]["contours"]
    for metric in clinic.cusum_charts:
        state = states.get(metric)
        if state is None:
            continue
        contour = next((name for name, items in contours.items() if metric in items), "—")
        watch = thresholds.get("h_watch", {}).get(metric)
        alert = thresholds.get("h_alert", {}).get(metric)
        peak = getattr(state, "peak", 0.0)
        episodes = getattr(state, "watch_episodes", 0)
        peak_css = ' class="num lvl-watch"' if watch and peak > watch else ' class="num"'
        parts.append(
            f"<tr><td><b>{escape(metric)}</b></td><td>{escape(contour)}</td>"
            f'<td>{_spark(getattr(state, "trajectory", []), watch, alert)}</td>'
            f'<td{peak_css}>{peak:.2f}</td>'
            f'<td class="num muted">{state.statistic:.2f}</td>'
            f'<td class="num">{f"{watch:.2f}" if watch is not None else "—"}</td>'
            f'<td class="num">{f"{alert:.2f}" if alert is not None else "—"}</td>'
            f'<td class="num">{episodes}</td>'
            f'<td class="num">{state.signals}</td></tr>')
    parts.append("</table></div>")

    parts.append('<div class="note"><b>Как читать.</b> Статистика CUSUM — не уровень '
                 'показателя, а накопленное свидетельство против него; после сигнала '
                 'она обнуляется, в спокойные периоды разряжается сама. Поэтому '
                 'колонка «сейчас» почти всегда близка к нулю и о прошедшем периоде '
                 'не говорит ничего — смотреть надо на <b>траекторию</b> и <b>пик</b> '
                 'относительно пунктирных линий порогов. Пик выше порога watch '
                 'означает, что эпизод был, даже если сейчас всё спокойно.</div>')

    excluded = clinic.subjective_denominator
    if excluded:
        for metric, reason in excluded.items():
            parts.append(f'<div class="note"><b>{escape(metric)} — без карты.</b> '
                         f'{escape(str(reason).strip())}</div>')

    # --- абсолютный контур ---
    parts.append("<h2>Абсолютный контур: опорное окно против консенсуса</h2>")
    parts.append('<div class="card scroll"><table><tr><th>метрика</th>'
                 '<th class="num">уровень</th><th class="num">95% CI</th>'
                 '<th class="num">competency</th><th class="num">benchmark</th>'
                 '<th>статус</th></tr>')
    for metric, base in baselines.items():
        rate = base.get("rate")
        if rate is None or pd.isna(rate):
            continue
        ci = (f'{base["ci_low"]:.3f}–{base["ci_high"]:.3f}'
              if base.get("ci_low") is not None and not pd.isna(base["ci_low"]) else "—")
        parts.append(
            f"<tr><td><b>{escape(metric)}</b></td>"
            f'<td class="num">{rate:.3f}</td><td class="num">{ci}</td>'
            f'<td class="num">{base.get("competency") or "—"}</td>'
            f'<td class="num">{base.get("benchmark") or "—"}</td>'
            f"<td>{escape(str(base.get('status') or ''))}</td></tr>")
    parts.append("</table></div>")

    # --- литературный контур ---
    if literature is not None and not literature.empty:
        parts.append("<h2>Литературный контур: где мы относительно ожидаемого</h2>")
        parts.append('<div class="card scroll"><table><tr><th>метрика</th>'
                     '<th class="num">наша доля</th><th class="num">литература</th>'
                     '<th class="num">разрыв, пп</th><th>источник</th></tr>')
        for _, row in literature.iterrows():
            gap = row.get("gap_pp")
            lit = row.get("literature")
            css = "" if gap is None or pd.isna(gap) else (
                ' class="lvl-ok"' if gap >= 0 else ' class="lvl-watch"')
            parts.append(
                f"<tr><td><b>{escape(str(row['metric']))}</b></td>"
                f'<td class="num">{row["observed"] if row["observed"] is not None else "—"}</td>'
                f'<td class="num">{lit if lit is not None and not pd.isna(lit) else "—"}</td>'
                f'<td class="num"{css}>{f"{gap:+.1f}" if gap is not None and not pd.isna(gap) else "—"}</td>'
                f'<td class="muted">{escape(str(row.get("source", "")))}</td></tr>')
        parts.append("</table></div>")
        parts.append('<div class="note">Опора, которая <b>не двигается вместе с нами</b>. '
                     'Собственная базовая линия самореферентна: если лаборатория сползает, '
                     'а опорное окно переобъявляется, карты не увидят ничего. Здесь этого '
                     'произойти не может.</div>')

    # --- QA конверта ---
    if envelope_qc is not None and not envelope_qc.empty:
        parts.append("<h2>QA конверта: факт против предсказанного интервала</h2>")
        parts.append('<div class="card scroll"><table><tr><th>стадия</th>'
                     '<th class="num">n</th><th class="num">покрытие</th>'
                     '<th class="num">опорное</th><th class="num">разница, пп</th>'
                     '<th class="num">MAE P50</th>'
                     '<th class="num">вырожденных</th></tr>')
        for _, row in envelope_qc.iterrows():
            delta = row.get("delta_pp")
            css = "" if delta is None else (
                ' class="num lvl-watch"' if delta < -5 else ' class="num"')
            parts.append(
                f"<tr><td><b>{escape(str(row['stage']))}</b></td>"
                f'<td class="num">{row["n"]}</td>'
                f'<td class="num">{row["coverage"]:.3f}</td>'
                f'<td class="num">{row["baseline"] if row["baseline"] else "—"}</td>'
                f'<td{css}>{f"{delta:+.1f}" if delta is not None else "—"}</td>'
                f'<td class="num">{row["mae_p50"]}</td>'
                f'<td class="num muted">{row["degenerate_share"]:.0%}</td></tr>')
        parts.append("</table></div>")
        note = (f'<b>Покрытие просело.</b> {escape(coverage_note)}' if coverage_note
                else 'Модель хорошо предсказывает <b>счёт</b> и не предсказывает '
                     '<b>долю</b>, поэтому конверт работает здесь — как независимая '
                     'проверка «попал ли факт в предсказанное», — и не работает как '
                     'риск-коррекция карт. Тревог этот слой не поднимает.')
        parts.append(f'<div class="note">{note}</div>')

    # --- журнал тревог ---
    raised = [e for e in evaluations if e.level != "ok"]
    parts.append(f"<h2>Журнал тревог ({len(raised)})</h2>")
    if not raised:
        parts.append('<div class="card muted">Тревог нет. Это нормальный и самый '
                     'частый исход — система, которая сообщает что-то каждый день, '
                     'перестаёт читаться.</div>')
    else:
        parts.append('<div class="card scroll"><table><tr><th>дата закрытия</th>'
                     '<th>уровень</th><th>карта</th><th>что это значит</th>'
                     '<th>среда</th><th>cycle_uid</th></tr>')
        for item in raised:
            triggered = ", ".join(sorted(m for m, c in item.charts.items() if c.get("crossed")))
            date = (item.closure_date.date().isoformat()
                    if pd.notna(item.closure_date) else "—")
            miri = {"pending": "ожидается", "unavailable": "нет выгрузки",
                    "enriched": "разобрана"}.get(item.miri_status, item.miri_status)
            label, css = ACTION_LABEL.get(item.action, (item.action, "muted"))
            css_class = {"alert": "lvl-alert", "watch": "lvl-watch"}.get(css, "muted")
            parts.append(
                f"<tr><td>{escape(date)}</td><td>{_level_cell(item.level)}</td>"
                f"<td>{escape(triggered)}</td>"
                f'<td><span class="{css_class}">{escape(label)}</span>'
                f'<div class="muted" style="font-size:12px">'
                f'{escape(item.verdict_summary)}</div></td>'
                f'<td class="muted">{escape(miri)}</td>'
                f'<td class="muted">{escape(item.cycle_uid)}</td></tr>')
        parts.append("</table></div>")
        parts.append('<div class="note"><b>«Тренд» — не провал и не повод разбирать '
                     'протокол.</b> Он означает, что лаборатория изменилась относительно '
                     'СВОЕЙ истории, оставаясь в пределах консенсуса и литературного '
                     'ожидания. Вероятные причины лежат вне лаборатории: изменилась '
                     'стимуляция либо пришли более сложные пациентки. Разбор протокола '
                     'оправдан только когда просадка совпала с выходом за уровень '
                     'компетентности.</div>')
        parts.append('<div class="note">Пустое поле среды означает «логи ещё не '
                     'выгружены», а не «нарушений инкубации не было». Выгрузка '
                     'приходит раз в 7 дней и дописывается отдельным проходом.</div>')

    # --- медленный контур ---
    if outcome_windows:
        parts.append("<h2>Медленный контур: подтверждение по исходу</h2>")
        parts.append('<div class="card scroll"><table><tr><th>период</th>'
                     '<th class="num">циклов</th><th class="num">исход известен</th>'
                     '<th class="num">факт</th><th class="num">95% CI</th>'
                     '<th>прогноз моделей</th></tr>')
        from .outcome import calibration_note  # noqa: PLC0415

        for window in outcome_windows:
            fact = f"{window.observed:.0%}" if window.observed is not None else "—"
            ci = (f"{window.ci_low:.0%}–{window.ci_high:.0%}"
                  if window.ci_low is not None and not pd.isna(window.ci_low) else "—")
            parts.append(
                f"<tr><td><b>{escape(window.label)}</b></td>"
                f'<td class="num">{window.n_cycles}</td>'
                f'<td class="num">{window.n_known}</td>'
                f'<td class="num">{fact}</td><td class="num">{ci}</td>'
                f'<td class="muted">{escape(calibration_note(window))}</td></tr>')
        parts.append("</table></div>")
        parts.append(f'<div class="note"><b>Что это даёт.</b> Процесс виден на 6-й '
                     f'день, исход — через недели. Медленный контур не поднимает '
                     f'тревог: исход зависит от переноса, эндометрия и десятка вещей '
                     f'вне лаборатории. Его роль — подтверждать или опровергать тренды '
                     f'быстрого контура.<br>{escape(outcome_note)}</div>')
    if outcome_ra is not None:
        parts.extend(_outcome_ra_section(outcome_ra))

    # --- сводка ---
    parts.append("<h2>Итог прогона</h2>")
    parts.append('<div class="grid">')
    for label, value, foot in (
        ("Циклов в работе", str(counts.get("opu", 0) + counts.get("fertilisation", 0)),
         "T0 и T+1 — регистрация, тревог не поднимают"),
        ("Оценено на закрытии", str(len(evaluations)), "единственная точка оценки"),
        ("watch", str(levels.get("watch", 0)), "на панель, никого не будит"),
        ("alert", str(levels.get("alert", 0)), block or "инициирует разбор"),
    ):
        parts.append(f'<div class="card tile"><div class="metric">{escape(label)}</div>'
                     f'<div class="value">{escape(value)}</div>'
                     f'<div class="foot">{escape(foot)}</div></div>')
    parts.append("</div>")

    parts.append('<div class="sub" style="margin-top:26px">Исследовательский '
                 'инструмент. Наблюдения и поводы для разбора — не диагнозы, не '
                 'назначения и не оценка конкретного специалиста.</div></div>')

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".html.tmp")
    tmp.write_text("\n".join(parts), encoding="utf-8")
    tmp.replace(path)
    return path
