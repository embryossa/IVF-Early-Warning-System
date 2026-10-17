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
.cc-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;margin-bottom:4px}
.cc-head .name{font-size:19px;font-weight:700}
.cc-head .label{color:var(--muted);font-size:13px}
.pill{display:inline-block;padding:1px 9px;border-radius:11px;font-size:12px;font-weight:600}
.pill.calm{background:var(--ghost);color:var(--muted)}
.pill.watch{background:#fdf3e3;color:#8a6314}
.pill.alert{background:#fbe9e7;color:#9c2b1f}
.chips{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--muted);margin:6px 0 4px}
.chips b{color:var(--ink);font-variant-numeric:tabular-nums}
.cc svg{display:block;width:100%;height:auto;overflow:visible}
.cc text{font:11px -apple-system,'Segoe UI',Roboto,sans-serif;fill:var(--muted)}
.cc .cap{font-size:12px;color:var(--muted);margin:10px 0 2px}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--muted);margin-top:8px}
.legend i{display:inline-block;width:14px;height:0;border-top:2px solid;vertical-align:middle;margin-right:5px}
.legend i.dash{border-top-style:dashed}.legend i.dot{border-top-style:dotted}
.legend s{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:5px;text-decoration:none}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ink:#f0ede8;--muted:#a5a29c;--line:#33302c;--bg:#1a1917;--card:#22201d;--ghost:#2c2a26}
.pill.watch{background:#3a2f1a;color:#e8b860}.pill.alert{background:#3d211d;color:#f0a093}}
"""

MONTHS_RU = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
CONTOUR_RU = {"culture": "контур культивирования", "embryologist": "контур эмбриолога",
              "doctor": "контур врача"}
MIN_MONTH_CYCLES = 5    # месяц на меньшем числе циклов показывается, но приглушён


def _monthly_rates(frame: pd.DataFrame, metric: str) -> list[dict]:
    """Месячная доля Σ числителей / Σ знаменателей по закрытым циклам — те же
    циклы и те же правила наблюдаемости, что у карты (cycles.metric_counts)."""
    from .cycles import metric_counts  # noqa: PLC0415
    from .outcome import _wilson  # noqa: PLC0415

    closed = frame.loc[frame["checkpoint"] == "closure"]
    sums: dict[pd.Period, list[float]] = {}
    for _, row in closed.iterrows():
        counts = metric_counts(row, metric)
        if counts is None or pd.isna(row.get("closure_due")):
            continue
        month = pd.Timestamp(row["closure_due"]).to_period("M")
        bucket = sums.setdefault(month, [0.0, 0.0, 0])
        bucket[0] += counts[0]; bucket[1] += counts[1]; bucket[2] += 1
    out = []
    for month in sorted(sums):
        num, den, n = sums[month]
        low, high = _wilson(num, den)
        out.append({"month": month, "num": num, "den": den, "cycles": n,
                    "rate": num / den, "ci_low": low, "ci_high": high})
    return out


def _rate_status(rate: float, base: dict) -> str:
    """ok / watch / alert относительно benchmark и competency с учётом направления."""
    competency, benchmark = base.get("competency"), base.get("benchmark")
    upper = base.get("direction") == "upper"
    if competency is not None and (rate > competency if upper else rate < competency):
        return "alert"
    if benchmark is not None and (rate > benchmark if upper else rate < benchmark):
        return "watch"
    return "ok"


def _control_chart_card(metric: str, contour: str, state, watch: float | None,
                        alert: float | None, events: list[dict], monthly: list[dict],
                        base: dict, window: tuple | None) -> str:
    """Карточка карты: CUSUM по оси дат с зонами и событиями + месячные доли под ней.

    Две панели делят ось времени: сверху — накопленное свидетельство (CUSUM), снизу —
    сам показатель, из которого оно копится. Без нижней панели «статистика 7.03» не
    говорит, о каком уровне доли идёт речь; без верхней месячная доля не говорит,
    выходит ли колебание за то, что ожидаемо от разнородности пациенток.
    """
    trajectory = [(pd.Timestamp(p[0]), float(p[1])) for p in getattr(state, "trajectory", [])
                  if len(p) > 1 and p[0]]
    if not trajectory:
        return ""
    W, H, L, R, T, B = 1000, 230, 46, 16, 12, 26
    H2, T2 = 150, 10
    start = min(t for t, _ in trajectory)
    if monthly:
        start = min(start, monthly[0]["month"].to_timestamp())
    end = max(t for t, _ in trajectory) + pd.Timedelta(days=2)
    span_days = max((end - start).days, 1)

    def x(ts: pd.Timestamp) -> float:
        return L + (ts - start).days / span_days * (W - L - R)

    peak = getattr(state, "peak", 0.0)
    y_top = max(peak, watch or 0.0, alert or 0.0, 1.0) * 1.08

    def y(value: float) -> float:
        return T + (1 - value / y_top) * (H - T - B)

    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="CUSUM {escape(metric)}">']
    # зоны: спокойно / между watch и alert / выше alert
    if watch:
        svg.append(f'<rect x="{L}" y="{y(min(alert or y_top, y_top)):.1f}" width="{W - L - R}" '
                   f'height="{y(watch) - y(min(alert or y_top, y_top)):.1f}" fill="#c98a1e" opacity=".08"/>')
    if alert and alert < y_top:
        svg.append(f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{y(alert) - T:.1f}" '
                   f'fill="#c0392b" opacity=".08"/>')
    # опорное окно — полоса у оси
    if window and window[0] is not None:
        w0 = max(pd.Timestamp(window[0]), start)
        w1 = min(pd.Timestamp(window[1]) if window[1] is not None else end, end)
        if w1 > w0:
            svg.append(f'<rect x="{x(w0):.1f}" y="{H - B + 1}" width="{x(w1) - x(w0):.1f}" height="4" '
                       f'fill="var(--clay)" opacity=".35"><title>опорное окно: '
                       f'{w0:%d.%m.%Y} — {w1:%d.%m.%Y}</title></rect>')
    # месячная сетка
    month = pd.Timestamp(start.year, start.month, 1)
    while month <= end:
        if month >= start:
            xm = x(month)
            svg.append(f'<line x1="{xm:.1f}" y1="{T}" x2="{xm:.1f}" y2="{H - B}" stroke="var(--line)"/>')
            label = MONTHS_RU[month.month - 1] + (f" {month.year}" if month.month == 1 or month == pd.Timestamp(start.year, start.month, 1) else "")
            svg.append(f'<text x="{xm + 3:.1f}" y="{H - 8}">{label}</text>')
        month += pd.offsets.MonthBegin(1)
    # шкала статистики
    for tick in (0.0, watch, alert):
        if tick is None:
            continue
        svg.append(f'<text x="{L - 6}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick:.1f}</text>')
    for value, colour, name in ((watch, "#c98a1e", "watch"), (alert, "#c0392b", "alert")):
        if value:
            svg.append(f'<line x1="{L}" y1="{y(value):.1f}" x2="{W - R}" y2="{y(value):.1f}" '
                       f'stroke="{colour}" stroke-width="1.2" stroke-dasharray="5 4"/>'
                       f'<text x="{L + 6}" y="{y(value) - 4:.1f}" '
                       f'style="fill:{colour}">порог {name} {value:.2f}</text>')
    # траектория: несколько циклов в один день разносятся внутри дня
    seen: dict = {}
    points = []
    for ts, value in trajectory:
        k = seen.get(ts, 0); seen[ts] = k + 1
        points.append((x(ts + pd.Timedelta(hours=6 * k)), y(value), ts, value))
    # Ступенька: статистика меняется только на закрытии цикла и до следующего держится.
    # Диагональ через пустые недели нарисовала бы рост, которого не было.
    steps = [f"{points[0][0]:.1f},{points[0][1]:.1f}"]
    for (px0, py0, _, _), (px1, py1, _, _) in zip(points, points[1:]):
        steps.append(f"{px1:.1f},{py0:.1f}")
        steps.append(f"{px1:.1f},{py1:.1f}")
    line = " ".join(steps)
    area = f"{points[0][0]:.1f},{y(0):.1f} {line} {points[-1][0]:.1f},{y(0):.1f}"
    svg.append(f'<polygon points="{area}" fill="var(--clay)" opacity=".13"/>')
    svg.append(f'<polyline points="{line}" fill="none" stroke="var(--clay)" stroke-width="1.8" '
               f'stroke-linejoin="round"/>')
    if watch:
        for px, py, ts, value in points:
            if value > watch:
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2" fill="var(--clay)">'
                           f'<title>{ts:%d.%m.%Y}: статистика {value:.2f}</title></circle>')
    # события журнала
    for event in events:
        ts = pd.Timestamp(event["date"])
        if ts < start or ts > end:
            continue
        px = x(ts)
        crossing = event["crossing"]
        if crossing and alert:
            py = y(max([v for t, v in trajectory if t == ts] + [alert]))
            svg.append(f'<path d="M{px:.1f},{py - 13:.1f} l6,10 h-12 z" fill="#c0392b" stroke="var(--card)" '
                       f'stroke-width="1"><title>{escape(event["tip"])}</title></path>')
        else:
            value = max((v for t, v in trajectory if t == ts), default=watch or 0.0)
            svg.append(f'<circle cx="{px:.1f}" cy="{y(value):.1f}" r="5.5" fill="#c98a1e" '
                       f'stroke="var(--card)" stroke-width="1.5"><title>{escape(event["tip"])}</title></circle>')
    svg.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="var(--muted)" stroke-width=".8"/>')
    svg.append("</svg>")

    # нижняя панель: месячная доля
    rates = [m["rate"] for m in monthly] + [m["ci_low"] for m in monthly]
    refs = [v for v in (base.get("rate"), base.get("competency"), base.get("benchmark")) if v is not None]
    r_low = max(0.0, min([*rates, *refs, 1.0]) - 0.05)
    r_high = min(1.0, max([m["ci_high"] for m in monthly] + refs + [0.0]) + 0.03) if monthly else 1.0

    def y2(value: float) -> float:
        return T2 + (1 - (value - r_low) / max(r_high - r_low, 1e-6)) * (H2 - T2 - B)

    colours = {"ok": "#3d8b5f", "watch": "#c98a1e", "alert": "#c0392b"}
    low = [f'<svg viewBox="0 0 {W} {H2}" role="img" aria-label="{escape(metric)} по месяцам">']
    for tick in (r_low, (r_low + r_high) / 2, r_high):
        low.append(f'<text x="{L - 6}" y="{y2(tick) + 4:.1f}" text-anchor="end">{tick:.0%}</text>'
                   f'<line x1="{L}" y1="{y2(tick):.1f}" x2="{W - R}" y2="{y2(tick):.1f}" stroke="var(--line)"/>')
    for m in monthly:
        m0 = m["month"].to_timestamp()
        m1 = (m["month"] + 1).to_timestamp()
        x0, x1 = max(x(m0), L), min(x(m1), W - R)
        if x1 <= x0:
            continue
        pad = (x1 - x0) * 0.05
        status = _rate_status(m["rate"], base)
        faint = m["cycles"] < MIN_MONTH_CYCLES
        opacity = ".35" if faint else ".6"
        tip = (f"{MONTHS_RU[m0.month - 1]} {m0.year}: {m['num']:.0f}/{m['den']:.0f} = {m['rate']:.3f} "
               f"(95% CI {m['ci_low']:.3f}–{m['ci_high']:.3f}), циклов {m['cycles']}"
               + (" — мало циклов, не интерпретируется" if faint else ""))
        # Ось долей обрезана (иначе различия не видны), поэтому точка с интервалом,
        # а не столбец: высота столбца от обрезанного нуля преувеличивает разницу.
        top = y2(m["rate"])
        xc = (x0 + x1) / 2
        low.append(f'<rect x="{x0 + pad:.1f}" y="{T2}" width="{x1 - x0 - 2 * pad:.1f}" '
                   f'height="{H2 - B - T2}" fill="transparent"><title>{escape(tip)}</title></rect>')
        low.append(f'<line x1="{xc:.1f}" y1="{y2(m["ci_low"]):.1f}" x2="{xc:.1f}" y2="{y2(m["ci_high"]):.1f}" '
                   f'stroke="{colours[status]}" stroke-width="2.5" stroke-linecap="round" opacity="{opacity}"/>')
        low.append(f'<circle cx="{xc:.1f}" cy="{top:.1f}" r="6" fill="{colours[status]}" '
                   f'stroke="var(--card)" stroke-width="1.5" opacity="{"0.45" if faint else "1"}">'
                   f'<title>{escape(tip)}</title></circle>')
        low.append(f'<text x="{xc + 10:.1f}" y="{top + 4:.1f}">{m["rate"]:.0%}</text>')
    for value, colour, dash, name in ((base.get("rate"), "var(--clay)", "6 4", "p0"),
                                      (base.get("benchmark"), "#3d8b5f", "2 3", "benchmark"),
                                      (base.get("competency"), "#c0392b", "2 3", "competency")):
        if value is None or not r_low <= value <= r_high:
            continue
        low.append(f'<line x1="{L}" y1="{y2(value):.1f}" x2="{W - R}" y2="{y2(value):.1f}" stroke="{colour}" '
                   f'stroke-width="1.4" stroke-dasharray="{dash}"><title>{name} {value:.3f}</title></line>')
    low.append(f'<line x1="{L}" y1="{H2 - B}" x2="{W - R}" y2="{H2 - B}" stroke="var(--muted)" stroke-width=".8"/>')
    low.append("</svg>")

    statistic = state.statistic
    # После пересечения alert статистика сброшена в 0 — «спокойно» по ней было бы
    # неправдой, пока тревога свежая и ещё разбирается.
    recent = [e for e in events if e["crossing"]
              and pd.Timestamp(e["date"]) >= pd.Timestamp(datetime.now()) - pd.Timedelta(days=30)]
    if recent:
        last = max(pd.Timestamp(e["date"]) for e in recent)
        pill = f'<span class="pill alert">alert {last:%d.%m} — на разборе</span>'
    elif alert and getattr(state, "above_watch", False):
        pill = '<span class="pill watch">эпизод watch открыт</span>'
    elif watch and statistic > 0:
        pill = '<span class="pill calm">копится, ниже порога</span>'
    else:
        pill = '<span class="pill calm">спокойно</span>'
    p0 = base.get("rate")
    chips = [("сейчас", f"{statistic:.2f}"), ("пик за историю", f"{peak:.2f}"),
             ("эпизодов watch", str(getattr(state, "watch_episodes", 0))),
             ("пересечений alert", str(state.signals))]
    if p0 is not None:
        chips.append(("p0 опорного окна", f"{p0:.3f}"))
    chips_html = "".join(f"<span>{escape(k)} <b>{escape(v)}</b></span>" for k, v in chips)
    label = escape(base.get("label") or "")
    return (f'<div class="card cc"><div class="cc-head"><span class="name">{escape(metric)}</span>'
            f'<span class="label">{label} · {escape(CONTOUR_RU.get(contour, contour))}</span>{pill}</div>'
            f'<div class="chips">{chips_html}</div>'
            f'<div class="cap">Накопленное свидетельство против уровня p0 (CUSUM), по дате закрытия цикла</div>'
            f'{"".join(svg)}'
            f'<div class="cap">Сама доля по месяцам закрытия: Σ числителей / Σ знаменателей, точка и 95% CI (шкала обрезана)</div>'
            f'{"".join(low)}</div>')


def _chart_events(journal, metric: str) -> list[dict]:
    """События карты из журнала тревог: watch и пересечения alert (в т.ч. пониженные)."""
    if journal is None or journal.empty:
        return []
    out = []
    for _, row in journal.iterrows():
        charts_hit = [c.strip() for c in str(row.get("chart", "")).split(",")]
        if metric not in charts_hit:
            continue
        notes = "" if pd.isna(row.get("notes")) else str(row.get("notes"))
        crossing = row.get("level") == "alert" or "alert понижен" in notes
        tip = (f"{row['raised_at']} · {'пересечение alert' if crossing else 'watch'} · "
               f"{row['cycle_uid']}" + (f" · {notes}" if notes else ""))
        out.append({"date": row["raised_at"], "crossing": crossing, "tip": tip})
    return out


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


POSITION_COLOUR = {
    "ниже внешнего предела": "#c0392b", "ниже внутреннего предела": "#c98a1e",
    "в пределах": "var(--muted)", "выше внутреннего предела": "#3d8b5f",
    "выше внешнего предела": "#3d8b5f", "мало исходов": "var(--muted)",
}


def _month_grid(start: pd.Timestamp, end: pd.Timestamp, x, top: float, bottom: float,
                label_y: float) -> list[str]:
    """Вертикальная сетка по месяцам и подписи — общая для панелей одной оси."""
    out = []
    month = pd.Timestamp(start.year, start.month, 1)
    first = True
    while month <= end:
        if month >= start:
            xm = x(month)
            out.append(f'<line x1="{xm:.1f}" y1="{top}" x2="{xm:.1f}" y2="{bottom}" stroke="var(--line)"/>')
            label = MONTHS_RU[month.month - 1] + (f" {month.year}" if month.month == 1 or first else "")
            out.append(f'<text x="{xm + 3:.1f}" y="{label_y}">{label}</text>')
            first = False
        month += pd.offsets.MonthBegin(1)
    return out


def _steps(points: list[tuple[float, float]]) -> str:
    """Ступенчатая линия: значение держится до следующего наблюдения."""
    out = [f"{points[0][0]:.1f},{points[0][1]:.1f}"]
    for (_, y0), (x1, y1) in zip(points, points[1:]):
        out.append(f"{x1:.1f},{y0:.1f}")
        out.append(f"{x1:.1f},{y1:.1f}")
    return " ".join(out)


def _outcome_ra_section(result) -> list[str]:
    """Медленный контур с риск-коррекцией по DT: VLAD, RA-CUSUM, O/E по месяцам и разрезам.

    Тот же язык, что у карт стадий: одна ось дат на панели «по переносам», под ней
    O/E по месяцам с двумя уровнями пределов, затем forest plot по разрезам. Цифры
    берутся из результата outcome_ra как есть — панель ничего не пересчитывает.
    """
    summary = result.summary()
    parts = ["<h2>Медленный контур с поправкой на пациенток: исход против ожидания DT</h2>"]
    if not summary["transfers_with_outcome"]:
        parts.append('<div class="note">Переносов с известным исходом и ожиданием DT '
                     'нет — контур не построен.</div>')
        return parts
    overall, cusum, calib = summary["overall"], summary["cusum"], summary["calibration"]
    transfers = result.transfers.copy()
    transfers["date_opu"] = pd.to_datetime(transfers["date_opu"])
    transfers = transfers.sort_values("date_opu", kind="stable").reset_index(drop=True)

    # --- шапка ---
    oe = overall.get("oe")
    all_row = result.strata.loc[result.strata["dimension"] == "всего"].iloc[0]
    position = str(overall.get("position") or "")
    pill_class = {"ниже внешнего предела": "alert", "ниже внутреннего предела": "watch"}.get(position, "calm")
    chips = [("переносов с исходом", str(summary["transfers_with_outcome"])),
             ("факт / ожидание", f"{overall['observed']} / {overall['expected']}"),
             ("O/E", f"{oe:.2f}" if oe is not None else "—"),
             ("пределы 95%", f"{all_row['inner_low']:.2f}–{all_row['inner_high']:.2f}"
              if pd.notna(all_row.get("inner_low")) else "—"),
             ("VLAD сейчас", f"{summary['vlad_last']:+.1f}"),
             ("RA-CUSUM", f"{cusum['last']:.2f} из {cusum['h']:.2f}" if cusum["h"] is not None else f"{cusum['last']:.2f}"),
             ("сигналов", str(cusum["signals"]))]
    chips_html = "".join(f"<span>{escape(k)} <b>{escape(v)}</b></span>" for k, v in chips)
    calib_text = (f"сдвиг δ {calib['delta']:+.2f} (SE {calib['se']:.2f}) заморожен по {calib['n']} исходам"
                  if calib["frozen"] else "сдвиг δ ещё не заморожен — ожидание DT как есть")

    W, L, R = 1000, 46, 16
    start = transfers["date_opu"].min() - pd.Timedelta(days=5)
    end = transfers["date_opu"].max() + pd.Timedelta(days=5)
    span = max((end - start).days, 1)

    def x(ts: pd.Timestamp) -> float:
        return L + (pd.Timestamp(ts) - start).days / span * (W - L - R)

    # --- VLAD ---
    H, T, B = 200, 12, 26
    vlad = transfers["vlad"].astype(float).tolist()
    top_v, bottom_v = max(max(vlad), 0.0) + 0.6, min(min(vlad), 0.0) - 0.6

    def yv(value: float) -> float:
        return T + (top_v - value) / (top_v - bottom_v) * (H - T - B)

    pts = [(x(d), yv(v)) for d, v in zip(transfers["date_opu"], vlad)]
    pts = [(x(start), yv(0.0))] + pts
    line = _steps(pts)
    zero = yv(0.0)
    area = f"{pts[0][0]:.1f},{zero:.1f} {line} {pts[-1][0]:.1f},{zero:.1f}"
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="VLAD">',
           '<defs><clipPath id="vl-up"><rect x="0" y="0" width="1000" height="' f'{zero:.1f}"/></clipPath>'
           f'<clipPath id="vl-dn"><rect x="0" y="{zero:.1f}" width="1000" height="{H}"/></clipPath></defs>']
    svg += _month_grid(start, end, x, T, H - B, H - 8)
    if calib["frozen"] and calib.get("n"):
        last = getattr(result.calibration, "last_date_opu", None)
        if last:
            xc = x(pd.Timestamp(last))
            svg.append(f'<rect x="{x(start):.1f}" y="{H - B + 1}" width="{xc - x(start):.1f}" height="4" '
                       f'fill="var(--clay)" opacity=".35"><title>калибровка δ: первые {calib["n"]} исходов, '
                       f'до {pd.Timestamp(last):%d.%m.%Y}</title></rect>')
            svg.append(f'<line x1="{xc:.1f}" y1="{T}" x2="{xc:.1f}" y2="{H - B}" stroke="var(--clay)" '
                       f'stroke-dasharray="3 3"/><text x="{xc + 4:.1f}" y="{T + 10}">δ заморожен</text>')
    svg.append(f'<polygon points="{area}" fill="#3d8b5f" opacity=".16" clip-path="url(#vl-up)"/>')
    svg.append(f'<polygon points="{area}" fill="#c0392b" opacity=".16" clip-path="url(#vl-dn)"/>')
    svg.append(f'<line x1="{L}" y1="{zero:.1f}" x2="{W - R}" y2="{zero:.1f}" stroke="var(--muted)" '
               f'stroke-dasharray="4 4"/><text x="{L - 6}" y="{zero + 4:.1f}" text-anchor="end">0</text>')
    for value in (top_v - 0.6, bottom_v + 0.6):
        if abs(value) > 0.05:
            svg.append(f'<text x="{L - 6}" y="{yv(value) + 4:.1f}" text-anchor="end">{value:+.1f}</text>')
    svg.append(f'<polyline points="{line}" fill="none" stroke="var(--clay)" stroke-width="1.8"/>')
    for (px, py), (_, row) in zip(pts[1:], transfers.iterrows()):
        pregnant = float(row["y"]) == 1.0
        tip = (f"{row['date_opu']:%d.%m.%Y} · {'беременность' if pregnant else 'нет беременности'} · "
               f"ожидание p* {float(row['p_expected']):.2f} · VLAD {float(row['vlad']):+.2f}")
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{"var(--clay)" if pregnant else "var(--card)"}" '
                   f'stroke="var(--clay)" stroke-width="1.4"><title>{escape(tip)}</title></circle>')
    svg.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="var(--muted)" stroke-width=".8"/></svg>')

    # --- RA-CUSUM ---
    H2, T2 = 130, 10
    values = transfers["cusum"].astype(float).tolist()
    h = cusum["h"]
    top_c = max(values + [h or 0.0, 0.5]) * 1.12

    def yc(value: float) -> float:
        return T2 + (1 - value / top_c) * (H2 - T2 - B)

    cpts = [(x(start), yc(0.0))] + [(x(d), yc(v)) for d, v in zip(transfers["date_opu"], values)]
    cline = _steps(cpts)
    csvg = [f'<svg viewBox="0 0 {W} {H2}" role="img" aria-label="RA-CUSUM">']
    csvg += _month_grid(start, end, x, T2, H2 - B, H2 - 8)
    if h:
        csvg.append(f'<rect x="{L}" y="{T2}" width="{W - L - R}" height="{yc(h) - T2:.1f}" fill="#c0392b" opacity=".07"/>'
                    f'<line x1="{L}" y1="{yc(h):.1f}" x2="{W - R}" y2="{yc(h):.1f}" stroke="#c0392b" '
                    f'stroke-dasharray="5 4"/><text x="{L + 6}" y="{yc(h) - 4:.1f}" style="fill:#c0392b">'
                    f'порог h {h:.2f}</text>')
    csvg.append(f'<text x="{L - 6}" y="{yc(0) + 4:.1f}" text-anchor="end">0</text>')
    area_c = f"{cpts[0][0]:.1f},{yc(0):.1f} {cline} {cpts[-1][0]:.1f},{yc(0):.1f}"
    csvg.append(f'<polygon points="{area_c}" fill="var(--clay)" opacity=".13"/>'
                f'<polyline points="{cline}" fill="none" stroke="var(--clay)" stroke-width="1.6"/>')
    for (px, _), (_, row) in zip(cpts[1:], transfers.iterrows()):
        if bool(row.get("cusum_signal")):
            csvg.append(f'<path d="M{px:.1f},{yc(h or 0) - 13:.1f} l6,10 h-12 z" fill="#c0392b">'
                        f'<title>{row["date_opu"]:%d.%m.%Y}: сигнал RA-CUSUM</title></path>')
    csvg.append(f'<line x1="{L}" y1="{H2 - B}" x2="{W - R}" y2="{H2 - B}" stroke="var(--muted)" stroke-width=".8"/></svg>')

    # --- O/E по месяцам пункции ---
    months = result.strata.loc[result.strata["dimension"] == "месяц"].copy()
    H3, T3 = 170, 10
    oe_top = 2.0

    def yo(value: float) -> float:
        return T3 + (1 - min(max(value, 0.0), oe_top) / oe_top) * (H3 - T3 - B)

    msvg = [f'<svg viewBox="0 0 {W} {H3}" role="img" aria-label="O/E по месяцам">']
    msvg += _month_grid(start, end, x, T3, H3 - B, H3 - 8)
    for tick in (0.5, 1.0, 1.5):
        msvg.append(f'<text x="{L - 6}" y="{yo(tick) + 4:.1f}" text-anchor="end">{tick:.1f}</text>')
    msvg.append(f'<line x1="{L}" y1="{yo(1.0):.1f}" x2="{W - R}" y2="{yo(1.0):.1f}" stroke="var(--clay)" '
                f'stroke-dasharray="6 4"><title>O/E = 1: факт как ожидалось для этих пациенток</title></line>')
    for _, row in months.iterrows():
        m0 = pd.Period(str(row["group"]), freq="M").to_timestamp()
        xc = (max(x(m0), L) + min(x(m0 + pd.offsets.MonthBegin(1)), W - R)) / 2
        few = row["position"] == "мало исходов"
        colour = POSITION_COLOUR.get(row["position"], "var(--muted)")
        if pd.notna(row.get("outer_low")):
            msvg.append(f'<line x1="{xc:.1f}" y1="{yo(row["outer_low"]):.1f}" x2="{xc:.1f}" y2="{yo(row["outer_high"]):.1f}" '
                        f'stroke="var(--muted)" stroke-width="1" opacity=".6"/>')
            msvg.append(f'<line x1="{xc:.1f}" y1="{yo(row["inner_low"]):.1f}" x2="{xc:.1f}" y2="{yo(row["inner_high"]):.1f}" '
                        f'stroke="var(--muted)" stroke-width="4" stroke-linecap="round" opacity=".35"/>')
        tip = (f"{row['group']}: исходов {row['n']}, факт {row['observed']}, ожидание {row['expected']}, "
               f"O/E {row['oe']:.2f} · {row['position']}")
        msvg.append(f'<circle cx="{xc:.1f}" cy="{yo(row["oe"]):.1f}" r="6" fill="{colour}" stroke="var(--card)" '
                    f'stroke-width="1.5" opacity="{"0.5" if few else "1"}"><title>{escape(tip)}</title></circle>'
                    f'<text x="{xc + 10:.1f}" y="{yo(row["oe"]) + 4:.1f}">{row["oe"]:.2f} · n {row["n"]}</text>')
    msvg.append(f'<line x1="{L}" y1="{H3 - B}" x2="{W - R}" y2="{H3 - B}" stroke="var(--muted)" stroke-width=".8"/></svg>')

    parts.append(
        f'<div class="card cc"><div class="cc-head"><span class="name">Исход / DT</span>'
        f'<span class="label">клиническая беременность на перенос против ожидания Digital Twin (L7 BEFE)</span>'
        f'<span class="pill {pill_class}">{escape(position)}</span></div>'
        f'<div class="chips">{chips_html}</div><div class="cap">{escape(calib_text)}</div>'
        f'<div class="cap">VLAD — накопленная разница «факт − ожидание» по переносам в порядке пункции. '
        f'Вверх — беременностей больше ожидаемого для этих пациенток, вниз — меньше; '
        f'● беременность, ○ нет</div>{"".join(svg)}'
        f'<div class="cap">RA-CUSUM на снижение шансов беременности в {1 / result.odds_ratio:.0f} раза</div>'
        f'{"".join(csvg)}'
        f'<div class="cap">O/E по месяцам пункции: точка, внутренний (95%, толстый) и внешний (99.8%) '
        f'пределы; бледная — меньше 10 исходов</div>{"".join(msvg)}</div>')

    # --- forest plot по разрезам ---
    def forest(rows: pd.DataFrame) -> str:
        row_h, top_pad = 26, 22
        Hf = top_pad + row_h * len(rows) + 10
        LX, RX = 300, 980
        cap = 2.5

        def xf(value: float) -> float:
            return LX + min(max(value, 0.0), cap) / cap * (RX - LX)

        out = [f'<svg viewBox="0 0 1000 {Hf}" role="img" aria-label="O/E по разрезам">']
        for tick in (0.5, 1.0, 1.5, 2.0):
            stroke = 'var(--clay)" stroke-dasharray="6 4' if tick == 1.0 else "var(--line)"
            out.append(f'<line x1="{xf(tick):.1f}" y1="{top_pad - 6}" x2="{xf(tick):.1f}" y2="{Hf - 6}" '
                       f'stroke="{stroke}"/>'
                       f'<text x="{xf(tick):.1f}" y="{top_pad - 9}" text-anchor="middle">{tick:.1f}</text>')
        previous = None
        for i, (_, row) in enumerate(rows.iterrows()):
            yy = top_pad + row_h * i + row_h / 2
            dim = str(row["dimension"])
            if dim != previous:
                out.append(f'<line x1="0" y1="{yy - row_h / 2:.1f}" x2="1000" y2="{yy - row_h / 2:.1f}" stroke="var(--line)"/>')
                previous = dim
                head = dim.replace("лот: ", "")
            else:
                head = ""
            few = row["position"] == "мало исходов"
            colour = POSITION_COLOUR.get(row["position"], "var(--muted)")
            out.append(f'<text x="4" y="{yy + 4:.1f}" style="fill:var(--ink);font-weight:600">{escape(head[:26])}</text>'
                       f'<text x="160" y="{yy + 4:.1f}">{escape(str(row["group"])[:20])}</text>'
                       f'<text x="{LX - 8}" y="{yy + 4:.1f}" text-anchor="end">n {row["n"]}</text>')
            if pd.notna(row.get("outer_low")):
                out.append(f'<line x1="{xf(row["outer_low"]):.1f}" y1="{yy:.1f}" x2="{xf(row["outer_high"]):.1f}" '
                           f'y2="{yy:.1f}" stroke="var(--muted)" stroke-width="1" opacity=".6"/>'
                           f'<line x1="{xf(row["inner_low"]):.1f}" y1="{yy:.1f}" x2="{xf(row["inner_high"]):.1f}" '
                           f'y2="{yy:.1f}" stroke="var(--muted)" stroke-width="5" stroke-linecap="round" opacity=".3"/>')
            tip = (f"{dim} · {row['group']}: исходов {row['n']}, факт {row['observed']}, ожидание "
                   f"{row['expected']}, O/E {row['oe']:.2f} · {row['position']}")
            out.append(f'<circle cx="{xf(row["oe"]):.1f}" cy="{yy:.1f}" r="5.5" fill="{colour}" stroke="var(--card)" '
                       f'stroke-width="1.5" opacity="{"0.5" if few else "1"}"><title>{escape(tip)}</title></circle>')
            if row["oe"] > cap:
                out.append(f'<text x="{RX + 4}" y="{yy + 4:.1f}">›</text>')
        out.append("</svg>")
        return "".join(out)

    strata = result.strata
    main = strata.loc[strata["dimension"].isin(["всего", "эмбриолог", "среда инкубатора"])]
    lots = strata.loc[strata["dimension"].astype(str).str.startswith("лот:")]
    parts.append('<div class="card cc"><div class="cc-head"><span class="name">O/E по разрезам</span>'
                 '<span class="label">где беременностей меньше или больше ожидаемого для этих пациенток</span></div>'
                 f'{forest(main)}')
    if not lots.empty:
        parts.append(f'<details style="margin-top:8px"><summary class="muted" style="cursor:pointer">'
                     f'Лоты расходников ({lots["dimension"].nunique()} позиций, {len(lots)} групп)</summary>'
                     f'{forest(lots)}</details>')

    parts.append('<details style="margin-top:8px"><summary class="muted" style="cursor:pointer">Таблица всех разрезов</summary>'
                 '<div class="scroll"><table><tr><th>разрез</th><th>группа</th>'
                 '<th class="num">n</th><th class="num">факт</th><th class="num">ожидание</th>'
                 '<th class="num">O/E</th><th class="num">пределы 95%</th>'
                 '<th class="num">пределы 99.8%</th><th>положение</th></tr>')
    for row in strata.to_dict("records"):
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
    parts.append("</table></div></details></div>")

    parts.append('<div class="card"><div class="legend">'
                 '<span><s style="background:#c0392b;border-radius:50%"></s>ниже внешнего предела</span>'
                 '<span><s style="background:#c98a1e;border-radius:50%"></s>ниже внутреннего</span>'
                 '<span><s style="background:var(--muted);border-radius:50%"></s>в пределах</span>'
                 '<span><s style="background:#3d8b5f;border-radius:50%"></s>выше предела</span>'
                 '<span><s style="background:var(--muted);opacity:.45;border-radius:50%"></s>мало исходов (&lt; 10)</span>'
                 '<span><i class="dash" style="border-color:var(--clay)"></i>O/E = 1</span>'
                 '</div></div>')
    notes = [summary["calibration"]["note"], summary["note"], *summary["notes"]]
    parts.append('<div class="note"><b>Как читать.</b> Ожидание — итоговая вероятность '
                 'Digital Twin для каждой пациентки, сдвинутая на уровень клиники '
                 '(сдвиг заморожен и не подстраивается под факт). O/E ниже предела — '
                 'беременностей меньше, чем ждали для ЭТИХ пациенток; это повод посмотреть '
                 'на группу, не вывод о лоте или специалисте. Групп много, и часть выходов за '
                 '95% ожидаема случайно. Контур тревог не поднимает.'
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
           envelope_qc=None, coverage_note: str = "", outcome_ra=None,
           alerts_journal=None) -> Path:
    environment = cfg.environment
    block = cfg.alert_block_reason()
    counts = frame.data["checkpoint"].value_counts().to_dict()
    levels = pd.Series([e.level for e in evaluations]).value_counts().to_dict()

    badge = ('<span class="badge draft">ЧЕРНОВОЙ СТЕНД</span>' if environment == "draft"
             else '<span class="badge prod">лаборатория</span>')

    parts: list[str] = [
        '<!doctype html><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
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
    contours = cfg.settings["localisation"]["contours"]
    try:
        from .vendor import kpi as kpi_mod  # noqa: PLC0415
        specs = kpi_mod.KPI_SPECS
    except ImportError:
        specs = {}
    window = clinic.window_bounds()
    for metric in clinic.cusum_charts:
        state = states.get(metric)
        if state is None:
            continue
        contour = next((name for name, items in contours.items() if metric in items), "—")
        base = dict(baselines.get(metric, {}))
        if metric in specs:
            base.setdefault("label", specs[metric].label)
            base.setdefault("direction", specs[metric].direction)
        parts.append(_control_chart_card(
            metric, contour, state,
            thresholds.get("h_watch", {}).get(metric), thresholds.get("h_alert", {}).get(metric),
            _chart_events(alerts_journal, metric), _monthly_rates(frame.data, metric),
            base, window))

    parts.append(
        '<div class="card"><div class="legend">'
        '<span><i style="border-color:var(--clay)"></i>CUSUM — накопленное свидетельство</span>'
        '<span><i class="dash" style="border-color:#c98a1e"></i>порог watch</span>'
        '<span><i class="dash" style="border-color:#c0392b"></i>порог alert</span>'
        '<span>● эпизод watch</span><span style="color:#c0392b">▲ пересечение alert</span>'
        '<span><s style="background:var(--clay);opacity:.35"></s>опорное окно</span>'
        '<span><i class="dash" style="border-color:var(--clay)"></i>p0</span>'
        '<span><i class="dot" style="border-color:#3d8b5f"></i>benchmark</span>'
        '<span><i class="dot" style="border-color:#c0392b"></i>competency</span>'
        '<span><s style="background:#3d8b5f;border-radius:50%"></s>месяц не хуже benchmark</span>'
        '<span><s style="background:#c98a1e;border-radius:50%"></s>между competency и benchmark</span>'
        '<span><s style="background:#c0392b;border-radius:50%"></s>хуже competency</span>'
        f'<span><s style="background:var(--muted);opacity:.45;border-radius:50%"></s>месяц &lt; {MIN_MONTH_CYCLES} циклов — не интерпретируется</span>'
        '</div></div>')
    parts.append('<div class="note"><b>Как читать.</b> Верхняя панель — не уровень '
                 'показателя, а накопленное свидетельство против опорного уровня p0. После '
                 'пересечения alert статистика обнуляется, в спокойные периоды разряжается сама. '
                 'Поэтому «сейчас» почти всегда близко к нулю, и смысл несут траектория, пик и '
                 'маркеры событий (подсказка — при наведении). Нижняя панель на той же оси — сама '
                 'доля: она показывает, какой уровень стоит за накоплением. Месяц ниже benchmark '
                 'без эпизода CUSUM укладывается в разброс пациенток. Эпизод CUSUM без провала '
                 'месяца — свидетельство, собранное по нескольким месяцам.</div>')

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
    parts.append(f"<h2>Тревоги этого прогона ({len(raised)})</h2>")
    if not raised:
        parts.append('<div class="card muted">В этом прогоне новых тревог нет. Это '
                     'нормальный и самый частый исход — система, которая сообщает что-то '
                     'каждый день, перестаёт читаться. История — в журнале ниже.</div>')
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

    # --- журнал за 90 дней: при инкрементальном прогоне выше видны только новые ---
    if alerts_journal is not None and not alerts_journal.empty:
        recent = alerts_journal.copy()
        recent["_date"] = pd.to_datetime(recent["raised_at"], errors="coerce")
        recent = recent.loc[recent["_date"] >= pd.Timestamp(datetime.now()) - pd.Timedelta(days=90)]
        recent = recent.sort_values("_date", ascending=False)
        parts.append(f"<h2>Журнал за 90 дней ({len(recent)})</h2>")
        if recent.empty:
            parts.append('<div class="card muted">За 90 дней записей нет.</div>')
        else:
            parts.append('<div class="card scroll"><table><tr><th>дата закрытия</th>'
                         '<th>уровень</th><th>карта</th><th>контур</th><th>подтверждение</th>'
                         '<th>среда</th><th>примечание</th><th>cycle_uid</th></tr>')
            for _, row in recent.iterrows():
                notes = "" if pd.isna(row.get("notes")) else str(row["notes"])
                miri = {"pending": "ожидается", "unavailable": "нет выгрузки",
                        "enriched": "разобрана"}.get(row.get("miri_status"), row.get("miri_status"))
                parts.append(
                    f"<tr><td>{escape(str(row['raised_at']))}</td>"
                    f"<td>{_level_cell(str(row['level']))}</td>"
                    f"<td><b>{escape(str(row['chart']))}</b></td>"
                    f"<td class=\"muted\">{escape(CONTOUR_RU.get(str(row.get('contour')), str(row.get('contour'))))}</td>"
                    f"<td class=\"muted\">{escape(str(row.get('corroboration_rank')))}</td>"
                    f"<td class=\"muted\">{escape(str(miri))}</td>"
                    f"<td class=\"muted\" style=\"font-size:12px\">{escape(notes)}</td>"
                    f"<td class=\"muted\">{escape(str(row['cycle_uid']))}</td></tr>")
            parts.append("</table></div>")

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
        ("watch в этом прогоне", str(levels.get("watch", 0)), "на панель, никого не будит"),
        ("alert в этом прогоне", str(levels.get("alert", 0)),
         block or "инициирует разбор; история — в журнале за 90 дней"),
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
