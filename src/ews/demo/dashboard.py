"""Интерактивная HTML-панель KPI для демонстрации (Plotly, автономный файл).

Стиль — как у рисунков публикации CSBJ (Figure 7): DejaVu Sans, открытые верхняя и
правая оси, палитра navy / rust / purple по контурам (эмбриолог / культивирование /
врач), пунктир — порог компетентности, штрих — бенчмарк, буквенные метки панелей.

Панель читает человек на презентации: одна мысль на вкладку, 1–3 графика, короткий
текст «что показано», факты отдельно от гипотез.
"""
from __future__ import annotations

import json
import re
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..vendor import kpi
from . import stats as st
from .report import month_ru, pct, pval

NAVY, RUST, PURPLE, AQUA = "#2c4a5e", "#c0492c", "#7b4fa8", "#2e7d8a"
YELLOW, GREEN, GREY = "#b8860b", "#3f7e5a", "#8c8c8c"
LNAVY, LRUST, LPURPLE, LGREY = "#cfd8dd", "#edc7bf", "#d6c9e3", "#e3e3e3"
MRUST, INK, INK2 = "#d07863", "#222222", "#555555"
FONT = "DejaVu Sans, Segoe UI, Helvetica, Arial, sans-serif"
CATEGORICAL = [NAVY, RUST, AQUA, PURPLE, YELLOW, GREEN, "#4a6fa5", "#b0503c", "#5b4b9a"]
STATUS_COLOR = {"benchmark": AQUA, "competency": YELLOW, "outside_control": RUST,
                "insufficient_data": GREY, "no_threshold": GREY}
CONTOUR = {"OCCR": "doctor", "MIIR": "doctor", "FR": "embryologist", "DegR": "embryologist",
           "3pNR": "embryologist", "ClR": "culture", "BlR": "culture", "TGBDR": "culture"}
CONTOUR_COLOR = {"doctor": PURPLE, "embryologist": NAVY, "culture": RUST}
CONTOUR_FILL = {"doctor": LPURPLE, "embryologist": LNAVY, "culture": LRUST}
CONTOUR_RU = {"doctor": "контур врача", "embryologist": "контур эмбриолога",
              "culture": "контур культивирования"}
METHOD_COLOR = {"ICSI": NAVY, "IVF": RUST, "IVF+ICSI": PURPLE}

PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
               "toImageButtonOptions": {"format": "png", "scale": 3}}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _axis(**kw) -> dict:
    base = dict(showline=True, linecolor=INK, linewidth=1, ticks="outside", tickcolor=INK,
                ticklen=4, showgrid=False, zeroline=False, mirror=False,
                title_font=dict(size=12.5), tickfont=dict(size=11.5))
    base.update(kw)
    return base


def _layout(fig: go.Figure, *, height: int = 380, title: str = "", letter: str = "",
            legend: bool = True, **kw) -> go.Figure:
    text = f"<b style='font-size:17px'>{letter}</b>&nbsp;&nbsp;{title}" if letter else title
    fig.update_layout(
        template="simple_white", height=height, font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=64, r=24, t=52 if title else 20, b=52),
        title=dict(text=text, x=0.0, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=14.5, color=INK)),
        hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=GREY),
        showlegend=legend,
        legend=dict(bgcolor="rgba(255,255,255,0)", borderwidth=0, font=dict(size=11.5)),
        **kw)
    fig.update_xaxes(**_axis())
    fig.update_yaxes(**_axis())
    return fig


def _thresholds(fig: go.Figure, metric_spec, *, scale: float = 100, row=None, col=None,
                label: bool = True) -> None:
    for value, dash, name in ((metric_spec.competency, "dot", "компетентность"),
                              (metric_spec.benchmark, "dash", "бенчмарк")):
        if value is None:
            continue
        kwargs = dict(y=value * scale, line=dict(color=GREY, dash=dash, width=1.3))
        if label:
            kwargs.update(annotation_text=f"{name} {value * scale:.0f}%",
                          annotation_position="top right",
                          annotation_font=dict(size=10.5, color=GREY))
        if row is not None:
            kwargs.update(row=row, col=col)
        fig.add_hline(**kwargs)


MONTH_SHORT = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def _ru_months(fig: go.Figure, dates, *, row=None, col=None) -> None:
    """Подписи месяцев по-русски: plotly без локали печатает Jan/Feb."""
    d = pd.to_datetime(pd.Series(list(dates))).dropna()
    if d.empty:
        return
    start = d.min().to_period("M").to_timestamp()
    ticks = pd.date_range(start, d.max() + pd.offsets.MonthBegin(1), freq="MS")
    step = 1 if len(ticks) <= 12 else 2 if len(ticks) <= 24 else 3
    ticks = ticks[::step]
    text = [f"{MONTH_SHORT[t.month - 1]}<br>{t.year}" if (i == 0 or t.month <= step) else MONTH_SHORT[t.month - 1]
            for i, t in enumerate(ticks)]
    kwargs = dict(tickvals=list(ticks), ticktext=text)
    if row is not None:
        kwargs.update(row=row, col=col)
    fig.update_xaxes(**kwargs)


def _div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG, div_id=div_id)


def _empty(message: str) -> str:
    return f'<div class="empty">{escape(message)}</div>'


# --------------------------------------------------------------------------
# Сводка
# --------------------------------------------------------------------------
def fig_kpi_bullet(summary: pd.DataFrame) -> go.Figure:
    s = summary[summary["cycles"] > 0].copy()
    s = s.iloc[::-1].reset_index(drop=True)
    fig = go.Figure()
    labels = [f"{r.label_ru}" for r in s.itertuples()]
    for i, r in enumerate(s.itertuples()):
        lo_t = min(v for v in (r.competency, r.benchmark) if v is not None) * 100
        hi_t = max(v for v in (r.competency, r.benchmark) if v is not None) * 100
        fig.add_shape(type="rect", x0=lo_t, x1=hi_t, y0=i - 0.32, y1=i + 0.32,
                      fillcolor="#e9edf0", line=dict(width=0), layer="below")
        for value, dash in ((r.competency, "dot"), (r.benchmark, "dash")):
            fig.add_shape(type="line", x0=value * 100, x1=value * 100, y0=i - 0.38, y1=i + 0.38,
                          line=dict(color=INK2, dash=dash, width=1.5))
    color = [STATUS_COLOR.get(v, GREY) for v in s["status"]]
    err_plus = (s["ci_high_phi"] - s["rate"]) * 100
    err_minus = (s["rate"] - s["ci_low_phi"]) * 100
    hover = [(f"<b>{r.label_ru}</b><br>{r.definition}<br>"
              f"Σ = {r.numerator:,.0f} / {r.denominator:,.0f} ({r.cycles} цикл.)<br>"
              f"доля {pct(r.rate)}; 95% ДИ {pct(r.ci_low_phi)}–{pct(r.ci_high_phi)} (φ = {r.phi:.2f})<br>"
              f"компетентность {pct(r.competency, 0)} · бенчмарк {pct(r.benchmark, 1)}<br>"
              f"<i>{st.STATUS_RU[r.status]}</i><br><span style='font-size:10px'>{r.source_note}</span>")
             .replace(",", " ") for r in s.itertuples()]
    fig.add_trace(go.Scatter(
        x=s["rate"] * 100, y=list(range(len(s))), mode="markers",
        marker=dict(size=13, color=color, line=dict(color="white", width=1.5), symbol="circle"),
        error_x=dict(type="data", array=err_plus.fillna(0), arrayminus=err_minus.fillna(0),
                     color=INK2, thickness=1.4, width=0),
        hovertext=hover, hoverinfo="text", showlegend=False))
    for i, r in enumerate(s.itertuples()):
        fig.add_annotation(x=1.0, xref="paper", y=i, text=f"<b>{pct(r.rate)}</b>", showarrow=False,
                           xanchor="left", font=dict(size=12, color=STATUS_COLOR.get(r.status, GREY)))
    for status, name in (("benchmark", "бенчмарк достигнут"), ("competency", "уровень компетентности"),
                         ("outside_control", "за порогом компетентности")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(size=11, color=STATUS_COLOR[status])))
    xmax = max(105.0, float(np.nanmax(s["ci_high_phi"] * 100)) + 3 if len(s) else 105.0)
    _layout(fig, height=110 + 46 * len(s), letter="A",
            title="KPI лаборатории против порогов Венского консенсуса", legend=True)
    fig.update_layout(margin=dict(l=250, r=70, t=84, b=56),
                      legend=dict(orientation="h", y=1.0, x=0, yanchor="bottom"))
    fig.update_yaxes(tickvals=list(range(len(s))), ticktext=labels, showline=False, ticks="")
    fig.update_xaxes(range=[0, xmax], title="доля, % (Σ числитель / Σ знаменатель; линия — 95% ДИ)",
                     ticksuffix="%")
    return fig


def fig_violin_overall(dist: dict, summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    order = [m for m in st.DISPLAY_ORDER if m in dist and not dist[m].empty]
    for m in order:
        d = dist[m]
        contour = CONTOUR[m]
        fig.add_trace(go.Violin(
            x=[m] * len(d), y=d["rate"] * 100, name=m, line=dict(color=CONTOUR_COLOR[contour], width=1.2),
            fillcolor=CONTOUR_FILL[contour], opacity=0.9, box=dict(visible=True, width=0.18,
                                                                     fillcolor="white", line=dict(color=INK2)),
            meanline=dict(visible=False), points=False, spanmode="hard", showlegend=False,
            hoveron="violins+kde", scalemode="width", width=0.8,
            hovertemplate=f"{m}: %{{y:.1f}}%<extra>{len(d)} цикл.</extra>"))
    s = summary.set_index("metric")
    for m in order:
        r = s.loc[m]
        fig.add_trace(go.Scatter(x=[m], y=[r["rate"] * 100], mode="markers", showlegend=False,
                                 marker=dict(symbol="diamond", size=11, color=INK, line=dict(color="white", width=1)),
                                 hovertemplate=f"{m}: Σ/Σ = {pct(r['rate'])}<extra></extra>"))
        for value, dash in ((r["competency"], "dot"), (r["benchmark"], "dash")):
            fig.add_shape(type="line", xref="x", x0=order.index(m) - 0.42, x1=order.index(m) + 0.42,
                          y0=value * 100, y1=value * 100, line=dict(color=INK2, dash=dash, width=1.4))
    for contour in ("doctor", "embryologist", "culture"):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=CONTOUR_RU[contour],
                                 marker=dict(symbol="square", size=11, color=CONTOUR_COLOR[contour])))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="доля лаборатории (Σ/Σ)",
                             marker=dict(symbol="diamond", size=10, color=INK)))
    _layout(fig, height=430, letter="B", title=f"Распределение KPI по циклам (знаменатель ≥ {st.MIN_DEN_DISPLAY})")
    fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0, yanchor="top"), violingap=0.25)
    fig.update_yaxes(title="доля в цикле, %", ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(title="")
    return fig


# --------------------------------------------------------------------------
# Динамика
# --------------------------------------------------------------------------
def fig_dynamics(metric: str, e: dict, cusum: dict | None, window: int) -> go.Figure:
    s = st.spec(metric)
    color = CONTOUR_COLOR[CONTOUR[metric]]
    fill = CONTOUR_FILL[CONTOUR[metric]]
    fig = go.Figure()
    roll = e["rolling"]
    if not roll.empty:
        fig.add_trace(go.Scatter(x=pd.concat([roll["date"], roll["date"][::-1]]),
                                 y=pd.concat([roll["hi"], roll["lo"][::-1]]) * 100, fill="toself",
                                 fillcolor=fill, line=dict(width=0), opacity=0.6, hoverinfo="skip",
                                 name="95% ДИ скользящей доли"))
        fig.add_trace(go.Scatter(x=roll["date"], y=roll["rate"] * 100, mode="lines",
                                 line=dict(color=color, width=2.2), name=f"скользящая доля, {window} цикл.",
                                 customdata=np.column_stack([roll["lo"] * 100, roll["hi"] * 100, roll["n"]]),
                                 hovertemplate="%{x|%d.%m.%Y}: %{y:.1f}% [%{customdata[0]:.1f}–%{customdata[1]:.1f}]"
                                               "<br>знаменатель окна %{customdata[2]:.0f}<extra></extra>"))
    mon = e["monthly"]
    if not mon.empty:
        mid = pd.PeriodIndex(mon["month"], freq="M").to_timestamp() + pd.Timedelta(days=14)
        fig.add_trace(go.Scatter(
            x=mid, y=mon["rate"] * 100, mode="markers", name="месяц (Σ/Σ)",
            marker=dict(size=9, color="white", line=dict(color=INK, width=1.6)),
            error_y=dict(type="data", array=(mon["hi"] - mon["rate"]) * 100,
                         arrayminus=(mon["rate"] - mon["lo"]) * 100, color=INK2, thickness=1.1, width=3),
            customdata=np.column_stack([mon["cycles"], mon["x"], mon["n"]]),
            hovertemplate="%{x|%b %Y}: %{y:.1f}%<br>%{customdata[1]:.0f}/%{customdata[2]:.0f}, "
                          "%{customdata[0]:.0f} цикл.<extra></extra>"))
    _thresholds(fig, s)
    if cusum and cusum.get("available") and not cusum["events"].empty:
        ev = cusum["events"]
        fig.add_trace(go.Scatter(x=ev["date"], y=[2] * len(ev), mode="markers", name="сигнал CUSUM (watch)",
                                 marker=dict(symbol="triangle-up", size=12, color=RUST),
                                 hovertemplate="watch %{x|%d.%m.%Y}<extra></extra>"))
    _layout(fig, height=420, letter="A", title=f"{st.METRIC_RU[metric]}: скользящая доля и помесячные значения")
    fig.update_layout(legend=dict(orientation="h", y=-0.16, x=0, yanchor="top"))
    fig.update_yaxes(title=f"{metric}, %", ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(title="дата пункции")
    dates = []
    if not roll.empty:
        dates += list(roll["date"])
    if not mon.empty:
        dates += list(mid)
    _ru_months(fig, dates)
    return fig


def fig_heat(heat: pd.DataFrame) -> go.Figure | None:
    if heat.empty:
        return None
    metrics = [m for m in st.DISPLAY_ORDER if m in set(heat["metric"])]
    months = sorted(heat["month"].unique())
    z = np.full((len(metrics), len(months)), np.nan)
    text = np.full(z.shape, "", dtype=object)
    hover = np.full(z.shape, "", dtype=object)
    for r in heat.itertuples():
        i, j = metrics.index(r.metric), months.index(r.month)
        z[i, j] = r.worse_pp
        star = "*" if r.q < 0.05 else ""
        text[i, j] = f"{100 * r.rate_a:.0f}{star}"
        hover[i, j] = (f"<b>{r.metric}, {r.month}</b><br>{pct(r.rate_a)} против {pct(r.rate_b)} в остальные "
                       f"месяцы<br>разность {r.diff_pp:+.1f} п.п. [{r.lo_pp:+.1f}; {r.hi_pp:+.1f}]<br>"
                       f"{r.test}, {pval(r.p)}, q(BH) = {r.q:.3f}<br>{r.cycles} цикл.")
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 10.0
    lim = max(min(lim, 30.0), 5.0)
    fig = go.Figure(go.Heatmap(
        z=z, x=[month_ru(pd.Period(m).to_timestamp()) for m in months], y=metrics, text=text,
        texttemplate="%{text}", textfont=dict(size=11), hovertext=hover, hoverinfo="text",
        colorscale=[[0, NAVY], [0.5, "#f7f7f7"], [1, RUST]], zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="хуже (+) / лучше (−)<br>остального<br>периода, п.п.", font=dict(size=10.5)),
                      thickness=12, len=0.9)))
    _layout(fig, height=110 + 40 * len(metrics), letter="B",
            title="Месяц против остального периода (число — доля месяца, %; * — q < 0.05)", legend=False)
    fig.update_yaxes(autorange="reversed", showline=False, ticks="")
    fig.update_xaxes(showline=False, ticks="", side="bottom")
    return fig


# --------------------------------------------------------------------------
# Контрольные карты
# --------------------------------------------------------------------------
def fig_cusum(cusum: dict, reference: dict) -> go.Figure | None:
    avail = {m: c for m, c in cusum.items() if c.get("available")}
    if not avail:
        return None
    n = len(avail)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    total = 0
    for i, (m, c) in enumerate(avail.items(), start=1):
        color = CONTOUR_COLOR[CONTOUR[m]]
        t = c["trace"]
        total += len(c["events"])
        fig.add_trace(go.Scatter(x=t["date"], y=t["statistic"], mode="lines", line_shape="hv",
                                 line=dict(color=color, width=1.6), name=m, showlegend=False,
                                 hovertemplate=f"{m} %{{x|%d.%m.%Y}}: C = %{{y:.2f}}<extra></extra>"),
                      row=i, col=1)
        fig.add_hline(y=c["h_watch"], line=dict(color=GREY, dash="dash", width=1.1), row=i, col=1)
        if not c["events"].empty:
            ev = c["events"]
            fig.add_trace(go.Scatter(x=ev["date"], y=[c["h_watch"]] * len(ev), mode="markers",
                                     marker=dict(symbol="triangle-down", size=11, color=color),
                                     showlegend=False,
                                     customdata=ev["segment_start"].dt.strftime("%d.%m.%Y"),
                                     hovertemplate=f"{m}: watch %{{x|%d.%m.%Y}}<br>накопление с "
                                                   "%{customdata}<extra></extra>"), row=i, col=1)
        ymax = max(float(t["statistic"].max()) * 1.25, c["h_watch"] * 2.2, 1.0)
        fig.update_yaxes(title_text=m, range=[0, ymax], row=i, col=1)
        fig.add_annotation(xref=f"x{'' if i == 1 else i} domain", yref=f"y{'' if i == 1 else i} domain",
                           x=0.005, y=1.0, xanchor="left", yanchor="top", showarrow=False,
                           text=f"h watch {c['h_watch']:.2f} · h alert {c['h_alert']:.1f} (заблокирован)",
                           font=dict(size=10.5, color=INK2))
        if reference.get("declared"):
            fig.add_vrect(x0=reference.get("start") or t["date"].min(), x1=reference.get("end") or t["date"].max(),
                          fillcolor=LGREY, opacity=0.45, line_width=0, row=i, col=1)
    _layout(fig, height=150 + 150 * n, letter="A", legend=False,
            title=f"Бернуллиевский CUSUM на закрытии культивирования: {total} watch, 0 alert")
    fig.update_xaxes(title_text="дата пункции", row=n, col=1)
    all_dates = pd.concat([c["trace"]["date"] for c in avail.values()])
    for i in range(1, n + 1):
        _ru_months(fig, all_dates, row=i, col=1)
    return fig


def fig_ewma(metric: str, e: dict) -> go.Figure | None:
    ew = e["ewma"]
    if ew.empty:
        return None
    color = CONTOUR_COLOR[CONTOUR[metric]]
    s = st.spec(metric)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.concat([ew["date"], ew["date"][::-1]]),
                             y=pd.concat([ew["ucl"], ew["lcl"][::-1]]) * 100, fill="toself",
                             fillcolor=CONTOUR_FILL[CONTOUR[metric]], opacity=0.45, line=dict(width=0),
                             hoverinfo="skip", name="пределы ±3σ (с поправкой на φ)"))
    for col in ("ucl", "lcl"):
        fig.add_trace(go.Scatter(x=ew["date"], y=ew[col] * 100, mode="lines",
                                 line=dict(color=MRUST if CONTOUR[metric] == "culture" else GREY, width=0.7),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=ew["date"], y=ew["ewma"] * 100, mode="lines", line=dict(color=color, width=2),
                             name="EWMA", hovertemplate="%{x|%d.%m.%Y}: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=e["p0"] * 100, line=dict(color=INK2, width=1))
    _thresholds(fig, s, label=False)
    below = int((ew["ewma"] < ew["lcl"]).sum())
    above = int((ew["ewma"] > ew["ucl"]).sum())
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=1.0, xanchor="left", yanchor="top",
                       showarrow=False, font=dict(size=10.5, color=GREY),
                       text=f"пунктир: компетентность · штрих: бенчмарк · вне пределов: {below} ниже, {above} выше")
    _layout(fig, height=420, letter="B", title=f"EWMA {metric} (λ = 0.2) с пределами на сверхдисперсию")
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0, yanchor="top"), margin=dict(b=90))
    fig.update_yaxes(title=f"EWMA {metric}, %", ticksuffix="%")
    fig.update_xaxes(title="")
    _ru_months(fig, ew["date"])
    return fig


def fig_pchart(metric: str, e: dict) -> go.Figure | None:
    p = e["pchart"]
    if p.empty:
        return None
    x = [month_ru(pd.Period(m).to_timestamp()) for m in p["month"]]
    color = CONTOUR_COLOR[CONTOUR[metric]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(p["ucl_laney"].clip(0, 1) * 100) +
                             list(p["lcl_laney"].clip(0, 1)[::-1] * 100), mode="lines", fill="toself", fillcolor=LNAVY,
                             opacity=0.6, line=dict(width=0, shape="hvh"), name="пределы Laney p′",
                             hoverinfo="skip"))
    for col, show in (("ucl_binom", True), ("lcl_binom", False)):
        fig.add_trace(go.Scatter(x=x, y=p[col].clip(0, 1) * 100, mode="lines", line_shape="hvh",
                                 line=dict(color=GREY, dash="dash", width=1.1), name="биномиальные 3σ",
                                 showlegend=show, hoverinfo="skip"))
    out = p["out_binom"].to_numpy()
    fig.add_trace(go.Scatter(x=x, y=p["rate"] * 100, mode="lines+markers", line=dict(color=color, width=1.6),
                             marker=dict(size=[12 if o else 7 for o in out],
                                         color=[RUST if o else color for o in out]),
                             name="доля месяца",
                             customdata=np.column_stack([p["x"], p["n"], p["out_laney"]]),
                             hovertemplate="%{x}: %{y:.1f}% (%{customdata[0]:.0f}/%{customdata[1]:.0f})"
                                           "<extra></extra>"))
    fig.add_hline(y=e["p0"] * 100, line=dict(color=INK2, width=1))
    info = e["pchart_info"]
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=1.0, xanchor="right", yanchor="top", showarrow=False,
                       font=dict(size=10.5, color=RUST),
                       text=f"вне пределов: {info['outside_binomial']}/{info['months']} биномиальных, "
                            f"{info['outside_laney']}/{info['months']} Laney")
    _layout(fig, height=420, letter="C", title=f"p-карта Шухарта {metric} по месяцам")
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0, yanchor="top"), margin=dict(b=90))
    fig.update_xaxes(tickangle=0, tickvals=x, ticktext=[t.replace(" ", "<br>") for t in x])
    fig.update_yaxes(title=f"{metric}, %", ticksuffix="%", rangemode="tozero")
    return fig


# --------------------------------------------------------------------------
# Эмбриологи
# --------------------------------------------------------------------------
def fig_emb_violin(metric: str, dist: pd.DataFrame, test: dict, colors: dict) -> go.Figure | None:
    d = dist.dropna(subset=["embryologist"]) if "embryologist" in dist else pd.DataFrame()
    if d.empty:
        return None
    s = st.spec(metric)
    groups = sorted(d["embryologist"].unique(), key=str)
    per = {g["group"]: g for g in test.get("per_group", [])}
    fig = go.Figure()
    for grp in groups:
        v = d[d["embryologist"] == grp]
        fig.add_trace(go.Violin(
            x=[grp] * len(v), y=v["rate"] * 100, name=grp, line=dict(color=colors[grp], width=1.2),
            fillcolor=_rgba(colors[grp], 0.22), points="all", pointpos=0, jitter=0.35,
            marker=dict(size=4.5, color=colors[grp], opacity=0.75), box=dict(visible=False),
            spanmode="hard", scalemode="width", width=0.8, showlegend=False,
            hovertemplate=f"{grp}: %{{y:.1f}}%<extra>{len(v)} цикл.</extra>"))
        if grp in per:
            g = per[grp]
            fig.add_trace(go.Scatter(
                x=[grp], y=[g["rate"] * 100], mode="markers", showlegend=False,
                marker=dict(symbol="diamond", size=13, color=INK, line=dict(color="white", width=1)),
                error_y=dict(type="data", array=[(g["hi"] - g["rate"]) * 100],
                             arrayminus=[(g["rate"] - g["lo"]) * 100], color=INK, thickness=1.6, width=6),
                hovertemplate=f"{grp}: Σ/Σ = {pct(g['rate'])} [{pct(g['lo'])}–{pct(g['hi'])}]<br>"
                              f"{g['x']:.0f}/{g['n']:.0f}, {g['cycles']} цикл.<extra></extra>"))
    _thresholds(fig, s)
    _layout(fig, height=440, letter="A", title=f"{st.METRIC_RU[metric]} по эмбриологам", legend=False)
    fig.update_yaxes(title=f"{metric} в цикле, % (ромб — Σ/Σ с 95% ДИ)", ticksuffix="%", rangemode="tozero")
    return fig


def fig_funnel(metric: str, f: dict, colors: dict) -> go.Figure | None:
    if not f:
        return None
    s = st.spec(metric)
    fig = go.Figure()
    for lo, hi, name, dash in (("l998", "u998", "99.8%", "dash"), ("l95", "u95", "95%", "dot")):
        for key, show in ((lo, True), (hi, False)):
            fig.add_trace(go.Scatter(x=f["n"], y=np.clip(f[key], 0, 1.2) * 100, mode="lines",
                                     line=dict(color=GREY, dash=dash, width=1.2), name=f"предел {name}",
                                     showlegend=show, hoverinfo="skip"))
    fig.add_hline(y=f["p"] * 100, line=dict(color=INK2, width=1))
    g = f["groups"]
    fig.add_trace(go.Scatter(x=g["n"], y=g["rate"] * 100, mode="markers+text",
                             text=[str(v).split()[-1] for v in g.iloc[:, 0]], textposition="top center",
                             textfont=dict(size=11),
                             marker=dict(size=13, color=[colors.get(v, NAVY) for v in g.iloc[:, 0]],
                                         line=dict(color="white", width=1)),
                             customdata=np.column_stack([g.iloc[:, 0], g["cycles"]]),
                             hovertemplate="%{customdata[0]}: %{y:.1f}%<br>знаменатель %{x:.0f}, "
                                           "%{customdata[1]} цикл.<extra></extra>", showlegend=False))
    _layout(fig, height=440, letter="B", title=f"Funnel plot {metric}: доля против объёма")
    fig.update_layout(legend=dict(orientation="h", y=-0.16, x=0, yanchor="top"))
    fig.update_yaxes(title=f"{metric}, %", ticksuffix="%")
    fig.update_xaxes(title=f"знаменатель ({st.METRIC_SHORT[metric].split('/')[-1].strip()})")
    return fig


# --------------------------------------------------------------------------
# Оплодотворение
# --------------------------------------------------------------------------
def fig_fr_violin(cycles: pd.DataFrame, fm: pd.DataFrame) -> go.Figure | None:
    d = cycles[cycles["n"] >= st.MIN_DEN_DISPLAY]
    if d.empty:
        return None
    order = [m for m in ("ICSI", "IVF", "IVF+ICSI") if m in set(d["method"])]
    fig = go.Figure()
    fmi = fm.set_index("method")
    for i, m in enumerate(order):
        v = d[d["method"] == m]
        fig.add_trace(go.Violin(x=[m] * len(v), y=v["rate"] * 100, name=m, line=dict(color=METHOD_COLOR[m]),
                                fillcolor=_rgba(METHOD_COLOR[m], 0.22), points="all", pointpos=0, jitter=0.3,
                                marker=dict(size=4.5, color=METHOD_COLOR[m], opacity=0.75), spanmode="hard",
                                box=dict(visible=False), showlegend=False, width=0.8,
                                hovertemplate=f"{m}: %{{y:.1f}}%<extra>{len(v)} цикл.</extra>"))
        if m in fmi.index:
            r = fmi.loc[m]
            fig.add_trace(go.Scatter(x=[m], y=[r["rate"] * 100], mode="markers", showlegend=False,
                                     marker=dict(symbol="diamond", size=13, color=INK),
                                     error_y=dict(type="data", array=[(r["ci_high_phi"] - r["rate"]) * 100],
                                                  arrayminus=[(r["rate"] - r["ci_low_phi"]) * 100],
                                                  color=INK, thickness=1.6, width=6),
                                     hovertemplate=f"{m}: Σ/Σ = {pct(r['rate'])}<extra></extra>"))
            for value, dash in ((r["competency"], "dot"), (r["benchmark"], "dash")):
                if value is not None and np.isfinite(value):
                    fig.add_shape(type="line", x0=i - 0.42, x1=i + 0.42, y0=value * 100, y1=value * 100,
                                  line=dict(color=INK2, dash=dash, width=1.4))
    _layout(fig, height=440, letter="A", title="Оплодотворение по методу", legend=False)
    fig.update_yaxes(title="2PN / знаменатель метода, % (ромб — Σ/Σ)", ticksuffix="%", rangemode="tozero")
    return fig


def fig_fr_ecdf(cycles: pd.DataFrame, fm: pd.DataFrame) -> go.Figure | None:
    d = cycles[cycles["n"] >= st.MIN_DEN_DISPLAY]
    if d.empty:
        return None
    fig = go.Figure()
    fmi = fm.set_index("method")
    for m in ("ICSI", "IVF", "IVF+ICSI"):
        v = np.sort(d.loc[d["method"] == m, "rate"].to_numpy() * 100)
        if len(v) == 0:
            continue
        y = np.arange(1, len(v) + 1) / len(v)
        fig.add_trace(go.Scatter(x=np.r_[0, v], y=np.r_[0, y], mode="lines", line_shape="hv",
                                 line=dict(color=METHOD_COLOR[m], width=2.2), name=f"{m} (n = {len(v)})",
                                 hovertemplate=f"{m}: %{{y:.0%}} циклов с FR ≤ %{{x:.1f}}%<extra></extra>"))
        if m in fmi.index and np.isfinite(fmi.loc[m, "competency"] or np.nan):
            fig.add_vline(x=fmi.loc[m, "competency"] * 100,
                          line=dict(color=METHOD_COLOR[m], dash="dot", width=1.2))
    _layout(fig, height=440, letter="B", title="Эмпирическая функция распределения FR")
    fig.update_layout(legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"))
    fig.update_xaxes(title="FR цикла, % (пунктир — компетентность метода)", ticksuffix="%", range=[0, 101])
    fig.update_yaxes(title="доля циклов с FR ≤ x", tickformat=".0%", range=[0, 1.02])
    return fig


# --------------------------------------------------------------------------
# TGBDR
# --------------------------------------------------------------------------
def fig_corr(corr: pd.DataFrame) -> go.Figure | None:
    if corr.empty:
        return None
    order = (corr[corr["target"] == "bl_good"].sort_values("rho", key=abs)["feature"].tolist())
    order += [f for f in corr["feature"].unique() if f not in order]
    titles = {"bl_good": "Число Bl хорошего качества на цикл", "TGBDR": f"TGBDR цикла (2PN ≥ {st.MIN_DEN_DISPLAY})"}
    targets = [t for t in ("bl_good", "TGBDR") if t in set(corr["target"])]
    fig = make_subplots(rows=1, cols=len(targets), shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=[titles[t] for t in targets])
    coupled_any = set(corr.loc[corr["coupled"].astype(bool), "feature"])
    labels = {r.feature: r.feature_ru + (" †" if r.feature in coupled_any else "") for r in corr.itertuples()}
    for j, t in enumerate(targets, start=1):
        c = corr[corr["target"] == t].set_index("feature").reindex(order).dropna(subset=["rho"])
        sig = c["q"] < 0.05
        ylab = [labels[f] for f in c.index]
        fig.add_trace(go.Scatter(
            x=c["rho"], y=ylab, mode="markers",
            marker=dict(size=11, color=[RUST if (s and r < 0) else NAVY if s else "white"
                                        for s, r in zip(sig, c["rho"])],
                        line=dict(color=[RUST if r < 0 else NAVY for r in c["rho"]], width=1.6)),
            error_x=dict(type="data", array=c["hi"] - c["rho"], arrayminus=c["rho"] - c["lo"],
                         color=INK2, thickness=1.2, width=0),
            customdata=np.column_stack([c["n"], c["q"], c["lo"], c["hi"]]),
            hovertemplate="%{y}: ρ = %{x:+.2f} [%{customdata[2]:+.2f}; %{customdata[3]:+.2f}]"
                          "<br>n = %{customdata[0]:.0f}, q(BH) = %{customdata[1]:.3f}<extra></extra>",
            showlegend=False), row=1, col=j)
        fig.add_vline(x=0, line=dict(color=INK2, width=1), row=1, col=j)
        fig.update_xaxes(range=[-1, 1], title_text="ρ Спирмена (95% ДИ)", row=1, col=j)
    _layout(fig, height=120 + 34 * len(order), letter="A", legend=False,
            title="Связь признаков цикла с выходом бластоцист хорошего качества")
    fig.update_annotations(font=dict(size=12, color=INK2))
    fig.update_layout(margin=dict(l=170, r=24, t=80, b=56))
    return fig


def fig_age(age: pd.DataFrame) -> go.Figure | None:
    if age.empty:
        return None
    s = st.spec("TGBDR")
    fig = go.Figure(go.Bar(x=age["age_group"], y=age["rate"] * 100, marker=dict(color=LRUST, line=dict(color=RUST, width=1.2)),
                           error_y=dict(type="data", array=(age["hi"] - age["rate"]) * 100,
                                        arrayminus=(age["rate"] - age["lo"]) * 100, color=INK2, thickness=1.3),
                           text=[f"{c} цикл." for c in age["cycles"]], textposition="inside",
                           insidetextanchor="start", textfont=dict(size=10.5, color=INK2),
                           customdata=np.column_stack([age["x"], age["n"]]),
                           hovertemplate="%{x}: %{y:.1f}% (%{customdata[0]:.0f}/%{customdata[1]:.0f})<extra></extra>"))
    _thresholds(fig, s)
    _layout(fig, height=400, letter="B", title="TGBDR по возрасту пациентки (Σ/Σ, 95% ДИ)", legend=False)
    fig.update_yaxes(title="TGBDR, %", ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(title="возраст, лет")
    return fig


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
CSS = """
:root{--navy:#2c4a5e;--rust:#c0492c;--purple:#7b4fa8;--aqua:#2e7d8a;--yellow:#b8860b;
--ink:#222;--ink2:#555;--muted:#8c8c8c;--line:#e3e3e3;--bg:#f6f7f8;--card:#fff;color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 'DejaVu Sans','Segoe UI',Helvetica,Arial,sans-serif}
header{background:#fff;border-bottom:1px solid var(--line)}
.wrap{max-width:1240px;margin:0 auto;padding:0 20px}
.head{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:flex-end;justify-content:space-between;padding:22px 0 14px}
h1{font-size:23px;margin:0;color:var(--navy);letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:13.5px;margin-top:4px}
.stamp{font-size:12px;font-weight:600;color:#8a6314;background:#fdf3e3;border:1px solid #f0dcb4;
border-radius:6px;padding:5px 10px;white-space:nowrap}
nav{display:flex;gap:2px;overflow-x:auto;scrollbar-width:thin}
nav button{border:0;background:none;padding:11px 15px;font:inherit;font-size:14px;color:var(--ink2);
cursor:pointer;border-bottom:3px solid transparent;white-space:nowrap}
nav button:hover{color:var(--ink)}
nav button.on{color:var(--navy);border-bottom-color:var(--navy);font-weight:600}
main{padding:22px 0 60px}
.tab{display:none}.tab.on{display:block}
.lead{font-size:14.5px;color:var(--ink2);max-width:920px;margin:0 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:16px;min-width:0}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:980px){.row{grid-template-columns:1fr}}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
.tile{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{font-size:12.5px;color:var(--ink2)}.tile .v{font-size:26px;font-weight:600;color:var(--navy);line-height:1.2}
.tile .f{font-size:12px;color:var(--muted)}
.facts{border-left:3px solid var(--navy);background:#f3f6f8;padding:10px 16px;border-radius:0 8px 8px 0;margin:0 0 16px}
.checks{border-left:3px solid var(--rust);background:#fbf1ee;padding:10px 16px;border-radius:0 8px 8px 0;margin:0 0 16px}
.ai{border-left:3px solid var(--purple);background:#f5f1f9;padding:10px 16px;border-radius:0 8px 8px 0;margin:0 0 16px}
.facts h3,.checks h3,.ai h3{margin:4px 0 6px;font-size:14px;letter-spacing:.02em;text-transform:uppercase;color:var(--ink2)}
.facts ul,.checks ul,.ai ul{margin:4px 0 6px;padding-left:20px}.facts li,.checks li,.ai li{margin:3px 0;font-size:14px}
.tag{font-size:11px;font-weight:600;padding:1px 7px;border-radius:9px;margin-left:6px;vertical-align:1px}
.tag.f{background:#e6edf1;color:var(--navy)}.tag.h{background:#f6e3de;color:var(--rust)}.tag.a{background:#ece4f4;color:var(--purple)}
.ctrl{display:flex;align-items:center;gap:10px;margin:0 0 12px;font-size:14px;color:var(--ink2)}
select{font:inherit;padding:5px 10px;border:1px solid #cfd4d8;border-radius:6px;background:#fff}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--ink2);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{overflow-x:auto}
.st-benchmark{color:var(--aqua);font-weight:600}.st-competency{color:var(--yellow);font-weight:600}
.st-outside_control{color:var(--rust);font-weight:600}
.test{font-size:14px;margin:6px 0 2px}.test b{color:var(--navy)}
.muted{color:var(--muted)}.small{font-size:12.5px}
.empty{padding:28px;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:8px}
details{margin-top:8px}summary{cursor:pointer;color:var(--ink2);font-size:13.5px}
footer{color:var(--muted);font-size:12px;padding:10px 0 30px}
"""

JS = """
function showTab(id){document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.id===id));
document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.tab===id));
resizeVisible();try{history.replaceState(null,'','#'+id)}catch(e){}}
function resizeVisible(){setTimeout(()=>{document.querySelectorAll('.tab.on .js-plotly-plot').forEach(p=>{
if(p.offsetParent!==null){Plotly.Plots.resize(p)}})},30)}
function pick(group,key){document.querySelectorAll('div[data-group="'+group+'"]').forEach(el=>{
el.style.display=(el.dataset.key===key)?'':'none'});resizeVisible()}
document.addEventListener('DOMContentLoaded',()=>{document.querySelectorAll('nav button').forEach(b=>
b.addEventListener('click',()=>showTab(b.dataset.tab)));
document.querySelectorAll('select[data-group]').forEach(s=>{s.addEventListener('change',()=>pick(s.dataset.group,s.value));
pick(s.dataset.group,s.value)});
const h=location.hash.slice(1);showTab(document.getElementById(h)?h:'t-summary')});
window.addEventListener('resize',resizeVisible);
"""


def _facts_html(items: list[str], title: str = "Наблюдения") -> str:
    if not items:
        return ""
    head, tail = items[:7], items[7:]
    lis = "".join(f"<li>{escape(i)}</li>" for i in head)
    more = ""
    if tail:
        more = ("<details><summary>Ещё наблюдения и оговорки о данных (" + str(len(tail)) + ")</summary><ul>"
                + "".join(f"<li>{escape(i)}</li>" for i in tail) + "</ul></details>")
    return f'<div class="facts"><h3>{title}<span class="tag f">факт · посчитано кодом</span></h3><ul>{lis}</ul>{more}</div>'


def _checks_html(checks: list[dict]) -> str:
    if not checks:
        return ('<div class="checks"><h3>Что проверить<span class="tag h">гипотезы</span></h3>'
                '<ul><li>Поводов для разбора по правилам проекта нет: все KPI не ниже порога компетентности '
                'и подтверждённых эпизодов CUSUM нет.</li></ul></div>')
    lis = []
    for c in checks:
        items = "; ".join(escape(i) for i in c["items"]) or "перечень консенсуса не задан"
        lis.append(f"<li><b>{c['metric']}</b> ({escape(CONTOUR_RU.get(c['contour'], c['contour']))}; "
                   f"повод: {escape(c['why'])}) — по Венскому консенсусу проверить: {items}.</li>")
    return ('<div class="checks"><h3>Что проверить<span class="tag h">гипотезы, не выводы</span></h3>'
            f'<ul>{"".join(lis)}</ul><div class="small muted">Перечни — из config/consensus/vienna.yaml. '
            'Это поводы для разбора, а не диагноз и не оценка специалиста. В выгрузке нет логов '
            'инкубатора и лотов среды, поэтому независимого подтверждения здесь быть не может.</div></div>')


def _md(text: str) -> str:
    """Минимальный Markdown -> HTML для комментария ИИ (заголовки, списки, жирный)."""
    out, in_list = [], False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            if in_list:
                out.append("</ul>"); in_list = False
            continue
        s_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(s))
        if s.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{s_html[2:]}</li>")
        elif s.startswith("#"):
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<h4>{s_html.lstrip('#').strip()}</h4>")
        else:
            if in_list:
                out.append("</ul>"); in_list = False
            out.append(f"<p>{s_html}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def _test_html(t: dict, label: str) -> str:
    if not t or not t.get("test"):
        why = (t or {}).get("why", "недостаточно данных")
        return f'<div class="test muted">{escape(label)}: тест не выполнялся — {escape(why)}.</div>'
    rob = t.get("robust", {})
    rob_txt = ""
    if rob.get("test"):
        rob_txt = (f"<br><span class='small'>Проверка на уровне циклов: {rob['test']}, {pval(rob['p'])}, "
                   f"{rob['effect_name']} = {rob['effect']:.2f}.</span>")
    small = t.get("excluded_small") or []
    small_txt = (f"<br><span class='small muted'>Не участвуют в тесте (&lt; {st.MIN_CYCLES_TEST} циклов): "
                 f"{escape(', '.join(map(str, small)))}.</span>") if small else ""
    verdict = ("различие статистически значимо" if t["p"] < 0.05 else
               "статистически значимого различия не выявлено (это не доказательство равенства)")
    pw = [p for p in t.get("pairwise", []) if p["p_holm"] < 0.05]
    pw_txt = ""
    if pw:
        pw_txt = "<br><span class='small'>Значимые пары (Холм): " + "; ".join(
            f"{escape(str(p['a']))} − {escape(str(p['b']))}: {p['diff_pp']:+.1f} п.п. "
            f"[{p['lo_pp']:+.1f}; {p['hi_pp']:+.1f}], p = {p['p_holm']:.3f}" for p in pw) + "</span>"
    return (f'<div class="test">{escape(label)}: <b>{escape(t["test"])}</b>, {pval(t["p"])} — {verdict}.'
            f'<br><span class="small muted">Почему этот тест: {escape(t["why"])}.</span>{rob_txt}{pw_txt}{small_txt}</div>')


def _kpi_table(summary: pd.DataFrame) -> str:
    rows = []
    for r in summary.itertuples():
        if not r.cycles:
            rows.append(f"<tr><td>{r.label_ru}</td><td colspan='7' class='muted'>нет нужных колонок в выгрузке</td></tr>")
            continue
        rows.append(
            f"<tr><td><b>{r.metric}</b> · {escape(r.label_ru.split('(')[0])}<br><span class='small muted'>{escape(r.definition)}</span></td>"
            f"<td class='n'>{r.numerator:,.0f} / {r.denominator:,.0f}</td><td class='n'>{r.cycles}</td>"
            f"<td class='n'><b>{pct(r.rate)}</b></td><td class='n'>{pct(r.ci_low_phi)}–{pct(r.ci_high_phi)}</td>"
            f"<td class='n'>{pct(r.competency, 1)} / {pct(r.benchmark, 1)}</td>"
            f"<td class='st-{r.status}'>{st.STATUS_RU[r.status]}</td>"
            f"<td class='small muted'>{escape(r.indicator_class)}: {escape(r.source_note)}</td></tr>".replace(",", " "))
    return ("<div class='scroll'><table><tr><th>Показатель</th><th>Σ числ. / Σ знам.</th><th>Циклов</th><th>Доля</th>"
            "<th>95% ДИ (φ)</th><th>Компет. / бенчм.</th><th>Статус</th><th>Класс и источник порога</th></tr>"
            + "".join(rows) + "</table></div>")


def render(res: dict, *, commentary: str | None = None, plotly_js: str = "inline",
           show_operator_names: bool = False) -> str:
    summary: pd.DataFrame = res["kpi"]
    window = res["settings_used"]["rolling_window_cycles"]
    stamp = f"ДЕМО · environment=draft · {res['generated_at']} · alert заблокирован"
    period = res.get("period", {})
    ok = summary[summary["cycles"] > 0]
    n_bench = int((ok["status"] == "benchmark").sum())
    n_comp = int((ok["status"] == "competency").sum())
    n_out = int((ok["status"] == "outside_control").sum())
    n_watch = sum(len(c["events"]) for c in res["cusum"].values() if c.get("available"))
    n_emb = res.get("fr_cycles", pd.DataFrame())
    embryologists = sorted({g for d in res["distributions"].values() if "embryologist" in d
                            for g in d["embryologist"].dropna().unique()}, key=str)
    emb_colors = {g: CATEGORICAL[i % len(CATEGORICAL)] for i, g in enumerate(embryologists)}

    tabs: list[tuple[str, str, str]] = []

    # --- Сводка -------------------------------------------------------------
    tiles = (f"<div class='tiles'>"
             f"<div class='tile'><div class='k'>Циклов в анализе</div><div class='v'>{res['n_cycles']}</div>"
             f"<div class='f'>из {res['exclusions']['rows_total']} строк выгрузки</div></div>"
             f"<div class='tile'><div class='k'>Период пункций</div><div class='v' style='font-size:19px'>"
             f"{period.get('start', 'н/д')}<br>— {period.get('end', 'н/д')}</div></div>"
             f"<div class='tile'><div class='k'>Зигот 2PN</div><div class='v'>{res['n_2pn']:,.0f}</div>"
             f"<div class='f'>единица карт по долям — эмбрион</div></div>"
             f"<div class='tile'><div class='k'>KPI: бенчмарк · компет. · ниже</div><div class='v'>"
             f"<span style='color:#2e7d8a'>{n_bench}</span> · <span style='color:#b8860b'>{n_comp}</span> · "
             f"<span style='color:#c0492c'>{n_out}</span></div><div class='f'>из {len(ok)} показателей</div></div>"
             f"<div class='tile'><div class='k'>Эпизоды CUSUM</div><div class='v'>{n_watch} watch</div>"
             f"<div class='f'>0 alert — демо не поднимает alert</div></div></div>").replace(",", " ")
    ai_block = ""
    if commentary:
        ai_block = (f"<div class='ai'><h3>Комментарий ИИ<span class='tag a'>интерпретация по пакету агрегатов</span></h3>"
                    f"{_md(commentary)}</div>")
    body = (f"<p class='lead'>Сводные показатели лаборатории, посчитанные по правилам EWS: доля — Σ числитель / "
            f"Σ знаменатель по всем циклам, пороги — Венский консенсус. Серая полоса — зона между "
            f"компетентностью (пунктир) и бенчмарком (штрих).</p>{tiles}{_facts_html(res['facts'])}"
            f"{_checks_html(res['checks'])}{ai_block}"
            f"<div class='card'>{_div(fig_kpi_bullet(summary), 'f-bullet')}</div>"
            f"<div class='card'>{_div(fig_violin_overall(res['distributions'], summary), 'f-violin-all')}"
            f"<div class='small muted'>Скрипка — разброс долей между циклами; ромб — доля лаборатории (Σ/Σ). "
            f"Ромб не обязан совпадать с медианой: крупные циклы весят больше. Цвет — контур "
            f"(врач / эмбриолог / культивирование).</div></div>"
            f"<div class='card'><details><summary>Таблица KPI с источниками порогов</summary>{_kpi_table(summary)}</details></div>")
    tabs.append(("t-summary", "Сводка", body))

    # --- Динамика ---------------------------------------------------------
    series = res["series"]
    if series:
        options = "".join(f"<option value='{m}'>{st.METRIC_RU[m]}</option>" for m in series)
        default = "BlR" if "BlR" in series else next(iter(series))
        options = options.replace(f"value='{default}'", f"value='{default}' selected")
        variants = []
        for m, e in series.items():
            fig = fig_dynamics(m, e, res["cusum"].get(m), window)
            variants.append(f"<div data-group='dyn' data-key='{m}'><div class='card'>{_div(fig, f'f-dyn-{m}')}"
                            f"{_test_html(e['month_test'], 'Различие между месяцами')}</div></div>")
        heat = fig_heat(res["heat"])
        body = (f"<p class='lead'>Как менялись показатели во времени. Линия — доля последних {window} циклов "
                f"(Σ/Σ) с 95% ДИ, расширенным на межцикловый разброс; кружки — месяцы. Треугольники внизу — "
                f"сигналы карты CUSUM.</p><div class='ctrl'>Показатель: <select data-group='dyn'>{options}</select></div>"
                + "".join(variants)
                + (f"<div class='card'>{_div(heat, 'f-heat')}<div class='small muted'>Каждый месяц сравнивается "
                   f"с остальными месяцами того же показателя (z-тест долей с поправкой на φ или точный тест "
                   f"Фишера при малых частотах; поправка Бенджамини–Хохберга внутри показателя). Цвет учитывает "
                   f"направление: для DegR и 3pNR рост — ухудшение.</div></div>" if heat else ""))
    else:
        body = _empty("Нет колонки даты пункции — временной анализ недоступен.")
    tabs.append(("t-dyn", "Динамика", body))

    # --- Контрольные карты ------------------------------------------------
    cus = fig_cusum(res["cusum"], res["reference"])
    if series:
        options = "".join(f"<option value='{m}'>{st.METRIC_RU[m]}</option>" for m in series)
        default = "BlR" if "BlR" in series else next(iter(series))
        options = options.replace(f"value='{default}'", f"value='{default}' selected")
        variants = []
        for m, e in series.items():
            fe, fp = fig_ewma(m, e), fig_pchart(m, e)
            variants.append(f"<div class='row' data-group='spc' data-key='{m}'>"
                            f"<div class='card'>{_div(fe, f'f-ewma-{m}') if fe else _empty('нет данных')}</div>"
                            f"<div class='card'>{_div(fp, f'f-p-{m}') if fp else _empty('нет данных')}</div></div>")
        ep = res["episodes"]
        ep_html = ""
        if not ep.empty:
            def _ep_rows(frame):
                return "".join(
                f"<tr><td><b>{r.metric}</b></td><td>{pd.Timestamp(r.segment_start):%d.%m.%Y} — "
                f"{pd.Timestamp(r.signal_date):%d.%m.%Y}</td><td class='n'>{r.segment_cycles}</td>"
                f"<td class='n'>{pct(r.rate_a)}</td><td class='n'>{pct(r.rate_b)}</td>"
                f"<td class='n'>{r.diff_pp:+.1f}"
                + (f" [{r.lo_pp:+.1f}; {r.hi_pp:+.1f}]" if np.isfinite(r.lo_pp) else "")
                + f"</td><td class='n'>{pval(r.p) if np.isfinite(r.p) else '—'}</td>"
                f"<td class='small'>{escape(r.baseline)}; {escape(r.test)}"
                f"{'; порог alert пересечён — понижено до watch' if r.alert_threshold_crossed else ''}</td></tr>"
                for r in frame.itertuples())
            head = ("<tr><th>Карта</th><th>Накопление</th><th>Циклов</th><th>Доля в эпизоде</th><th>База</th>"
                    "<th>Разность, п.п. [95% ДИ]</th><th>p</th><th>Сравнение</th></tr>")
            long_ep = ep[ep["segment_cycles"] >= 3]
            short_ep = ep[ep["segment_cycles"] < 3]
            trs = (f"<table>{head}{_ep_rows(long_ep)}</table>" if not long_ep.empty else
                   "<p class='small muted'>Эпизодов, накопившихся за 3 и более цикла, нет.</p>")
            if not short_ep.empty:
                trs += (f"<details><summary>Эпизоды из 1–2 циклов ({len(short_ep)}) — отдельные пациентки, "
                        f"тест не выполнялся</summary><table>{head}{_ep_rows(short_ep)}</table></details>")
            ep_html = ("<div class='card'><h3 style='margin:4px 0 8px;font-size:15px'>Эпизоды watch: период накопления "
                       f"против базы</h3><div class='scroll'>{trs}</div><div class='small muted'>Начало эпизода — последний ноль статистики "
                       "перед сигналом (оценка точки сдвига). Сравнение ретроспективное и справочное: эпизод "
                       "выделен тем же рядом, поэтому p здесь оптимистичен.</div></div>")
        ref = res["reference"]
        ref_txt = (f"опорное окно {ref.get('start') or 'начало'} — {ref.get('end') or 'конец'} (объявлено при запуске)"
                   if ref.get("declared") else "центр карт — доля за весь период (ретроспективная фаза I)")
        body = (f"<p class='lead'>Статистический контроль процесса, как в публикации (Figure 7). CUSUM копит "
                f"свидетельство ухудшения цикл за циклом и подаёт сигнал watch при пересечении порога. Пороги "
                f"подобраны симуляцией под цель ложных сигналов проекта ({res['settings_used']['targets_per_year']['watch']} "
                f"watch и {res['settings_used']['targets_per_year']['alert']} alert в год на систему); {ref_txt}.</p>"
                + (f"<div class='card'>{_div(cus, 'f-cusum')}</div>" if cus else _empty("CUSUM не построен: мало циклов с датой."))
                + ep_html
                + f"<div class='ctrl'>Показатель для EWMA и p-карты: <select data-group='spc'>{options}</select></div>"
                + "".join(variants)
                + "<div class='small muted'>CUSUM и EWMA одной метрики — одно и то же измерение, а не два "
                  "независимых подтверждения. Пределы Laney p′ учитывают межмесячный разброс сверх биномиального: "
                  "точки вне биномиальных, но внутри Laney-пределов, обычно отражают нормальную неоднородность "
                  "пациенток, а не сдвиг процесса.</div>")
    else:
        body = _empty("Нет колонки даты пункции — контрольные карты недоступны.")
    tabs.append(("t-spc", "Контрольные карты", body))

    # --- Эмбриологи -------------------------------------------------------
    emb = res["embryologists"]
    emb_metrics = [m for m in kpi.EMBRYOLOGIST_METRICS if m in emb]
    if emb_metrics:
        options = "".join(f"<option value='{m}'>{st.METRIC_RU[m]}</option>" for m in emb_metrics)
        default = "FR" if "FR" in emb_metrics else emb_metrics[0]
        options = options.replace(f"value='{default}'", f"value='{default}' selected")
        variants = []
        for m in emb_metrics:
            fv = fig_emb_violin(m, res["distributions"][m], emb[m]["test"], emb_colors)
            ff = fig_funnel(m, emb[m]["funnel"], emb_colors)
            variants.append(f"<div data-group='emb' data-key='{m}'><div class='row'>"
                            f"<div class='card'>{_div(fv, f'f-embv-{m}') if fv else _empty('нет данных')}</div>"
                            f"<div class='card'>{_div(ff, f'f-funnel-{m}') if ff else _empty('нет данных')}</div></div>"
                            f"<div class='card'>{_test_html(emb[m]['test'], 'Различие между эмбриологами')}</div></div>")
        casemix = []
        if "_casemix_age" in emb:
            c = emb["_casemix_age"]
            casemix.append(f"возраст пациенток: {c['test']}, {pval(c['p'])} (медианы: "
                           + ", ".join(f"{k} — {v:.0f}" for k, v in c["medians"].items()) + ")")
        if "_casemix_method" in emb:
            casemix.append(f"доля методов оплодотворения: {emb['_casemix_method']['test']}, "
                           f"{pval(emb['_casemix_method']['p'])}")
        names_note = ("Имена показаны как в выгрузке (запуск с --show-operator-names)." if show_operator_names else
                      "Имена заменены псевдонимами; таблица соответствия — только локально, в data/local_private.")
        body = (f"<p class='lead'>Показатели контура эмбриолога и культивирования по специалисту, проводившему "
                f"оплодотворение. Точки — отдельные циклы, ромб — доля специалиста (Σ/Σ) с 95% ДИ. Funnel plot "
                f"показывает, выходит ли доля за пределы, ожидаемые при таком объёме работы. Врачебные OCCR/MIIR "
                f"здесь не показываются: таблицы эмбриолога и врача по правилам проекта не смешиваются.</p>"
                f"<div class='ctrl'>Показатель: <select data-group='emb'>{options}</select></div>" + "".join(variants)
                + (f"<div class='checks'><h3>Состав пациенток<span class='tag h'>оговорка</span></h3><ul>"
                   + "".join(f"<li>{escape(c)}</li>" for c in casemix)
                   + "</ul><div class='small muted'>Если состав различается, разница KPI между специалистами может "
                     "отражать пациенток, а не технику. Это наблюдение о процессе, не оценка специалиста.</div></div>"
                   if casemix else "")
                + f"<div class='small muted'>{names_note}</div>")
    else:
        body = _empty("В выгрузке нет колонки эмбриолога — сравнение специалистов недоступно.")
    tabs.append(("t-emb", "Эмбриологи", body))

    # --- Оплодотворение -----------------------------------------------------
    fm, fc = res.get("fr_method", pd.DataFrame()), res.get("fr_cycles", pd.DataFrame())
    if isinstance(fm, pd.DataFrame) and not fm.empty:
        fv, fe = fig_fr_violin(fc, fm), fig_fr_ecdf(fc, fm)
        trs = "".join(
            f"<tr><td><b>{r.method}</b></td><td class='n'>{r.cycles}</td><td class='n'>{r.numerator:.0f} / {r.denominator:.0f}</td>"
            f"<td class='n'><b>{pct(r.rate)}</b></td><td class='n'>{pct(r.ci_low_phi)}–{pct(r.ci_high_phi)}</td>"
            f"<td class='n'>{pct(r.competency, 0)} / {pct(r.benchmark, 0)}</td><td class='st-{r.status}'>{st.STATUS_RU[r.status]}</td>"
            f"<td class='small muted'>{escape(r.source_note or '—')}</td></tr>" for r in fm.itertuples())
        ks = res.get("fr_ks")
        ks_txt = (f"<div class='test'>Формы распределений ICSI и IVF: <b>тест Колмогорова–Смирнова</b>, D = {ks['D']:.2f}, "
                  f"{pval(ks['p'])} <span class='small muted'>(сравнивает распределения долей по циклам целиком, "
                  f"не только среднее; при повторяющихся значениях p приблизителен)</span></div>") if ks else ""
        body = (f"<p class='lead'>Нормальное оплодотворение раздельно по методу — с порогами Венского консенсуса "
                f"для каждого метода и знаменателем, который консенсус задаёт для метода: ICSI — инъецированные MII, "
                f"IVF — инсеминированные ОКК. Справа — эмпирическая функция распределения: чем правее кривая, тем "
                f"больше циклов с высоким FR.</p><div class='row'>"
                f"<div class='card'>{_div(fv, 'f-frv') if fv else _empty('нет данных')}</div>"
                f"<div class='card'>{_div(fe, 'f-ecdf') if fe else _empty('нет данных')}</div></div>"
                f"<div class='card'><div class='scroll'><table><tr><th>Метод</th><th>Циклов</th><th>2PN / знам.</th>"
                f"<th>FR</th><th>95% ДИ (φ)</th><th>Компет. / бенчм.</th><th>Статус</th><th>Источник</th></tr>{trs}</table></div>"
                f"{_test_html(res.get('fr_method_test', {}), 'Различие FR между методами')}{ks_txt}"
                f"<div class='small muted'>Метод — разный отбор пациенток (ICSI чаще при мужском факторе), поэтому "
                f"разница FR между методами не сравнивает технику напрямую. Смешанный IVF+ICSI отдельного порога в "
                f"консенсусе не имеет.</div></div>")
    else:
        body = _empty("В выгрузке не удалось определить метод оплодотворения — раздел недоступен.")
    tabs.append(("t-fr", "Оплодотворение", body))

    # --- TGBDR -----------------------------------------------------------
    fc_, fa = fig_corr(res["correlations"]), fig_age(res["tgbdr_age"])
    if fc_ or fa:
        body = (f"<p class='lead'>Какие признаки цикла связаны с выходом бластоцист хорошего качества. TGBDR — "
                f"хорошие бластоцисты на 2PN (знаменатель — зиготы, не бластоцисты). Закрашенная точка — связь "
                f"значима после поправки Бенджамини–Хохберга; † — признак математически сопряжён с целевой долей "
                f"(общий знаменатель 2PN), его корреляция частично артефакт построения.</p>"
                + (f"<div class='card'>{_div(fc_, 'f-corr')}</div>" if fc_ else "")
                + (f"<div class='row'><div class='card'>{_div(fa, 'f-age')}</div><div class='card'>"
                   f"<h3 style='margin:4px 0 8px;font-size:15px'>Как читать</h3><ul class='small'>"
                   f"<li>Корреляция Спирмена устойчива к выбросам и нелинейности, но это связь, а не причина: "
                   f"возраст влияет и на число ооцитов, и на качество эмбрионов одновременно.</li>"
                   f"<li>Число хороших бластоцист на цикл растёт с числом ооцитов почти механически; доля TGBDR "
                   f"от этого эффекта очищена и лучше отражает качество культивирования.</li>"
                   f"<li>Единица — цикл. Доли с 2PN &lt; {st.MIN_DEN_DISPLAY} исключены из корреляций TGBDR как "
                   f"вырожденные.</li><li>Модельная величина Bl→Good (CSDI) — другая доля; мост: TGBDR = (Bl→Good) × BlR. "
                   f"Сравнивать Bl→Good с порогами TGBDR нельзя.</li></ul></div></div>" if fa else ""))
    else:
        body = _empty("Нет колонок для TGBDR (2PN и число хороших бластоцист).")
    tabs.append(("t-tgb", "Факторы TGBDR", body))

    # --- Данные и методы --------------------------------------------------
    cols = res["columns"]
    mapped = "".join(f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in cols["mapped"].items()
                     if k not in ("patient_id", "patient_name", "dob"))
    phi_used = [k for k in ("patient_id", "patient_name", "dob") if k in cols["mapped"]]
    amb = "".join(f"<li>{escape(k)}: {escape(', '.join(v))}</li>" for k, v in cols["ambiguous"].items())
    ex = res["exclusions"]
    ex_rows = [("строк в выгрузке", ex["rows_total"]), ("перенос/инсеминация без своей воронки", ex["stale_funnel"]),
               ("строки итогов / неправдоподобные счётчики", ex.get("summary_or_implausible", 0)),
               ("без получения ооцитов", ex["no_retrieval"]), ("freeze-all", ex["freeze_all"]),
               ("цикл не закрыт (нет Bl)", ex["open_cycle_no_bl"]), ("вошли в популяцию лаборатории", ex["lab_population"]),
               ("из них хирургическая сперма (вне FR/DegR)", ex["surgical_sperm_excluded_from_fr"]),
               ("из них «только пункция» (только OCCR/MIIR)", ex["retrieval_only_doctor_only"])]
    if res.get("two_pn_zero_cycles"):
        ex_rows.append(("из них 2PN = 0 (вне KPI по правилу проекта)", res["two_pn_zero_cycles"]))
    ex_html = "".join(f"<tr><td>{escape(k)}</td><td class='n'>{v}</td></tr>" for k, v in ex_rows)
    notes = "".join(f"<li>{escape(n)}</li>" for n in res["notes"])
    su = res["settings_used"]
    body = (f"<div class='row'><div class='card'><h3 style='margin:4px 0 8px;font-size:15px'>Сопоставление колонок</h3>"
            f"<div class='small muted' style='margin-bottom:6px'>Только точное совпадение с синонимами из "
            f"config/demo/column_aliases.yaml — неизвестные имена не угадываются.</div>"
            f"<div class='scroll'><table><tr><th>Каноническое имя</th><th>Колонка файла</th></tr>{mapped}</table></div>"
            + (f"<p class='small'><b>Неоднозначно (не сопоставлено):</b></p><ul class='small'>{amb}</ul>" if amb else "")
            + f"<p class='small muted'>Не найдены: {escape(', '.join(cols['missing'])) or '—'}.</p>"
            + (f"<p class='small muted'>Идентификаторы ({escape(', '.join(phi_used))}) использованы только для "
               f"cycle_uid (HMAC) и удалены до расчётов.</p>" if phi_used else "")
            + f"</div><div class='card'><h3 style='margin:4px 0 8px;font-size:15px'>Популяция анализа</h3>"
            f"<table>{ex_html}</table>" + (f"<ul class='small'>{notes}</ul>" if notes else "")
            + "</div></div>"
            f"<div class='card'><h3 style='margin:4px 0 8px;font-size:15px'>Методы</h3><ul class='small'>"
            f"<li><b>KPI</b> — Σ числитель / Σ знаменатель, никогда не среднее долей. Пороги компетентности и "
            f"бенчмарка — из ews.vendor.kpi.KPI_SPECS (Венский консенсус); FR по методу — из "
            f"config/consensus/vienna_fertilization.yaml.</li>"
            f"<li><b>95% ДИ</b> — Уилсон по эффективному объёму n/φ, где φ — Pearson X²/df между циклами "
            f"(эмбрионы одного цикла не независимы). Для строк «числитель &gt; знаменателя» (ClR&gt;100%, MII&gt;ОКК) "
            f"ДИ не считается — такие строки остаются наблюдением.</li>"
            f"<li><b>Скользящая доля</b> — последние {su['rolling_window_cycles']} циклов, Σ/Σ.</li>"
            f"<li><b>CUSUM</b> — бернуллиевский, одно обновление на цикл при закрытии культивирования, сдвиг шансов "
            f"×0.5 для метрик нижнего контроля. Пороги h подобраны параметрическим бутстрепом "
            f"({su['replicates']} повторов, бета-биномиальный разброс, seed {su['seed']}) под совместную цель "
            f"проекта. Это демо-калибровка на той же выгрузке: alert заблокирован, пересечение h_alert понижается до watch.</li>"
            f"<li><b>EWMA</b> — λ = {su['ewma_lambda']}, пределы ±{su['ewma_L']:.0f}σ с поправкой на внутриклассовую "
            f"корреляцию ρ (L не откалиброван — конвенция для демо).</li>"
            f"<li><b>p-карта</b> — месячные доли, биномиальные 3σ и Laney p′ (σ_z по скользящему размаху).</li>"
            f"<li><b>Сравнение групп</b> — квазибиномиальный F-тест (χ²/φ) либо точный тест Фишера при малых "
            f"ожидаемых частотах; устойчивость — Краскел–Уоллис / Манн–Уитни по долям циклов (знаменатель ≥ "
            f"{su['min_den_display']}); попарно — с поправкой Холма, только после значимого омнибуса. Группы &lt; "
            f"{su['min_cycles_test']} циклов в тест не входят.</li>"
            f"<li><b>Корреляции</b> — Спирмен, 95% ДИ Бонетта–Райта, поправка Бенджамини–Хохберга.</li>"
            f"<li><b>Чего здесь нет</b> — логов инкубатора, лотов среды и прогнозного конверта Digital Twin / TabPFN: "
            f"риск-коррекция и независимое подтверждение сигналов на этих данных невозможны.</li></ul></div>"
            f"<div class='small muted'>Источник: {escape(res['source_name'])} · {stamp}</div>")
    tabs.append(("t-data", "Данные и методы", body))

    nav = "".join(f"<button data-tab='{tid}'>{escape(name)}</button>" for tid, name, _ in tabs)
    sections = "".join(f"<section class='tab' id='{tid}'>{content}</section>" for tid, _, content in tabs)
    if plotly_js == "cdn":
        script = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'
    else:
        from plotly.offline import get_plotlyjs
        script = f"<script>{get_plotlyjs()}</script>"
    meta = json.dumps({"environment": "draft", "mode": "demo", "generated_at": res["generated_at"],
                       "source": res["source_name"]}, ensure_ascii=False)
    return (f"<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<meta name='ews-stamp' content='{escape(meta)}'>"
            f"<title>EWS демо · KPI лаборатории</title><style>{CSS}</style>{script}</head><body>"
            f"<header><div class='wrap'><div class='head'><div><h1>Панель KPI лаборатории эмбриологии</h1>"
            f"<div class='sub'>Система раннего предупреждения EWS · демонстрация на выгрузке "
            f"«{escape(res['source_name'])}» · {res['n_cycles']} циклов</div></div>"
            f"<div class='stamp'>{escape(stamp)}</div></div><nav>{nav}</nav></div></header>"
            f"<main><div class='wrap'>{sections}</div></main>"
            f"<footer><div class='wrap'>Наблюдения и поводы для разбора — не диагнозы, не назначения и не оценка "
            f"конкретного специалиста. Числа считает код; текст «что проверить» — из Венского консенсуса. {escape(stamp)}"
            f"</div></footer><script>{JS}</script></body></html>")
