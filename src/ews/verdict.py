"""Совместное прочтение контуров: что тревога ЗНАЧИТ.

Три контура отвечают на три разных вопроса, и по отдельности ни один не даёт
основания для разбора протокола:

* **относительный** — «изменились ли мы относительно СВОЕЙ истории»;
* **абсолютный** — «где мы относительно Венского консенсуса»;
* **литературный** — «где мы относительно ожидаемого для ЭТИХ пациенток».

Неподвижное решение владельца (2026-08-13), задающее смысл всей системы:

    Если лаборатория укладывается в консенсус и в прогноз моделей, просадка
    относительно собственной истории — это НЕ провал и НЕ повод разбирать
    протокол. Это сигнал тренда: меняется стимуляция либо приходят более сложные
    пациентки. Разбор протокола оправдан только когда просадка относительно себя
    совпала с выходом за консенсус.

Почему это важно записать кодом, а не оставить на усмотрение читающего: карта
относительного контура центрирована на нашем собственном уровне, а он у этой
лаборатории ВЫШЕ и консенсуса, и литературы (BlR 0.590 против benchmark 0.60 при
литературных 0.699 и когорте обучения 0.479; FR 0.817 против benchmark 0.75 и
литературных 0.719). Лаборатория, работающая лучше среднего, при таком устройстве
получала бы тревоги чаще прочих — и без этого слоя они читались бы как «стало
плохо», хотя означают «стало иначе, чем у нас же раньше».
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Что делать с тревогой. Порядок — по возрастанию цены вмешательства.
ACTION_TREND = "trend"                  # наблюдать; вероятная причина вне лаборатории
ACTION_REVIEW_CASEMIX = "casemix"       # разобрать состав пациенток и стимуляцию
ACTION_REVIEW_PROTOCOL = "protocol"     # разбор лабораторного протокола
ACTION_STANDING_GAP = "standing_gap"    # давняя системная разница, не событие

STANDING = "standing"


@dataclass(frozen=True)
class ContourState:
    """Положение метрики в каждом из трёх контуров."""
    metric: str
    relative_triggered: bool           # сработала карта относительного контура
    absolute_status: str | None        # 'benchmark' | 'competency' | 'below' | None
    literature_gap_pp: float | None    # наша доля минус литературное ожидание, пп


@dataclass(frozen=True)
class Verdict:
    metric: str
    action: str
    headline: str
    reasoning: str


def _above_consensus(status: str | None) -> bool:
    """Укладываемся ли в консенсус на уровне benchmark."""
    return status == "benchmark"


def _below_competency(status: str | None) -> bool:
    """Ниже нижнего уровня компетентности — это уже не «иначе», а «мало»."""
    return status == "below"


def read(state: ContourState, literature_tolerance_pp: float = -5.0) -> Verdict:
    """Свести три контура в одно действие.

    ``literature_tolerance_pp`` — насколько ниже литературы допустимо быть, оставаясь
    в режиме «тренд». Отрицательное значение: -5 означает «до 5 пп ниже литературы
    не считаем признаком проблемы».
    """
    gap = state.literature_gap_pp
    below_literature = gap is not None and gap < literature_tolerance_pp

    if not state.relative_triggered:
        # Карта молчит. Единственное, что здесь может быть сказано, — давняя
        # разница уровня, которая существует всегда и событием не является.
        if _below_competency(state.absolute_status) or below_literature:
            return Verdict(
                state.metric, ACTION_STANDING_GAP,
                "давняя разница уровня, не событие",
                "относительный контур спокоен — значит, ничего не изменилось; "
                "разница с консенсусом или литературой существует постоянно и "
                "разбирается отдельно от раннего предупреждения, в плановом порядке",
            )
        return Verdict(state.metric, ACTION_TREND, "в норме по всем контурам", "")

    # Карта сработала. Дальше решает, где мы относительно внешних опор.
    if _below_competency(state.absolute_status) and below_literature:
        return Verdict(
            state.metric, ACTION_REVIEW_PROTOCOL,
            "просадка относительно себя И выход за консенсус",
            "совпали все три контура: изменились относительно своей истории, ушли "
            "ниже уровня компетентности и ниже литературного ожидания для этих "
            "пациенток — это основание разбирать лабораторный протокол",
        )

    if _below_competency(state.absolute_status):
        return Verdict(
            state.metric, ACTION_REVIEW_PROTOCOL,
            "просадка относительно себя, ушли ниже компетентности",
            "относительный и абсолютный контуры совпали; литература расхождения не "
            "подтверждает, но выход за уровень компетентности сам по себе достаточен",
        )

    if _above_consensus(state.absolute_status) and not below_literature:
        return Verdict(
            state.metric, ACTION_REVIEW_CASEMIX,
            "тренд: изменились относительно себя, оставаясь выше внешних опор",
            "лаборатория укладывается в консенсус и в литературное ожидание — "
            "это не провал и не повод разбирать протокол. Смотреть надо на состав "
            "пациенток и на стимуляцию: приход более сложной когорты или изменение "
            "протокола стимуляции дают ровно такую картину",
        )

    return Verdict(
        state.metric, ACTION_REVIEW_CASEMIX,
        "тренд на фоне промежуточного уровня",
        "карта сработала, при этом уровень между компетентностью и benchmark: "
        "оснований для разбора протокола нет, но и запаса нет — наблюдать, "
        "проверив состав пациенток",
    )


def build_states(evaluation_metrics: dict, triggered: set[str],
                 baselines: dict[str, dict],
                 literature: pd.DataFrame | None) -> list[ContourState]:
    """Собрать положение по контурам для метрик, где есть наблюдение."""
    gaps: dict[str, float] = {}
    if literature is not None and not literature.empty:
        for _, row in literature.iterrows():
            value = row.get("gap_pp")
            if value is not None and not pd.isna(value):
                gaps[row["metric"]] = float(value)

    states = []
    for metric in evaluation_metrics:
        base = baselines.get(metric) or {}
        status = base.get("status")
        # `status` из embryo.kpi: 'benchmark' | 'competency' | иное. Приводим к
        # трёхзначной шкале, где 'below' — ниже уровня компетентности.
        if status not in ("benchmark", "competency"):
            status = "below" if status else None
        states.append(ContourState(
            metric=metric,
            relative_triggered=metric in triggered,
            absolute_status=status,
            literature_gap_pp=gaps.get(metric),
        ))
    return states


def summarise(verdicts: list[Verdict]) -> str:
    """Одна строка на тревогу для карточки и панели."""
    escalating = [v for v in verdicts if v.action == ACTION_REVIEW_PROTOCOL]
    if escalating:
        return (f"разбор протокола: {', '.join(v.metric for v in escalating)} — "
                f"просадка совпала с выходом за консенсус")
    casemix = [v for v in verdicts if v.action == ACTION_REVIEW_CASEMIX]
    if casemix:
        return (f"тренд, не провал: {', '.join(v.metric for v in casemix)} — "
                f"изменились относительно своей истории, оставаясь в пределах "
                f"консенсуса; смотреть состав пациенток и стимуляцию")
    standing = [v for v in verdicts if v.action == ACTION_STANDING_GAP]
    if standing:
        return (f"давняя разница уровня: {', '.join(v.metric for v in standing)} — "
                f"не событие, плановый разбор")
    return "в норме по всем контурам"
