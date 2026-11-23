"""Прогноз Digital Twin 7.1 для циклов, у которых его нет или устарел.

Колонку DIGITAL TWIN из листа OPU убрали, и вероятность DT берётся из базы
«Two-Line IVF — Digital Twin» в Teable (`sync_opu.dt_probabilities`). Эту базу
раньше пополнял только ручной прогон пайплайна, поэтому новые циклы неделями
оставались без DT, а медленный контур — без опоры. Здесь это делается
автоматически, тем же расчётом, через REST API службы DT 7.1 (подключение
«Аналитика», `/analytics/batches`).

Что отправляется. Цикл считается, если (а) в базе DT для него нет вероятности
или (б) его входы изменились с прошлого расчёта: пришёл новый этап воронки
(ОКК/MII -> 2PN -> бластоцисты) или исправлены данные. Пересчёт не затирает
историю: каждый расчёт — новая строка Predictions с ключом
`<cycle_uid>::<run_id>`, в Cycles — снимок последнего. Так в базе копится
прогноз во времени, а не только итог.

Подстановки — только с флагом и только разрешённые владельцем (2026-10-18,
`dt_refresh.imputation` в профиле клиники). Служба сама ничего не подставляет
(`options` пакета принимает лишь `n_sim` и `clinic_prior`), поэтому подстановку
делает отправитель, и каждая попадает в «Заметки» прогноза. Реальное значение
всегда главнее: константа применяется только к пропуску. В Cycles пишутся только
реальные значения — подставленные остаются во входах расчёта, а не в фактах.

Чего этот шаг не делает. Не пересчитывает прогнозы, поставленные пайплайном, без
изменения входов (первый запуск снимает их отпечаток как исходный). Не считает
циклы без пункции (NO) и переносы из криоцикла (Cryo): там «инсеминировано = 0»,
и 2PN = 0 читался бы моделью как провал оплодотворения. Не является клиническим
прогнозом: `scoring=automatic_research`, `clinician_reviewed=false`, калибровка
центра не применяется.

Приватность. Шаг стоит на той же границе, что и синхронизация листа: читает
идентификаторы пациентки из зеркала EWS и пишет их в базу DT, где они уже
хранятся. В службу DT уходит только `cycle_uid` и клинические числа — без ФИО,
ID и дат. В вывод не попадает ничего, кроме счётчиков и `cycle_uid`.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from .config import Config

UNITS = {"age": "years", "amh": "ng/mL", "afc": "count", "bmi": "kg/m2",
         "height_cm": "cm", "weight_kg": "kg"}

# Поля зеркала EWS (таблица OPU), из которых собирается вход DT.
MIRROR_FIELDS = ["ID", "Имя", "Дата OPU", "cycle_uid", "Возраст", "AMH", "КАФ", "BMI",
                 "№ попытки", "Сперма", "Фолликулы", "ОКК", "MII", "Инсеминировано",
                 "2pN", "Бласт", "Хор.бласт", "Дробление D3", "Криоконсервировано",
                 "Перенесено", "День переноса", "Дата ET", "Метод", "Эмбриолог", "Врач",
                 "Исход"]

# Сперма в листе: N — эякулят; SMF/TESE/micro-TESE без диагноза OA/NOA —
# testicular_OA (правило DT 7.1, флаг sperm_source_inferred).
SPERM_TOKENS = {"N": "ejaculate", "EJ": "ejaculate", "EJACULATE": "ejaculate",
                "DONOR": "donor", "DON": "donor",
                "SMF": "testicular_OA", "TESE": "testicular_OA", "MICRO-TESE": "testicular_OA",
                "MESA": "epididymal", "PESA": "epididymal"}

CLUSTER_LABELS = {"0": "C0 Standard", "1": "C1 Poor", "2": "C2 High",
                  "c0": "C0 Standard", "c1": "C1 Poor", "c2": "C2 High"}

SOURCE_LABEL = "IVF DT 7.1 REST analytics (EWS)"


# --- значения ----------------------------------------------------------------

def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return None if number != number or number in (float("inf"), float("-inf")) else number


def _int(value) -> int | None:
    number = _number(value)
    return None if number is None else int(round(number))


def _text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean(fields: dict) -> dict:
    return {k: v for k, v in fields.items() if v is not None}


def local_date(stamp, offset_hours: float) -> datetime | None:
    """Дата OPU из Teable: UTC-метка полуночи клиники -> календарная дата."""
    text = _text(stamp)
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment + timedelta(hours=offset_hours)


def dt_key(patient_id, opu: datetime | None) -> str | None:
    """Ключ строки Cycles базы DT: `ID#ГГГГММДД`, ID в верхнем регистре без пробелов."""
    pid = (_text(patient_id) or "").upper().replace(" ", "")
    if not pid or opu is None:
        return None
    return f"{pid}#{opu:%Y%m%d}"


# --- вход расчёта -------------------------------------------------------------

@dataclass
class Rules:
    """Разрешённые владельцем подстановки (профиль клиники, `dt_refresh.imputation`)."""
    amh_constant: float | None = None
    bmi_constant: float | None = None
    attempt: int | None = None
    sperm_source: str | None = None
    afc_from_follicles: bool = False
    okk_for_thawed_oocytes: bool = False
    exclude_methods: tuple[str, ...] = ()

    @classmethod
    def from_profile(cls, profile: dict, columns: dict) -> "Rules":
        rules = (profile.get("dt_refresh") or {}).get("imputation") or {}
        exclusion = (columns or {}).get("competency_exclusion") or {}
        tokens = [*exclusion.get("no_retrieval_tokens", []), *exclusion.get("freeze_all_tokens", [])]
        return cls(amh_constant=_number(rules.get("amh_constant")),
                   bmi_constant=_number(rules.get("bmi_constant")),
                   attempt=_int(rules.get("attempt")),
                   sperm_source=_text(rules.get("sperm_source")),
                   afc_from_follicles=bool(rules.get("afc_from_follicles")),
                   okk_for_thawed_oocytes=bool(rules.get("okk_for_thawed_oocytes")),
                   exclude_methods=tuple(str(t).strip().upper() for t in tokens))


@dataclass
class Snapshot:
    """Вход одного цикла: то, что уходит в службу, и что с ним сделано."""
    patient: dict
    flags: list[str] = field(default_factory=list)
    skip: str | None = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.patient, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_snapshot(fields: dict, rules: Rules) -> Snapshot:
    """Вход DT из строки зеркала EWS. Пустое — неизвестно, ноль — наблюдение."""
    method = (_text(fields.get("Метод")) or "").upper()
    if method and method in rules.exclude_methods:
        return Snapshot({}, skip=f"метод {method}: не свежая пункция")

    flags: list[str] = []
    age = _number(fields.get("Возраст"))
    amh = _number(fields.get("AMH"))
    if amh is None and rules.amh_constant is not None:
        amh = rules.amh_constant
        flags.append("amh_imputed")
    bmi = _number(fields.get("BMI"))
    if bmi is None and rules.bmi_constant is not None:
        bmi = rules.bmi_constant
        flags.append("bmi_imputed")
    follicles = _int(fields.get("Фолликулы"))
    afc = _int(fields.get("КАФ"))
    if afc is None and rules.afc_from_follicles and follicles is not None:
        afc = follicles
        flags.append("afc_from_follicles")
    attempt = _int(fields.get("№ попытки"))
    if attempt is None and rules.attempt is not None:
        attempt = rules.attempt
        flags.append("attempt_imputed")
    token = (_text(fields.get("Сперма")) or "").upper()
    sperm = SPERM_TOKENS.get(token)
    if sperm is not None:
        flags.append("sperm_source_inferred")
    elif rules.sperm_source:
        sperm = rules.sperm_source
        flags.append("sperm_source_imputed")

    okk, mii = _int(fields.get("ОКК")), _int(fields.get("MII"))
    pn2, blasts, good = _int(fields.get("2pN")), _int(fields.get("Бласт")), _int(fields.get("Хор.бласт"))
    if okk is not None and mii is not None and mii > okk and rules.okk_for_thawed_oocytes:
        # MII > ОКК — в программу добавлены размороженные ооциты, не ошибка подсчёта.
        okk = mii
        flags.append("okk_raised_for_thawed")
    if _number(fields.get("Инсеминировано")) == 0:
        # Инсеминации не было: 2PN = 0 здесь не наблюдение, а отсутствие события.
        pn2 = blasts = good = None
        flags.append("no_insemination_stages_dropped")

    patient = _clean({"age": age, "amh": amh, "afc": afc, "bmi": bmi, "attempt": attempt,
                      "sperm_source": sperm, "follicles": follicles,
                      "known_okk": okk, "known_mii": mii, "known_pn2": pn2,
                      "known_blasts": blasts, "known_good": good})
    missing = [name for name in ("age", "amh", "afc", "bmi", "attempt", "sperm_source")
               if name not in patient]
    return Snapshot(patient, flags, skip=("нет обязательных: " + ", ".join(missing)) if missing else None)


def stage_of(patient: dict) -> str:
    for key, label in (("known_blasts", "бластоцисты"), ("known_pn2", "2PN"),
                       ("known_mii", "MII"), ("known_okk", "ОКК")):
        if key in patient:
            return label
    return "до пункции"


# --- выбор циклов ---------------------------------------------------------------

@dataclass
class Target:
    uid: str            # cycle_uid EWS: он же cycle_id пакета
    key: str            # ID пациента в базе DT
    fields: dict        # строка зеркала EWS
    snapshot: Snapshot
    record: dict | None  # строка Cycles базы DT, если есть
    reason: str


def select(mirror: list[dict], dt_rows: dict[str, dict], state: dict, rules: Rules,
           offset_hours: float) -> tuple[list[Target], dict]:
    """Какие циклы считать. Возвращает цели и счётчики (без идентификаторов).

    Первый проход по циклу, у которого прогноз уже есть, снимает отпечаток входов
    как исходный и ничего не считает: иначе первый запуск пересчитал бы всю
    историю, поставленную пайплайном, хотя входы не менялись.
    """
    scored = state.setdefault("scored", {})
    targets: list[Target] = []
    counts = {"циклов": 0, "исключено методом": 0, "не хватает данных": 0,
              "без DT": 0, "новый этап/правка": 0, "исходный отпечаток": 0, "актуальны": 0}
    for fields in mirror:
        uid = _text(fields.get("cycle_uid"))
        key = dt_key(fields.get("ID"), local_date(fields.get("Дата OPU"), offset_hours))
        if not uid or not key:
            continue
        counts["циклов"] += 1
        snapshot = build_snapshot(fields, rules)
        if snapshot.skip:
            counts["исключено методом" if snapshot.skip.startswith("метод") else "не хватает данных"] += 1
            continue
        record = dt_rows.get(key)
        previous = scored.get(uid)
        if previous and previous.get("invalid") and previous.get("fingerprint") == snapshot.fingerprint:
            # Служба уже отклонила ровно этот вход: слать его каждый день незачем,
            # он вернётся в работу, как только данные в листе исправят.
            counts["отклонены службой ранее"] = counts.get("отклонены службой ранее", 0) + 1
            continue
        has_dt = record is not None and _number(record["fields"].get("DIGITAL TWIN (посл.)")) is not None
        if not has_dt:
            reason = "без DT"
        elif previous is None:
            scored[uid] = {"fingerprint": snapshot.fingerprint, "run_id": None,
                           "stage": stage_of(snapshot.patient), "baseline": True}
            counts["исходный отпечаток"] += 1
            continue
        elif previous.get("fingerprint") != snapshot.fingerprint:
            reason = "новый этап/правка"
        else:
            counts["актуальны"] += 1
            continue
        counts[reason] += 1
        targets.append(Target(uid, key, fields, snapshot, record, reason))
    return targets, counts


# --- служба DT ----------------------------------------------------------------

class ApiError(RuntimeError):
    pass


class DigitalTwinApi:
    """Клиент `/analytics/*`. Ключ — только в заголовке, TLS — по сертификату службы."""

    def __init__(self, url: str, token: str, ca_cert: str | Path, timeout: float = 60):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.verify = str(ca_cert)
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict | None = None):
        data = None
        headers = dict(self.headers)
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        response = requests.request(method, f"{self.url}{path}", data=data, headers=headers,
                                    verify=self.verify, timeout=self.timeout)
        if response.status_code >= 400:
            raise ApiError(f"{method} {path}: HTTP {response.status_code} {response.text[:300]}")
        return response.status_code, (response.json() if response.content else {})

    def health(self) -> dict:
        return self._call("GET", "/health")[1]

    @staticmethod
    def _body(cycles: list[dict], n_sim: int) -> dict:
        return {"schema_version": "1", "units": UNITS, "options": {"n_sim": n_sim},
                "cycles": cycles}

    def validate(self, cycles: list[dict], n_sim: int) -> dict:
        """Та же проверка пакета службой, без сохранения и без расчёта."""
        return self._call("POST", "/analytics/validate", self._body(cycles, n_sim))[1]

    def submit(self, cycles: list[dict], n_sim: int) -> dict:
        return self._call("POST", "/analytics/batches", self._body(cycles, n_sim))[1]

    def status(self, batch_id: str) -> dict:
        return self._call("GET", f"/analytics/batches/{batch_id}")[1]

    def results(self, batch_id: str) -> dict | None:
        code, payload = self._call("GET", f"/analytics/batches/{batch_id}/results")
        return payload if code == 200 else None

    def delete(self, batch_id: str) -> None:
        self._call("DELETE", f"/analytics/batches/{batch_id}")

    def wait(self, batch_id: str, max_minutes: float, poll_seconds: float) -> dict | None:
        """Ждать результат не дольше `max_minutes`; None — ещё считается."""
        deadline = time.monotonic() + max_minutes * 60
        while True:
            results = self.results(batch_id)
            if results is not None:
                return results
            if time.monotonic() >= deadline:
                return None
            time.sleep(max(15.0, poll_seconds))   # документация: не чаще раза в 15 с


def connect_api(cfg: Config) -> DigitalTwinApi:
    machine = cfg.external("digital_twin_api")
    return DigitalTwinApi(machine["url"], cfg.secret("digital_twin_api"), machine["ca_cert"])


# --- строки базы DT -------------------------------------------------------------

def run_id_for(batch_id: str, when: datetime, clinic_code: str) -> str:
    return f"{clinic_code}-{when:%Y%m%d-%H%M}-{hashlib.sha256(batch_id.encode()).hexdigest()[:12]}"


def _pct(value) -> str:
    number = _number(value)
    return "—" if number is None else f"{number * 100:.1f}%"


def notes(values: dict, target: Target, batch: dict, n_sim: int) -> str:
    kind = _text(values.get("clinical_probability_kind")) or "—"
    flags = ", ".join(target.snapshot.flags) or "нет"
    return "\n".join([
        f"IVF Digital Twin 7.1 · REST analytics · пакет {batch.get('id', '')[:8]} · "
        f"as_entered · n_sim={n_sim} · model {str(batch.get('model_fingerprint') or '')[:12]}",
        "Автоматический исследовательский расчёт: не проверен врачом, без калибровки центра.",
        "Исход — клиническая беременность, подтверждённая УЗИ (включая внематочную), не рождение.",
        f"Повод: {target.reason}; известно до этапа: {stage_of(target.snapshot.patient)}.",
        f"Итог BEFE (L7): {_pct(values.get('clinical_probability'))} · вид: {kind}",
        f"Диапазон неопределённости модели (не доверительный интервал): "
        f"{_pct(values.get('befe_ci_low'))} – {_pct(values.get('befe_ci_high'))}",
        f"Надёжность: {_text(values.get('befe_reliability_band')) or '—'} · "
        f"OOD: {_text(values.get('ood_final')) or '—'}",
        f"Подстановки (решение владельца 2026-10-18): {flags}",
        "Слои L1–L6 не усредняются и не заменяют итог.",
    ])


def prediction_row(values: dict, uid: str, key: str, run_id: str, stamp: str,
                   target: Target, batch: dict, n_sim: int) -> dict:
    cluster = _text(values.get("dominant_cluster"))
    n = _number
    return _clean({
        "Ключ": f"{uid}::{run_id}", "Пациентка": key, "Прогон": run_id, "Timestamp": stamp,
        "med OCC": n(values.get("med_okk")), "med MII": n(values.get("med_mii")),
        "med 2PN": n(values.get("med_pn2")), "med Bl": n(values.get("med_blasts")),
        "med Good": n(values.get("med_good")), "med Euploid": n(values.get("med_euploid")),
        "DIGITAL TWIN": n(values.get("clinical_probability")),
        "P per transfer": n(values.get("p_per_transfer")),
        "P overall cycle": n(values.get("p_overall_cycle")),
        "P cum if viable": n(values.get("p_cum_if_viable")),
        "Bayes mean": n(values.get("bayes_mean")),
        "Bayes CI низ": n(values.get("bayes_ci_low")), "Bayes CI верх": n(values.get("bayes_ci_high")),
        "KAT": n(values.get("p_kat_raw")), "NVSA": n(values.get("p_nvsa")),
        "CSDI": n(values.get("p_csdi")),
        "GNN": n(values.get("gnn_ensemble_prob")), "GNN raw": n(values.get("gnn_prob")),
        "Кластер": CLUSTER_LABELS.get(cluster.replace(".0", "").lower()) if cluster else None,
        "C0 prob": n(values.get("cluster_c0_prob")), "C1 prob": n(values.get("cluster_c1_prob")),
        "C2 prob": n(values.get("cluster_c2_prob")),
        "P viable": n(values.get("p_viable")), "P cancel": n(values.get("p_cancel_risk")),
        "OHSS умеренный": n(values.get("ohss_moderate")), "OHSS тяжёлый": n(values.get("ohss_severe")),
        "OHSS risk": n(values.get("ohss_any")),
        "P без бластоцист": n(values.get("p_no_blast")),
        "P без хор. бластоцист": n(values.get("p_no_good_blast")),
        "Источник NN": SOURCE_LABEL,
        "Известно OCC": n(values.get("known_okk")), "Известно MII": n(values.get("known_mii")),
        "Известно 2PN": n(values.get("known_pn2")), "Известно Bl": n(values.get("known_blasts")),
        "Известно Good Bl": n(values.get("known_good")),
        "Заметки": notes(values, target, batch, n_sim),
    })


OUTCOME = {1: "Беременность", 0: "Нет"}


def cycle_facts(fields: dict, clinic_name: str, offset_hours: float) -> dict:
    """Факты цикла для Cycles базы DT — только реальные значения, без подстановок."""
    opu = local_date(fields.get("Дата OPU"), offset_hours)
    et = local_date(fields.get("Дата ET"), offset_hours)
    outcome = _int(fields.get("Исход"))
    n = _number
    return _clean({
        "Клиника": clinic_name,
        "Возраст": n(fields.get("Возраст")), "AMH": n(fields.get("AMH")),
        "AFC": n(fields.get("КАФ")), "BMI": n(fields.get("BMI")),
        "Попытка": n(fields.get("№ попытки")),
        "Дата OPU": opu.date().isoformat() if opu else None,
        "Дата ET": et.date().isoformat() if et else None,
        "N фолликулов": n(fields.get("Фолликулы")), "OCC": n(fields.get("ОКК")),
        "MII": n(fields.get("MII")), "Инсеминировано": n(fields.get("Инсеминировано")),
        "2pN": n(fields.get("2pN")), "Cleavage": n(fields.get("Дробление D3")),
        "Bl": n(fields.get("Бласт")), "Good Bl": n(fields.get("Хор.бласт")),
        "Cryo": n(fields.get("Криоконсервировано")), "ET": n(fields.get("Перенесено")),
        "День ET": n(fields.get("День переноса")), "IVF/ICSI": _text(fields.get("Метод")),
        "Эмбриолог": _text(fields.get("Эмбриолог")), "Врач": _text(fields.get("Врач")),
        "Исход": OUTCOME.get(outcome, "Неизвестно"),
    })


def cycle_snapshot(values: dict, run_id: str, n_runs: int) -> dict:
    cluster = _text(values.get("dominant_cluster"))
    n = _number
    return _clean({
        "DIGITAL TWIN (посл.)": n(values.get("clinical_probability")),
        "GNN (посл.)": n(values.get("gnn_ensemble_prob")),
        "GNN raw (посл.)": n(values.get("gnn_prob")),
        "KAT (посл.)": n(values.get("p_kat_raw")), "NVSA (посл.)": n(values.get("p_nvsa")),
        "CSDI (посл.)": n(values.get("p_csdi")), "Bayes (посл.)": n(values.get("bayes_mean")),
        "P per transfer (посл.)": n(values.get("p_per_transfer")),
        "Источник NN": SOURCE_LABEL,
        "Кластер": CLUSTER_LABELS.get(cluster.replace(".0", "").lower()) if cluster else None,
        "Прогонов": n_runs, "Последний прогон": run_id,
    })
