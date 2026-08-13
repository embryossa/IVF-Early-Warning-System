# CONTRACTS.md — версионированные контракты артефактов EWS

Версия контрактов: `0.1`. Прогон прекращается при отсутствии обязательной колонки,
дубликате ключа или несвежем результате внешнего инструмента. Мягкие шаги (среда, RAG,
лоты) при сбое пропускаются с записью причины в `run_manifest.json`, не роняя прогон.

## Артефакты

| Артефакт | Ключ | Обязательные поля |
|---|---|---|
| `ews_checkpoints.csv` | `cycle_uid,checkpoint` | `cycle_uid`, `checkpoint` (`opu`/`fertilisation`/`closure`), `recorded_at`, `environment` |
| `ews_envelopes.csv` | `cycle_uid` | `cycle_uid`, `registered_at`, `model_version`, `run_id`, ≥1 стадия `TPFN3_<stage>_p10..p90` |
| `ews_residuals.csv` | `cycle_uid,stage` | `cycle_uid`, `stage`, `actual`, `p10`, `p50`, `p90`, `z`, `in80` |
| `ews_kpi.csv` | `cycle_uid,metric` | `cycle_uid`, `metric`, `numerator`, `denominator`, `rate`, `ci_low`, `ci_high`, `direction`, `competency`, `benchmark`, `status`, `indicator_class`, `source_note` |
| `ews_charts_state.json` | `chart_name` | `chart_name`, `statistic`, `last_index`, `updated_at`, `h_watch`, `h_alert` |
| `ews_alerts.csv` | `alert_id` | `alert_id`, `cycle_uid`, `raised_at`, `level`, `chart`, `contour`, `evidence`, `corroboration`, `corroboration_rank`, `miri_status`, `environment` |
| Agent packet | `cycle_uid` | `schema_version`, `cohort`, `patients[].cycle_uid`; PHI запрещена |
| `calibration_manifest.json` | — | `calibrated_at`, `window`, `n_cycles`, `n_embryos`, `h_watch`, `h_alert`, `measured_false_alarm_rate` |
| `run_manifest.json` | `run_id` | `run_id`, `started_at`, `environment`, `soft_step_status`, `contract_version` |

## Правила

- `0` — известное наблюдение. Только пустое значение/`NaN` означает пропуск.
- Вероятности хранятся в 0–1; ширина CI — в процентных пунктах.
- Каждый CSV/XLSX/HTML/JSON пишется атомарно (`embryo._io`: временный файл рядом →
  `os.replace`). Полуфабрикатов на диске не остаётся.
- **Конверт неизменяем.** Строка в `ews_envelopes.csv` пишется один раз на T0. Повторная
  запись того же `cycle_uid` — ошибка контракта, а не обновление. Перерегистрация возможна
  только явным административным действием с записью причины.
- **Чекпойнт неизменяем.** `checkpoint=opu` и `checkpoint=fertilisation` не несут полей
  оценки вообще: колонок `level`, `chart`, `z` в этих строках быть не может. Если они
  появились — нарушено неподвижное решение «тревоги только на closure».
- `level` принимает `ok` / `watch` / `alert`. При `environment=draft` или
  `thresholds.calibrated=false` значение `alert` невозможно — контракт это проверяет, а не
  полагается на дисциплину вызывающего кода.
- `miri_status` ∈ {`pending`, `unavailable`, `enriched`}. **Пустое поле среды не означает
  «событий не было».** `pending` — выгрузка ещё не пришла; `unavailable` — окно не покрыто
  выгрузкой и уже не будет; `enriched` — логи разобраны.
- Обогащение средой (`ews_enrich_miri.py`) может изменить `corroboration`,
  `corroboration_rank`, `miri_status` и повысить `level` с `watch` до `alert`. Оно **не
  может** создать строку в `ews_alerts.csv` и не может понизить или удалить существующую.
- `corroboration_rank` ∈ {`strong`, `moderate`, `none`}. Значение `strong` допустимо только
  для источников из `settings.corroboration.strong`. Пара CUSUM+EWMA на одной метрике
  никогда не даёт ранга выше `none` — это не подтверждение (см. `docs/METHODOLOGY.md` §6).
- Лоты: событие `lot_change` возникает только при реальном пересечении более одного периода
  лота с окном культивирования. Смена лота — кандидат-confounder, не причина и не оценка
  качества лота. Два лота одной позиции, открытые одновременно → `lot_attribution=unresolved`,
  привязка не угадывается.
- Биномиальный CI не считается при `numerator > denominator` (`ClR>100%`, `MII>OCC`) —
  интервал математически не определён. Доля остаётся, `ci_low`/`ci_high` = `NaN`.
- `n < 4` на стадии → `"н/д"` (кириллицей). Намеренно не литерал `"n/a"`: он входит в
  `na_values` по умолчанию у `pandas.read_csv` и при повторном чтении молча станет `NaN`.
- Порог сравнения обязан нести `source_note` и `indicator_class` — происхождение значения
  не теряется. Значения, уже существующие в `embryo.kpi.KPI_SPECS`, импортируются, а не
  переписываются в этом проекте.
- Join OPU ↔ конверт ↔ модели выполняется локально по нормализованному ключу; агент получает
  только `cycle_uid`. Crosswalk — в `data/local_private/`.
- Каждый артефакт несёт `environment`. Артефакт, произведённый на черновом стенде, обязан
  быть отличим от лабораторного без чтения содержимого.
