# CONTRACTS.md — версионированные контракты артефактов EWS

Версия контрактов: `0.1`. Прогон прекращается при отсутствии обязательной колонки,
дубликате ключа или несвежем результате внешнего инструмента. Мягкие шаги (среда, RAG,
лоты) при сбое пропускаются с записью причины в `run_manifest.json`, не роняя прогон.

## Артефакты

Таблица описывает то, что скрипты пишут НА САМОМ ДЕЛЕ; расхождение с ней —
нарушение контракта, а не устаревание документа. Состав колонок закреплён тестами
(`tests/test_contracts.py`), потому что рвётся он молча: переименованная колонка не
даёт ни исключения, ни предупреждения.

| Артефакт | Пишет | Ключ | Обязательные поля |
|---|---|---|---|
| `ews_checkpoints.csv` | `ews_run.py` | `cycle_uid,checkpoint` | `cycle_uid`, `checkpoint` (`opu`/`fertilisation`/`closure`), `date_opu`, `closure_due`, `recorded_at`, `environment` |
| `ews_envelopes.csv` | `ews_register_envelope.py` | `cycle_uid,checkpoint` | `cycle_uid`, `registered_at`, `checkpoint`, `registration_mode`, `model_version`, `run_id`, `environment`, ≥1 стадия `TPFN3_<stage>_p10..p90` |
| `ews_envelope_qc.csv` | `ews_run.py` | `cycle_uid,stage` | `cycle_uid`, `stage`, `actual`, `p10`, `p50`, `p90`, `width`, `mode`, `status`, `in80`, `z` |
| `ews_alerts.csv` | `ews_run.py` | `alert_id` | `alert_id`, `cycle_uid`, `raised_at`, `level`, `chart`, `statistic`, `expected_source`, `contour`, `corroboration`, `corroboration_rank`, `miri_status`, `action`, `verdict`, `notes`, `environment`, `raised_in`, `upgraded_at` |
| `ews_window_context.csv` | `ews_enrich_miri.py` | `cycle_uid` | `cycle_uid`, `miri_status`, `episodes`, `critical_minutes`, `lot_changes`, `unusual`, `notes`, `environment` |
| `ews_charts_state.json` | `ews_run.py` | `clinic` | `clinic`, `environment`, `site`, `updated_at`, `evaluated_cycles`, `charts.<metric>.{statistic, observations, peak, signals, watch_episodes, above_watch, trajectory}` |
| `run_manifest.json` | `ews_run.py` | `run_id` | `run_id`, `started_at`, `clinic`, `environment`, `site`, `mode`, `cycles_loaded`, `cycles_evaluated`, `levels`, `escalated_by_context`, `alert_gate`, `soft_step_status`, `outcome_ra`, `contract_version` |
| `incontrol_parameters.json` | `ews_baseline.py` | — | `reference_window`, `metrics[]`, `baseline_caveats`, `block_sizes`, `overdispersion` |
| `calibration_manifest.json` | `ews_backfit.py` | — | `calibrated_at`, `calibrated_environment`, `reference_window`, `charts[]`, `system_false_alarms_per_year`, `baseline_caveats` |
| `ews_outcome_ra_transfers.csv` | `ews_run.py` | `cycle_uid` | `cycle_uid`, `date_opu`, `y`, `p_model`, `p_expected`, `vlad`, `cusum`, `cusum_signal`, `embryologist` (псевдоним), `environment_group`, `environment` |
| `ews_outcome_ra_strata.csv` | `ews_run.py` | `dimension,group` | `dimension`, `group`, `n`, `observed`, `expected`, `oe`, `inner_low/high`, `outer_low/high`, `position`, `environment` |
| `state/outcome_calibration.json` | `ews_run.py` | — | `delta`, `se`, `n`, `frozen_at`, `last_date_opu`, `expected_column`; пишется один раз |
| `agent_packets/outcome_ra_<clinic>.json` | `ews_run.py` | — | `summary`, `strata`; без `cycle_uid` и имён |
| Agent packet | локальный Python | `cycle_uid` | `schema_version`, `cohort`, `patients[].cycle_uid`; PHI запрещена |

Колонка `statistic` в карточке тревоги — это и есть свидетельство (`evidence` ранних
черновиков): накопленная статистика каждой сработавшей карты, рядом с
`expected_source` — откуда взято ожидание (`baseline` или `envelope:<режим>`).

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
- `alert_id` **уникален в пределах файла и не меняется**. Номер продолжается от
  максимума, который уже лежит в журнале (`ews_run.next_alert_number`), а пересборка
  `--replay` возвращает совпавшим карточкам их прежние номера (`ews_run.carry_over`).
  До 2026-09-10 номер брался из позиции цикла в списке оценок текущего прогона и в
  инкрементальном режиме сбрасывался на 1: в боевом журнале лежали две разные
  карточки `two_line-00001`. Идентификатор, на который ссылается разбор, обязан
  идентифицировать.
- `raised_in` ∈ {`прогон`, `replay`} — **происхождение карточки**. `прогон` означает,
  что тревогу подняла оценка в тот день; `replay` — что её восстановила пересборка
  истории, знающая будущее. Ранним обнаружением вторая не является, и без этой
  колонки журнал непригоден для того единственного, ради чего его ведут: судить о
  результативности проспективного отслеживания. Пересборка происхождение **не
  перезаписывает**: карточка, поднятая живым прогоном, остаётся таковой навсегда.
- Обогащение средой работает в два такта. Первый — недельный проход
  (`ews_enrich_miri.py`): считает обстановку по когорте и пересчитывает границы
  «необычного»; тревог не создаёт и уровней не трогает. Второй — прогон
  (`ews_run.py`): повышает `watch` до `alert` по сильному подтверждению, читая
  границы готовыми.
- **Отложенная доводка** (`evaluate.deferred_upgrades`, с 2026-09-10). Прогон
  пересматривает карточки с `miri_status=pending`, окно которых выгрузка логов
  наконец покрыла, и имеет право ровно на два действия: записать ставший известным
  `miri_status` и поднять `watch` до `alert`. Понизить уровень, снять тревогу,
  создать новую строку или тронуть состояние карт доводка не может; правится только
  список `evaluate.UPGRADABLE`. Отметка `upgraded_at` и причина остаются в карточке.
  Причина, по которой это понадобилось: обстановка окна физически приходит позже
  оценки — цикл закрывается на T+6, а логи приезжают недельной нарезкой, — после чего
  цикл попадал в `already_evaluated` и не пересматривался никогда. Маршрут
  `watch → alert` за всю историю площадки не сработал ни разу при том, что документы
  его описывали; обещание было невыполнимым по устройству, а не по обстоятельствам.
- Повышение до `alert` по обстановке считается в `run_manifest.json` **двумя**
  отдельными полями: `escalated_by_context` — в момент оценки, `upgraded_by_context` —
  отложенной доводкой. Калибровка измеряла частоту только для пересечений `h_alert`,
  и смешивать маршруты в одном числе значит терять возможность её сверить; у этих
  двух вдобавок разная задержка обнаружения.
- Поле `statistic` карточки несёт величину, **НА КОТОРОЙ сработала карта**, а не
  остаток после перезапуска. CUSUM обнуляется при пересечении `h_alert`, и до
  2026-09-10 значение читалось уже после обнуления: в журнале лежала карточка уровня
  `alert` со статистикой `0.0`, а панель печатала «сигналов alert 1, пик 6.25 при
  пороге 7.44» — пик ниже порога, который был пересечён. Источник величины —
  `charts.CusumRunner.attained`; в пик и траекторию идёт она же.
- Гейт покрытия конверта, блокирующий эскалацию, судит **только по проспективным
  конвертам** (`p_below_prospective`, решение владельца 2026-09-10). Ретроспективный
  добор писался моделью, обучавшейся в том числе на этих циклах: его покрытие —
  свойство добора, а не прогноза. Измерено 2026-09-09: `blasts_after_fert` давал
  0.699 на 73 ретроспективных против 0.833 на 6 проспективных, в пуле 0.709 — и
  именно пулированное число запирало `alert`. Пулированное покрытие остаётся в
  отчёте: оно нужно аудиту модели. Пока проспективных меньше
  `envelope_qc.MIN_N_FOR_TEST` на стадию, гейт молчит, и это состояние обязано быть
  названо в отчёте отдельно: «проверять нечем» — не то же, что «проверено и чисто».
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
