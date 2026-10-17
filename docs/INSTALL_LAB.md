# INSTALL_LAB.md — установка на лабораторной машине

Инструкция рассчитана на выполнение через Claude Code: команды даны так, чтобы их
можно было прогонять по порядку и на каждом шаге видеть, что получилось. Клиника та
же, что и дома, данные те же — различается только доступ к источникам.

## Что переносится

**Копируется папка проекта целиком**, а не `git clone`: два куска в git не входят.

| что | размер | почему не в git |
|---|---|---|
| `external/tabpfn/models/*.ckpt` | 233 МБ | слишком тяжело для репозитория |
| `data/<clinic>/local_private/dev_cohort.xlsx` | ~1 МБ | обучающая когорта, данные пациенток |

**Не копировать:** `external/tabpfn/.venv/` (венвы непереносимы — пути внутри
абсолютные), `data/<clinic>/` кроме `local_private/dev_cohort.xlsx`, `__pycache__`,
`.pytest_cache`.

Проектов `embryo-lab-data` и отдельной установки TabPFN на лабораторной машине **не
требуется**: всё вендорено внутрь (`src/ews/vendor/`, `external/tabpfn/`). Из Digital
Twin нужны только итоговые вероятности `PRAI` и `DIGITAL TWIN`, и они приходят в
мастер-таблице OPU — сам пайплайн DT запускать не нужно.

---

## Шаг 1. Окружение Python

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Окружение модели конверта — отдельное, у неё свой torch, и смешивать их не надо:

```bash
python -m venv external/tabpfn/.venv
external/tabpfn/.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0
external/tabpfn/.venv/Scripts/python.exe -m pip install -r external/tabpfn/requirements.txt
```

Подходит **тот же интерпретатор, что и для проекта** — Python 3.10 и новее
(`tabpfn` объявляет `requires-python >=3.10`, torch `>=3.9`; проверено на 3.11.5 и
3.12.4). Отдельную версию искать не нужно.

Venv при этом остаётся **отдельным**, и это не формальность: torch занимает 1.7 ГБ, а
окружению EWS он не нужен ни для чего. Разделение также означает, что сломанная
установка модели не утащит за собой основной конвейер — конверт просто не
зарегистрируется, а оценка закрывшихся циклов пройдёт как обычно.

Версии в `requirements.txt` закреплены точно: TabPFN чувствителен к версии torch, и
«поставим посвежее» даёт другой прогноз на тех же данных без единого предупреждения.

## Шаг 2. Конфигурация

```bash
cp config/machine.local.yaml.example config/machine.local.yaml
cp config/settings.local.yaml.example config/settings.local.yaml
```

В `machine.local.yaml` заполнить:

```yaml
environment: production      # данные боевые — те же, что дома
site: lab                    # площадка: здесь есть свежие логи инкубатора

embryo_lab_data:
  project_dir: ""            # пусто: проект автономен
  clinic: "two_line"

teable:
  url: "http://127.0.0.1:3000"
  base_id: ""                # заполнится bootstrap-ом
  opu_table_id: ""

miri_logs:
  # Недельные выгрузки лежат на Google Drive: «Miri/6391/Miri-6391-Incubation-[…]».
  # При установленном Google Drive для компьютера укажите синхронизированную папку
  # (например "G:/Мой диск/…/Miri"), в ней должна быть подпапка 6391. Без клиента
  # Drive — скачайте недельные папки в любой локальный каталог той же структуры.
  dump_dir: "<каталог, содержащий 6391/>"
  incubator: "6391"

airtable:
  base_id: "app..."           # идентификаторы базы расходников — из вашего Airtable
  items_table_id: "tbl..."
  changes_table_id: "tbl..."

tabpfn_regression:
  enabled: true
  vendored: true
  project_dir: "external/tabpfn"
  python: "external/tabpfn/.venv/Scripts/python.exe"
  script: "tabpfn3_ivf_regression_report_CPU_16K.py"
  model_cache_dir: "external/tabpfn/models"
```

Токены Teable и Airtable — только в `settings.local.yaml`, и там же гейты
`enabled.teable: true`, `enabled.airtable: true`.

`environment` — про ДАННЫЕ, `site` — про МАШИНУ. Клиника одна, выгрузка одна, поэтому
окружение боевое на обеих машинах и пороги между ними переносятся законно. Различие
машин — в доступе к логам инкубатора, и оно в `site`.

## Шаг 3. Проверки перед первым прогоном

```bash
python scripts/ews_sync_vendor.py --check
python scripts/ews_setup_tabpfn.py --clinic two_line --check
python scripts/preflight.py
```

Все три обязательны и все три говорят, чего именно не хватает. `preflight` не
формальность: он отдельно проверяет пакеты, отсутствие которых даёт **молчаливо
пропущенный шаг** вместо ошибки. В соседнем проекте такое случалось трижды, и однажды
стоило неверного вывода в отчёте, а не только времени.

Прогон не начинать, пока `preflight` не вернул 0.

## Шаг 4. База OPU и первое наполнение

```bash
python scripts/ews_bootstrap_opu.py --clinic two_line
python scripts/ews_sync_opu.py      --clinic two_line
```

Bootstrap идемпотентен: недостающие таблицы создаёт, поля добавляет, ничего не
переименовывает и не удаляет. Повторный запуск — штатный способ применить расширение
схемы.

## Шаг 5. Ретроспектива и калибровка

```bash
python scripts/ews_register_envelope.py --clinic two_line --retrospective
python scripts/ews_baseline.py          --clinic two_line
python scripts/ews_enrich_miri.py       --clinic two_line
python scripts/ews_backfit.py           --clinic two_line --replicates 4000 --write
python scripts/ews_run.py               --clinic two_line --replay
```

Порядок обязателен: бэкфит опирается на опорные параметры, а прогон — на пороги.
Регистрация конверта и обогащение средой занимают минуты каждая (TabPFN обучается в
контексте на 16k строк; логи MIRI перекрываются между окнами).

**Читать в первую очередь — `baseline_caveats`** в выводе `ews_baseline.py`: если
опорное окно вобрало значимый сдвиг, скрипт скажет об этом прямо.

Пороги записываются в `config/thresholds.two_line.local.yaml`. Если они приехали с
домашней машины и данные те же — их можно оставить; если выгрузка расширилась
заметно, пересчитайте.

## Шаг 6. Расписание

Мастер-таблица и база OPU обновляются в **15:15**. Ставим прогон с запасом:

```bash
python scripts/ews_schedule.py --clinic two_line --install
python scripts/ews_schedule.py --clinic two_line --status
```

Создаются две задачи планировщика Windows:

| задача | когда | что делает |
|---|---|---|
| `EWS_two_line_daily` | ежедневно 15:45 | синхронизация OPU → регистрация конверта → оценка и панель |
| `EWS_two_line_weekly` | понедельник 16:45 | обогащение средой и лотами, пересчёт границ «необычного» |

Раньше 15:45 запускать нельзя: прогон прочитает вчерашние данные и решит, что новых
циклов нет. Недельная задача отделена намеренно — разбор логов всей когорты занимает
минуты и мешать его с ежедневной цепочкой незачем.

Проверить вручную, ничего не записывая:

```bash
python scripts/ews_daily.py --clinic two_line --dry-run
```

Результат каждого запуска — `data/two_line/state/daily_last_run.json`: какие шаги
прошли, какие мягко пропущены и сколько заняли.

## Шаг 7. Что смотреть

- `data/two_line/EWS_Panel.html` — панель: карты с траекториями, три контура, журнал
  тревог с вердиктом.
- `data/two_line/ews_alerts.csv` — карточки тревог.
- `data/two_line/ews_window_context.csv` — обстановка окна по каждому циклу.
- Раздел панели «Медленный контур с поправкой на пациенток» и
  `data/two_line/ews_outcome_ra_strata.csv` — исход против ожидания DT: O/E по
  месяцам, эмбриологам, среде и лотам; `ews_outcome_ra_transfers.csv` — VLAD и
  RA-CUSUM по переносам.

---

## Работа с двух машин

Клиника одна, поэтому обе машины пишут в свои копии данных, но считают одно и то же.
Что важно помнить:

1. **Расписание ставится только на рабочей машине.** Дома оно бессмысленно: там нет
   свежих логов инкубатора, а суточная цепочка всё равно упрётся в устаревшую
   выгрузку.
2. **Состояние карт не сливается между машинами.** `ews_charts_state.json` помнит,
   какие циклы уже оценены; две машины, идущие по одной истории независимо, дадут
   одинаковый результат, но их состояния — разные файлы. Переносить состояние с
   машины на машину не нужно и не следует; при необходимости пересобрать —
   `--replay`.
3. **Дома `site: home`.** Это попадает в штамп артефакта, и по нему видно, откуда
   отчёт и можно ли доверять его средовой части.

## Если что-то пошло не так

| симптом | причина | что делать |
|---|---|---|
| `конвертов нет — карты пойдут на базовой доле` | не отработала регистрация конверта | `ews_setup_tabpfn.py --check` |
| `подтверждение по среде недоступно` | нет базы границ | `ews_enrich_miri.py` |
| `alert ЗАБЛОКИРОВАН` | пороги калиброваны в другом окружении | `ews_backfit.py --write` |
| `колонок нет в выгрузке` | разошлись имена колонок | `config/clinics/two_line/columns.yaml` |
| прогон в 15:45 не нашёл новых циклов | синхронизация OPU ещё не прошла | сдвинуть время задачи `--time` |
