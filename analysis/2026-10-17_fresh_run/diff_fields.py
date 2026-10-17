"""Какие поля различаются между новым CSV и Teable. Печатает только имена полей,
счётчики и формы значений (цифра -> 9, буква -> a) для числовых/датовых полей."""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))

from ews import sync_opu, teable_client  # noqa: E402
from ews.config import embryo_lab_data_on_path, load  # noqa: E402

cfg = load()
embryo_lab_data_on_path(cfg)
from ews.vendor.privacy import load_or_create_secret, make_cycle_uid  # noqa: E402

frame = sync_opu.load_sheet_export(ROOT / "data/local_private/OPU table.csv")
client = teable_client.connect(cfg, dry_run=True)
tables = teable_client.resolve_tables(cfg, client, cfg.external("teable")["base_id"])
secret = load_or_create_secret(cfg.path(cfg.settings["privacy"]["secret_file"]))

existing = {}
for rec in client.all_records(tables["OPU"]):
    key = sync_opu._clean(rec["fields"].get("Ключ"))
    if key:
        existing[key] = rec["fields"]

safe = set(sync_opu.NUMERIC_FIELDS) | set(sync_opu.DATE_FIELDS)
ignored = {"Хеш строки", "Обновлено", "Окружение", "Чекпойнт"}


def shape(v):
    return re.sub(r"[^\W\d_]", "a", re.sub(r"\d", "9", "" if v is None else str(v)))


diffs, shapes = Counter(), {}
for _, row in frame.iterrows():
    key = sync_opu.build_key(row)
    if key not in existing:
        continue
    new = sync_opu.row_to_fields(row, make_cycle_uid(key, secret), cfg.environment)
    old = existing[key]
    for name in new:
        if name in ignored:
            continue
        a, b = old.get(name), new.get(name)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) == float(b):
            continue
        if (a in (None, "")) and (b in (None, "")):
            continue
        if str(a) != str(b):
            diffs[name] += 1
            if name in safe:
                shapes.setdefault(name, Counter())[(shape(a), shape(b))] += 1

print("строк сопоставлено:", sum(1 for _, r in frame.iterrows() if sync_opu.build_key(r) in existing))
for name, n in diffs.most_common():
    tag = "" if name in safe else "  (текстовое поле, значения не печатаются)"
    print(f"{n:4d}  {name}{tag}")
    for (a, b), k in shapes.get(name, Counter()).most_common(4):
        print(f"        было «{a}» -> стало «{b}»: {k}")
