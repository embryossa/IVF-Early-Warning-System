"""Надёжный ввод/вывод: свежие результаты и атомарная запись артефактов."""
from __future__ import annotations
import os
from pathlib import Path
import re
import tempfile
from typing import Callable
import pandas as pd

# XML 1.0 (and so XLSX) forbids most C0 control characters. Raw scanner barcode
# strings (Airtable "All Barcodes", GS1-128 field separators) carry them — openpyxl
# raises IllegalCharacterError outright rather than dropping them, so every sheet
# is sanitized before write instead of trusting upstream data to be Excel-safe.
_ILLEGAL_XLSX_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    # Column dtype is deliberately not used to pre-filter which columns to scan:
    # pandas' native "str" dtype (distinct from "object" in this project's pandas
    # version) would silently bypass an `dtype == object` check and let illegal
    # characters back into openpyxl. isinstance() below is the only guard needed.
    out = frame.copy()
    for col in out.columns:
        out[col] = out[col].map(
            lambda v: _ILLEGAL_XLSX_CHARS.sub("", v) if isinstance(v, str) else v
        )
    return out


def latest_file(directory: str, pattern: str) -> Path | None:
    files = list(Path(directory).glob(pattern))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def load_latest(directory: str, pattern: str) -> pd.DataFrame:
    f = latest_file(directory, pattern)
    if f is None:
        raise FileNotFoundError(f"нет файлов {pattern} в {directory}")
    return pd.read_csv(f)


def latest_file_after(directory: str, pattern: str, started_at: float) -> Path | None:
    """Возвращает только файл, созданный/обновлённый текущим запуском."""
    files = [p for p in Path(directory).glob(pattern)
             if p.stat().st_mtime >= started_at - 2]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _atomic_path(target: str | Path) -> tuple[Path, Path]:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=target.suffix,
                                dir=target.parent)
    os.close(fd)
    return target, Path(name)


def atomic_write(target: str | Path, writer: Callable[[Path], None]) -> Path:
    """Пишет во временный файл той же папки и заменяет цель через os.replace."""
    target, tmp = _atomic_path(target)
    try:
        writer(tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def atomic_write_csv(df: pd.DataFrame, target: str | Path, **kwargs) -> Path:
    opts = {"index": False, "encoding": "utf-8-sig", **kwargs}
    return atomic_write(target, lambda tmp: df.to_csv(tmp, **opts))


def atomic_write_excel(sheets: dict[str, pd.DataFrame], target: str | Path,
                       engine: str = "openpyxl") -> Path:
    def _write(tmp: Path):
        with pd.ExcelWriter(tmp, engine=engine) as xl:
            for name, frame in sheets.items():
                _excel_safe(frame).to_excel(xl, sheet_name=name[:31], index=False)
            if engine == "openpyxl":
                _style_workbook(xl.book)
    return atomic_write(target, _write)


def _style_workbook(workbook) -> None:
    """Единый читаемый стиль клинических/аналитических экспортов."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="1B4F72")
    fills = {
        "benchmark": PatternFill("solid", fgColor="D9EAD3"),
        "competency": PatternFill("solid", fgColor="FFF2CC"),
        "outside_control": PatternFill("solid", fgColor="F4CCCC"),
        "critical": PatternFill("solid", fgColor="F4CCCC"),
        "warning": PatternFill("solid", fgColor="FCE5CD"),
    }
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.sheet_view.showGridLines = False
        sheet.auto_filter.ref = sheet.dimensions
        sheet.row_dimensions[1].height = 30
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        headers = {cell.column: str(cell.value or "") for cell in sheet[1]}
        for idx, header in headers.items():
            values = [header]
            for row in range(2, min(sheet.max_row, 250) + 1):
                value = sheet.cell(row, idx).value
                values.append("" if value is None else str(value))
                if header in {"rate", "ci_low_95", "ci_high_95", "competency", "benchmark"}:
                    sheet.cell(row, idx).number_format = "0.0%"
                elif header in {"PRAI", "DIGITAL TWIN", "DNN", "p_per_transfer", "p_kat_raw"}:
                    sheet.cell(row, idx).number_format = "0.0%"
                elif "Date" in header or header in {"window_start", "window_end", "start", "end"}:
                    sheet.cell(row, idx).number_format = "yyyy-mm-dd hh:mm"
            longest = max((len(v) for v in values), default=8)
            cap = 55 if header in {"reason", "Reason"} else 36
            sheet.column_dimensions[get_column_letter(idx)].width = max(9, min(cap, longest + 2))

        status_idx = next((idx for idx, name in headers.items()
                           if name in {"status", "severity", "max_severity"}), None)
        if status_idx:
            for row in range(2, sheet.max_row + 1):
                cell = sheet.cell(row, status_idx)
                key = str(cell.value or "").lower()
                if key in fills:
                    cell.fill = fills[key]
                    cell.font = Font(bold=True)
        sheet.print_title_rows = "1:1"
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth = 1
        sheet.sheet_properties.pageSetUpPr.fitToPage = True


def atomic_write_text(text: str, target: str | Path, encoding="utf-8") -> Path:
    return atomic_write(target, lambda tmp: tmp.write_text(text, encoding=encoding))
