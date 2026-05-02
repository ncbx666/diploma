from __future__ import annotations

import calendar
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

YEAR_SHEET_PATTERN = re.compile(r"^G?\d{4}$")


def to_snake_case(name: object) -> str:
    text = str(name).strip().replace(">", "_gt_")
    text = re.sub(r"[^0-9a-zA-Zа-яА-Я]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def canonical_day_sequence(year: int) -> np.ndarray:
    end = 244 if calendar.isleap(year) else 243
    return np.arange(152, end + 1)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.replace({"": np.nan, " ": np.nan, "-": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def load_excel_dataset(excel_path: str | Path) -> pd.DataFrame:
    """Load yearly sheets from the workbook used by the source notebook."""
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel dataset not found: {excel_path}")

    xls = pd.ExcelFile(excel_path)
    frames: list[pd.DataFrame] = []
    for sheet_name in xls.sheet_names:
        if not YEAR_SHEET_PATTERN.fullmatch(str(sheet_name)):
            continue
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=0)
        if raw.empty:
            continue
        df = raw.rename(columns=to_snake_case).copy()
        if "unnamed_0" in df.columns:
            df = df.rename(columns={"unnamed_0": "year"})
        if "year" not in df.columns:
            year_match = re.search(r"\d{4}", str(sheet_name))
            if not year_match:
                continue
            df.insert(0, "year", int(year_match.group(0)))
        for column in df.columns:
            df[column] = _coerce_numeric(df[column])
        df = df.dropna(subset=["year", "day"]).copy()
        df["year"] = df["year"].astype(int)
        df["day"] = df["day"].astype(int)
        frames.append(df)

    if not frames:
        raise ValueError(f"No yearly sheets found in {excel_path}")
    dataset = pd.concat(frames, ignore_index=True).sort_values(["year", "day"]).reset_index(drop=True)
    if "target_favorable" not in dataset.columns:
        raise ValueError("Expected target_favorable column in Excel workbook")
    return dataset


def create_horizon_targets(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    result = df.sort_values(["year", "day"]).copy()
    for horizon in horizons:
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        result[f"target_h{horizon}"] = (
            result.groupby("year", sort=False)["target_favorable"].shift(-horizon)
        )
    return result


def split_years(df: pd.DataFrame, val_year_count: int = 7, test_year_count: int = 5) -> dict[str, list[int]]:
    years = sorted(int(y) for y in df["year"].dropna().unique())
    if len(years) < val_year_count + test_year_count + 1:
        raise ValueError("Not enough years for train/validation/test split")
    train_end = len(years) - val_year_count - test_year_count
    val_end = len(years) - test_year_count
    return {"train": years[:train_end], "val": years[train_end:val_end], "test": years[val_end:]}
