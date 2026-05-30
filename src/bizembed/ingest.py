from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
import zipfile

import pandas as pd

from bizembed.normalize import FIELD_LABELS, normalize_text


CANONICAL_COLUMNS = [
    "entity_name",
    "entity_type",
    "industry_name",
    "industry_code",
    "item_name",
    "source",
    "text",
]

SOURCE_ENTITY_TYPES = {
    "sbiz": "merchant",
    "nara": "supplier",
    "license": "supplier",
    "ksic": "company",
}

COLUMN_ALIASES: Dict[str, Iterable[str]] = {
    "entity_name": (
        "상호명",
        "업체명",
        "업체명칭",
        "공급업체명",
        "가맹점명",
        "법인명",
        "회사명",
    ),
    "industry_name": (
        "업종명",
        "업종",
        "대표업종",
        "업태명",
        "업태구분명",
        "표준산업분류명",
        "상권업종소분류명",
        "상권업종중분류명",
        "상권업종대분류명",
    ),
    "industry_code": (
        "업종코드",
        "표준산업분류코드",
        "상권업종소분류코드",
        "상권업종중분류코드",
        "상권업종대분류코드",
    ),
    "item_name": (
        "공급물품명",
        "물품명",
        "품목명",
        "대표품명",
        "대표세부품명",
        "세부품명",
        "세부품명(명칭)",
        "물품분류명",
        "상품명",
    ),
}


def first_present_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def normalize_series(series: pd.Series) -> pd.Series:
    """Vectorized companion to normalize_text for large public-data files."""
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace("nan", "")
        .str.replace(r"[()]", " ", regex=True)
        .str.replace(r"[|,/\\;:\t\r\n]+", " ", regex=True)
        .str.replace(r"[\[\]{}<>]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def standardize_dataframe(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Map public-data columns into the canonical embedding-record schema."""
    result = pd.DataFrame(index=df.index)

    for target, aliases in COLUMN_ALIASES.items():
        source_column = first_present_column(df, aliases)
        if source_column is None:
            result[target] = ""
        else:
            result[target] = normalize_series(df[source_column])

    entity_type = SOURCE_ENTITY_TYPES.get(source, "company")
    result["entity_type"] = entity_type
    result["source"] = source

    name_label = FIELD_LABELS.get(entity_type, FIELD_LABELS["company"])
    result["text"] = ""
    has_name = result["entity_name"].str.len() > 0
    result.loc[has_name, "text"] = name_label + ": " + result.loc[has_name, "entity_name"]
    has_industry = result["industry_name"].str.len() > 0
    result.loc[has_industry, "text"] = (
        result.loc[has_industry, "text"]
        + result.loc[has_industry, "text"].map(lambda value: " | " if value else "")
        + "업종명: "
        + result.loc[has_industry, "industry_name"]
    )
    has_item = result["item_name"].str.len() > 0
    result.loc[has_item, "text"] = (
        result.loc[has_item, "text"]
        + result.loc[has_item, "text"].map(lambda value: " | " if value else "")
        + "품목명: "
        + result.loc[has_item, "item_name"]
    )

    result = result[CANONICAL_COLUMNS]
    result = result[result["text"].str.len() > 0].reset_index(drop=True)
    return result


def read_table(path: str | Path, *, usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read CSV, Excel, Parquet, or a ZIP containing CSV files."""
    path = Path(path)
    suffix = path.suffix.lower()
    usecols_set = set(usecols or [])
    usecols_arg = (lambda column: column in usecols_set) if usecols_set else None

    if suffix == ".parquet":
        return pd.read_parquet(path, columns=list(usecols_set) if usecols_set else None)
    if suffix in {".xlsx", ".xls"}:
        excel_usecols = list(usecols_set) if usecols_set else None
        try:
            frame = pd.read_excel(path, dtype=str, usecols=excel_usecols)
        except ValueError:
            frame = pd.DataFrame()
        if usecols_set and not set(frame.columns).intersection(usecols_set):
            preview = pd.read_excel(path, dtype=str, header=None, nrows=30)
            header_row = None
            for idx, row in preview.iterrows():
                values = {str(value).strip() for value in row.dropna().tolist()}
                if len(values.intersection(usecols_set)) >= 2:
                    header_row = idx
                    break
            if header_row is not None:
                frame = pd.read_excel(path, dtype=str, header=header_row, usecols=usecols_arg)
        return frame
    if suffix == ".zip":
        frames = []
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                with archive.open(name) as csv_file:
                    frames.append(pd.read_csv(csv_file, dtype=str, encoding="utf-8-sig", usecols=usecols_arg))
        if not frames:
            raise ValueError(f"No CSV files found inside {path}")
        return pd.concat(frames, ignore_index=True)

    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding, usecols=usecols_arg)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, encoding_errors="ignore", usecols=usecols_arg)


def source_usecols(source: str) -> Optional[list[str]]:
    if source == "sbiz":
        return [
            "상호명",
            "상권업종소분류코드",
            "상권업종소분류명",
            "표준산업분류코드",
            "표준산업분류명",
        ]
    if source in {"nara", "license"}:
        return [
            "업체명",
            "업종명",
            "대표업종",
            "업종코드",
            "공급물품명",
            "물품명",
            "품목명",
            "대표품명",
            "대표세부품명",
            "세부품명",
            "세부품명(명칭)",
            "물품분류명",
        ]
    return None
