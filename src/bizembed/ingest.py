from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from bizembed.normalize import format_record_text, normalize_text


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
        "세부품명",
        "물품분류명",
        "상품명",
    ),
}


def first_present_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def standardize_dataframe(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Map public-data columns into the canonical embedding-record schema."""
    result = pd.DataFrame(index=df.index)

    for target, aliases in COLUMN_ALIASES.items():
        source_column = first_present_column(df, aliases)
        if source_column is None:
            result[target] = ""
        else:
            result[target] = df[source_column].map(normalize_text)

    entity_type = SOURCE_ENTITY_TYPES.get(source, "company")
    result["entity_type"] = entity_type
    result["source"] = source

    result["text"] = result.apply(
        lambda row: format_record_text(
            entity_name=row["entity_name"],
            entity_type=row["entity_type"],
            industry_name=row["industry_name"],
            item_name=row["item_name"],
        ),
        axis=1,
    )

    result = result[CANONICAL_COLUMNS]
    result = result[result["text"].str.len() > 0].reset_index(drop=True)
    return result


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV, Excel, Parquet, or a ZIP containing CSV files."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".zip":
        return pd.read_csv(path, compression="zip", dtype=str, encoding_errors="ignore")

    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, encoding_errors="ignore")
