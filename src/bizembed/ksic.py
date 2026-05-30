from __future__ import annotations

import re
from typing import List

import pandas as pd


KSIC_LEVELS = (1, 2, 3, 4, 5)
KSIC_TAXONOMY_COLUMNS = [
    "ksic1_code",
    "ksic1_name",
    "ksic2_code",
    "ksic2_name",
    "ksic3_code",
    "ksic3_name",
    "ksic4_code",
    "ksic4_name",
    "ksic5_code",
    "ksic5_name",
    "industry_code",
    "industry_name",
    "standard_industry_name",
    "parent_industry_name",
    "large_industry_name",
    "path",
]


def clean_ksic_name(value: str) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()


def normalize_industry_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip()).lower()


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for level in KSIC_LEVELS:
        name_col = f"ksic{level}_nm"
        code_col = f"ksic{level}_cd"
        work[name_col] = work[name_col].map(clean_ksic_name)
        work[code_col] = work[code_col].astype(str).str.strip()
    return work


def build_ksic11_taxonomy(tree: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized 11th-revision KSIC hierarchy from ksicTreeDB rows."""
    if tree.empty:
        return pd.DataFrame(columns=KSIC_TAXONOMY_COLUMNS)

    work = tree[tree["ksic_C"].eq("C11")].copy()
    work = _clean_columns(work)
    work = work.drop_duplicates("ksic5_cd").sort_values("ksic5_cd")

    result = pd.DataFrame(
        {
            "ksic1_code": work["ksic1_cd"],
            "ksic1_name": work["ksic1_nm"],
            "ksic2_code": work["ksic2_cd"],
            "ksic2_name": work["ksic2_nm"],
            "ksic3_code": work["ksic3_cd"],
            "ksic3_name": work["ksic3_nm"],
            "ksic4_code": work["ksic4_cd"],
            "ksic4_name": work["ksic4_nm"],
            "ksic5_code": work["ksic5_cd"],
            "ksic5_name": work["ksic5_nm"],
        }
    )
    result["industry_code"] = result["ksic5_code"]
    result["industry_name"] = result["ksic5_name"]
    result["standard_industry_name"] = result["ksic5_name"]
    result["parent_industry_name"] = result["ksic3_name"]
    result["large_industry_name"] = result["ksic1_name"]
    result["path"] = result.apply(ksic_path, axis=1)
    return result[KSIC_TAXONOMY_COLUMNS].reset_index(drop=True)


def ksic_path(row: pd.Series) -> str:
    names: List[str] = [
        str(row.get("ksic1_name", "")).strip(),
        str(row.get("ksic2_name", "")).strip(),
        str(row.get("ksic3_name", "")).strip(),
        str(row.get("ksic4_name", "")).strip(),
        str(row.get("ksic5_name", "")).strip(),
    ]
    return " > ".join(name for name in names if name)


def format_ksic_semantic_text(
    *,
    entity_name: str,
    row: pd.Series,
    context: str = "",
    entity_label: str = "가맹점명",
) -> str:
    segments: List[str] = [
        f"{entity_label}: {str(entity_name).strip()}",
        f"표준업종명: {str(row['ksic5_name']).strip()}",
        f"세분류명: {str(row['ksic4_name']).strip()}",
        f"소분류명: {str(row['ksic3_name']).strip()}",
        f"중분류명: {str(row['ksic2_name']).strip()}",
        f"대분류명: {str(row['ksic1_name']).strip()}",
    ]
    if str(context).strip():
        segments.append(f"문맥: {str(context).strip()}")
    return " | ".join(segments)
