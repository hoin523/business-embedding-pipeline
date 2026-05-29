from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import pandas as pd

from bizembed.normalize import normalize_entity_name


PAIR_COLUMNS = ["text_a", "text_b", "label", "relation"]


def _pair(text_a: str, text_b: str, label: float, relation: str) -> Dict[str, object]:
    return {
        "text_a": text_a,
        "text_b": text_b,
        "label": label,
        "relation": relation,
    }


def _industry_key(row: pd.Series) -> str:
    code = str(row.get("industry_code", "") or "").strip()
    name = str(row.get("industry_name", "") or "").strip()
    return code or name


def generate_pairs(records: pd.DataFrame, *, max_pairs_per_group: int = 50) -> pd.DataFrame:
    """Generate weak-supervised similarity pairs from canonical records.

    Labels are intentionally conservative:
    - 0.85: same detailed industry bucket across different entities.
    - 0.20: same normalized entity name but different industry, a hard negative.
    - 0.05: clearly different industry buckets.
    """
    if records.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    work = records.copy()
    work["industry_key"] = work.apply(_industry_key, axis=1)
    work["entity_key"] = work["entity_name"].map(normalize_entity_name)

    pairs: List[Dict[str, object]] = []

    for _, group in work[work["industry_key"].astype(bool)].groupby("industry_key"):
        unique = group.drop_duplicates("text").head(max_pairs_per_group + 1)
        for left, right in combinations(unique.to_dict("records"), 2):
            if left["entity_key"] == right["entity_key"]:
                continue
            pairs.append(_pair(left["text"], right["text"], 0.85, "same_industry"))
            if len(pairs) >= max_pairs_per_group:
                break

    for _, group in work[work["entity_key"].astype(bool)].groupby("entity_key"):
        unique = group.drop_duplicates("industry_key")
        for left, right in combinations(unique.to_dict("records"), 2):
            if left["industry_key"] == right["industry_key"]:
                continue
            pairs.append(_pair(left["text"], right["text"], 0.20, "same_name_different_industry"))

    if not pairs and len(work) >= 2:
        left = work.iloc[0]
        for _, right in work.iloc[1:].iterrows():
            if left["industry_key"] != right["industry_key"]:
                pairs.append(_pair(left["text"], right["text"], 0.05, "different_industry"))
                break

    return pd.DataFrame(pairs, columns=PAIR_COLUMNS).drop_duplicates().reset_index(drop=True)
