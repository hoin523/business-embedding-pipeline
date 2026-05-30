from __future__ import annotations

from typing import Iterable

import pandas as pd


def strip_labeled_segments(text: str, *, labels: Iterable[str]) -> str:
    blocked = tuple(labels)
    parts = []
    for raw_segment in str(text).split("|"):
        segment = raw_segment.strip()
        if not segment:
            continue
        if segment.startswith(blocked):
            continue
        parts.append(segment)
    return " | ".join(parts)


def filter_industry_training_pairs(
    pairs: pd.DataFrame,
    *,
    require_label: str = "업종명:",
    strip_labels: tuple[str, ...] = ("품목명:",),
) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()

    result = pairs.copy()
    result["text_a"] = result["text_a"].map(lambda value: strip_labeled_segments(value, labels=strip_labels))
    result["text_b"] = result["text_b"].map(lambda value: strip_labeled_segments(value, labels=strip_labels))

    has_required = result["text_a"].str.contains(require_label, regex=False) & result["text_b"].str.contains(
        require_label,
        regex=False,
    )
    result = result[has_required]
    result = result[result["text_a"].str.len().gt(0) & result["text_b"].str.len().gt(0)]
    return result.drop_duplicates().reset_index(drop=True)
