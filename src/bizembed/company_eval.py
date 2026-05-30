from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from bizembed.compact_pairs import compact_card_text
from bizembed.ingest import COLUMN_ALIASES, first_present_column, normalize_series, read_table


CONTEXT_ALIASES = (
    "기타텍스트",
    "문맥",
    "적요",
    "메모",
    "사용내역",
    "집행목적",
    "사용목적",
    "부서",
    "사용부서",
    "부서명",
    "품목명",
    "상품명",
)


@dataclass(frozen=True)
class ThresholdMetric:
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = normalize_series(result[column])
    return result


def _first_value(row: pd.Series, columns: Sequence[str]) -> str:
    for column in columns:
        if column in row.index and str(row[column]).strip():
            return str(row[column]).strip()
    return ""


def infer_context_columns(frame: pd.DataFrame, explicit_columns: Iterable[str] = ()) -> list[str]:
    explicit = [column for column in explicit_columns if column in frame.columns]
    inferred = [column for column in CONTEXT_ALIASES if column in frame.columns and column not in explicit]
    return explicit + inferred


def compact_records_from_frame(
    frame: pd.DataFrame,
    *,
    entity_column: str = "",
    industry_column: str = "",
    context_columns: Iterable[str] = (),
    include_field_tokens: bool = False,
) -> pd.DataFrame:
    work = _normalize_columns(frame)
    entity_source = entity_column or first_present_column(work, COLUMN_ALIASES["entity_name"])
    industry_source = industry_column or first_present_column(work, COLUMN_ALIASES["industry_name"])
    if entity_source is None and industry_source is None:
        raise ValueError("No entity or industry column found. Pass --entity-column and/or --industry-column.")

    context_sources = infer_context_columns(work, context_columns)
    rows = []
    for idx, row in work.iterrows():
        entity = str(row.get(entity_source, "")).strip() if entity_source else ""
        industry = str(row.get(industry_source, "")).strip() if industry_source else ""
        context = " ".join(_first_value(row, [column]) for column in context_sources).strip()
        text = compact_card_text(
            entity_name=entity,
            industry_name=industry,
            context=context,
            include_field_tokens=include_field_tokens,
        )
        if not text:
            continue
        rows.append(
            {
                "record_id": idx,
                "entity_name": entity,
                "industry_name": industry,
                "context": context,
                "compact_text": text,
            }
        )
    return pd.DataFrame(rows)


def compact_records_from_path(
    path: str,
    *,
    entity_column: str = "",
    industry_column: str = "",
    context_columns: Iterable[str] = (),
    include_field_tokens: bool = False,
) -> pd.DataFrame:
    return compact_records_from_frame(
        read_table(path),
        entity_column=entity_column,
        industry_column=industry_column,
        context_columns=context_columns,
        include_field_tokens=include_field_tokens,
    )


def normalize_binary_label(value: object) -> int | None:
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "similar", "same", "positive", "유사", "같음", "동일"}:
        return 1
    if text in {"0", "false", "no", "n", "different", "negative", "비유사", "다름", "상이"}:
        return 0
    try:
        numeric = float(text)
    except ValueError:
        return None
    return 1 if numeric >= 0.7 else 0


def threshold_metrics(scores: Sequence[float], labels: Sequence[int], *, step: float = 1.0) -> pd.DataFrame:
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    rows = []
    for threshold in np.arange(0, 100 + step, step):
        pred = score_array >= threshold
        truth = label_array == 1
        tp = int(np.logical_and(pred, truth).sum())
        fp = int(np.logical_and(pred, ~truth).sum())
        tn = int(np.logical_and(~pred, ~truth).sum())
        fn = int(np.logical_and(~pred, truth).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / len(label_array) if len(label_array) else 0.0
        rows.append(
            ThresholdMetric(
                threshold=float(threshold),
                precision=precision,
                recall=recall,
                f1=f1,
                accuracy=accuracy,
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
            ).__dict__
        )
    return pd.DataFrame(rows)


def best_threshold(metrics: pd.DataFrame) -> pd.Series:
    if metrics.empty:
        return pd.Series(dtype=object)
    return metrics.sort_values(["f1", "accuracy", "threshold"], ascending=[False, False, True]).iloc[0]
