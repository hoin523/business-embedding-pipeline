from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List

import pandas as pd

from bizembed.ksic import format_ksic_semantic_text
from bizembed.pairs import PAIR_COLUMNS
from bizembed.semantic_pairs import DEFAULT_CONTEXTS


def _pair(text_a: str, text_b: str, label: float, relation: str) -> Dict[str, object]:
    return {"text_a": text_a, "text_b": text_b, "label": label, "relation": relation}


def _entity_name(industry_name: str, index: int) -> str:
    compact = str(industry_name).replace(" ", "")
    return f"{compact}사업체{index + 1}"


def build_ksic_semantic_pairs(
    taxonomy: pd.DataFrame,
    *,
    contexts: Iterable[str] = DEFAULT_CONTEXTS,
    max_sibling_pairs_per_group: int = 4,
    max_cousin_pairs_per_group: int = 4,
    max_negative_pairs: int = 4000,
) -> pd.DataFrame:
    """Generate weak-supervised similarity pairs from the real KSIC hierarchy.

    Labels encode hierarchy distance:
    - 0.90: same 5-digit KSIC industry across different entities.
    - 0.72: sibling 5-digit industries under the same 4-digit class.
    - 0.55: related industries under the same 3-digit group but different 4-digit class.
    - 0.08: industries from different KSIC large sections.
    """
    if taxonomy.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    context_list = tuple(contexts)
    rows = taxonomy.drop_duplicates("ksic5_code").reset_index(drop=True)
    pairs: List[Dict[str, object]] = []

    for idx, row in rows.iterrows():
        context = context_list[idx % len(context_list)]
        pairs.append(
            _pair(
                format_ksic_semantic_text(entity_name=_entity_name(row["ksic5_name"], 0), row=row, context=context),
                format_ksic_semantic_text(entity_name=_entity_name(row["ksic5_name"], 1), row=row, context=context),
                0.90,
                "same_ksic5_different_entity",
            )
        )

    for _, group in rows.groupby("ksic4_code"):
        count = 0
        for left, right in combinations(group.head(max_sibling_pairs_per_group + 1).to_dict("records"), 2):
            if left["ksic5_code"] == right["ksic5_code"]:
                continue
            context = context_list[count % len(context_list)]
            pairs.append(
                _pair(
                    format_ksic_semantic_text(entity_name=_entity_name(left["ksic5_name"], 0), row=pd.Series(left), context=context),
                    format_ksic_semantic_text(entity_name=_entity_name(right["ksic5_name"], 0), row=pd.Series(right), context=context),
                    0.72,
                    "same_ksic4_sibling_ksic5",
                )
            )
            count += 1
            if count >= max_sibling_pairs_per_group:
                break

    for _, group in rows.groupby("ksic3_code"):
        unique_classes = group.drop_duplicates("ksic4_code").head(max_cousin_pairs_per_group + 1)
        count = 0
        for left, right in combinations(unique_classes.to_dict("records"), 2):
            if left["ksic4_code"] == right["ksic4_code"]:
                continue
            context = context_list[count % len(context_list)]
            pairs.append(
                _pair(
                    format_ksic_semantic_text(entity_name=_entity_name(left["ksic5_name"], 0), row=pd.Series(left), context=context),
                    format_ksic_semantic_text(entity_name=_entity_name(right["ksic5_name"], 0), row=pd.Series(right), context=context),
                    0.55,
                    "same_ksic3_different_ksic4",
                )
            )
            count += 1
            if count >= max_cousin_pairs_per_group:
                break

    negatives = 0
    large_groups = [group.head(25).to_dict("records") for _, group in rows.groupby("ksic1_code")]
    for left_group, right_group in combinations(large_groups, 2):
        for left, right in zip(left_group, right_group):
            context = context_list[negatives % len(context_list)]
            pairs.append(
                _pair(
                    format_ksic_semantic_text(entity_name=_entity_name(left["ksic5_name"], 0), row=pd.Series(left), context=context),
                    format_ksic_semantic_text(entity_name=_entity_name(right["ksic5_name"], 0), row=pd.Series(right), context=context),
                    0.08,
                    "different_ksic1",
                )
            )
            negatives += 1
            if negatives >= max_negative_pairs:
                return pd.DataFrame(pairs, columns=PAIR_COLUMNS).drop_duplicates().reset_index(drop=True)

    return pd.DataFrame(pairs, columns=PAIR_COLUMNS).drop_duplicates().reset_index(drop=True)
