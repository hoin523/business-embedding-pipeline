from __future__ import annotations

from itertools import cycle
from typing import Iterable, List

import pandas as pd


UNLABELED_TAIL_LABELS = ("품목명:", "상품명:", "사용목적:", "적요:", "메모:")

GENERIC_CONTEXT_TAILS = (
    "내부회계관리팀 출근 국내 평일",
    "총무팀 정기구매",
    "임직원 회의 비용",
    "출장 관련 사용",
    "부서 공통 경비",
)

FOOD_CONTEXT_TAILS = (
    "내부회계관리팀 출근 국내 평일",
    "임직원 식대 회의",
    "야근 식사 비용",
    "급식재료 정기구매",
)

TRAVEL_CONTEXT_TAILS = (
    "국내 출장 예약",
    "출장 항공권 국내선",
    "임직원 출장 교통비",
)

LODGING_CONTEXT_TAILS = (
    "출장 숙박 예약",
    "임직원 국내 출장 숙박",
    "현장 방문 숙박비",
)

OFFICE_CONTEXT_TAILS = (
    "총무팀 사무용품 구매",
    "부서 운영 소모품",
    "문구 비품 정기구매",
)

FACILITY_CONTEXT_TAILS = (
    "시설관리팀 유지보수",
    "현장 공사 자재",
    "전기 설비 보수",
)


def _strip_known_label(segment: str) -> str:
    text = segment.strip()
    for label in UNLABELED_TAIL_LABELS:
        if text.startswith(label):
            return text[len(label) :].strip()
    return text


def _tail_pool(text: str) -> Iterable[str]:
    lower = text.lower()
    if any(keyword in text for keyword in ("한식", "식자재", "급식", "음식", "식당", "케이터링")):
        return FOOD_CONTEXT_TAILS
    if any(keyword in text for keyword in ("항공", "항공권", "공항")):
        return TRAVEL_CONTEXT_TAILS
    if any(keyword in text for keyword in ("호텔", "숙박", "펜션", "모텔")):
        return LODGING_CONTEXT_TAILS
    if any(keyword in text for keyword in ("문구", "사무", "비품", "용품")):
        return OFFICE_CONTEXT_TAILS
    if any(keyword in text for keyword in ("전기", "공사", "배전반", "설비")):
        return FACILITY_CONTEXT_TAILS
    if "mcc" in lower:
        return GENERIC_CONTEXT_TAILS
    return GENERIC_CONTEXT_TAILS


def context_tail_for_text(text: str, *, offset: int = 0) -> str:
    pool = list(_tail_pool(text))
    return pool[offset % len(pool)]


def hybridize_labeled_text(text: str, *, tail: str = "") -> str:
    """Keep entity/industry labels up front and append item/memo text without labels."""
    labeled: List[str] = []
    unlabeled: List[str] = []

    for raw_segment in str(text).split("|"):
        segment = raw_segment.strip()
        if not segment:
            continue
        if segment.startswith(("가맹점명:", "공급업체명:", "업체명:", "업종명:")):
            labeled.append(segment)
        else:
            stripped = _strip_known_label(segment)
            if stripped:
                unlabeled.append(stripped)

    if tail:
        unlabeled.append(tail.strip())

    parts = labeled[:]
    if unlabeled:
        parts.append(" ".join(value for value in unlabeled if value))
    return " | ".join(parts)


def augment_pairs_with_context(
    pairs: pd.DataFrame,
    *,
    variants_per_pair: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Create card-like context variants: labeled head fields plus unlabeled memo tail."""
    if pairs.empty:
        return pairs.copy()

    rows = []
    work = pairs.reset_index(drop=True)
    variant_offsets = list(range(seed, seed + variants_per_pair))
    for idx, row in work.iterrows():
        for offset in variant_offsets:
            text_a = str(row["text_a"])
            text_b = str(row["text_b"])
            rows.append(
                {
                    "text_a": hybridize_labeled_text(
                        text_a,
                        tail=context_tail_for_text(text_a, offset=idx + offset),
                    ),
                    "text_b": hybridize_labeled_text(
                        text_b,
                        tail=context_tail_for_text(text_b, offset=idx + offset + 1),
                    ),
                    "label": float(row["label"]),
                    "relation": f"{row['relation']}_context",
                }
            )
    return pd.DataFrame(rows, columns=list(pairs.columns)).drop_duplicates().reset_index(drop=True)
