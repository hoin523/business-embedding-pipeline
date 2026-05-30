from __future__ import annotations

from itertools import combinations
import re
from typing import Dict, Iterable, List

import pandas as pd

from bizembed.industry_taxonomy import INDUSTRY_CONCEPTS, IndustryConcept
from bizembed.pairs import PAIR_COLUMNS
from bizembed.semantic_pairs import ENTITY_NAMES


COMPACT_CONTEXTS = (
    "내부회계관리팀 천호랄 출근 국내 평일",
    "재무팀 월말 결산 법인카드",
    "경영지원팀 정기 감사 비용",
    "영업본부 고객사 방문 국내 평일",
    "총무팀 사무실 운영 경비",
    "구매팀 정기 계약 비용 정산",
    "현장 점검 일정 법인카드 결제",
    "워크숍 준비 참석자 지원 비용",
)

HARD_NEGATIVE_CONTEXTS = (
    "내부회계관리팀 천호랄 출근 국내 평일",
    "재무팀 월말 결산 법인카드",
    "총무팀 사무실 운영 경비",
    "출장 차량 유류비",
    "부서 공통 경비",
)

HARD_NEGATIVE_PAIRS = (
    (("보나비", "일반한식"), ("대한항공", "항공사")),
    (("본우리집밥", "한식음식점"), ("제주항공", "항공권")),
    (("스타벅스", "커피전문점"), ("SK주유소", "주유소")),
    (("이디야커피", "카페"), ("GS칼텍스", "유류판매")),
    (("이마트", "대형마트"), ("대한항공", "항공사")),
    (("CU", "편의점"), ("세브란스병원", "종합병원")),
    (("더존비즈온", "소프트웨어 개발업"), ("SK주유소", "주유소")),
    (("온누리약국", "약국"), ("파리바게뜨", "베이커리")),
    (("신라스테이", "호텔"), ("스타벅스", "커피전문점")),
    (("대성전기", "전기공사업"), ("맛나식당", "일반음식점")),
)

HARD_MEDIUM_PAIRS = (
    (("스타벅스", "커피전문점"), ("파리바게뜨", "베이커리"), 0.55),
    (("이마트", "대형마트"), ("CU", "편의점"), 0.58),
    (("세브란스병원", "종합병원"), ("온누리약국", "약국"), 0.62),
    (("더존비즈온", "소프트웨어 개발업"), ("안랩", "정보보안"), 0.70),
    (("보나비", "일반한식"), ("맛나식당", "일반음식점"), 0.72),
)


def _clean(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[|:]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_card_text(
    *,
    entity_name: object,
    industry_name: object,
    context: object = "",
    entity_label: str = "가맹점명",
    include_field_tokens: bool = False,
) -> str:
    """Format text close to card input: merchant/supplier, industry, then memo tail."""
    entity = _clean(entity_name)
    industry = _clean(industry_name)
    tail = _clean(context)

    if include_field_tokens:
        parts = [f"{entity_label} {entity}", f"업종명 {industry}"]
    else:
        parts = [entity, industry]
    if tail:
        parts.append(tail)
    return " ".join(part for part in parts if part)


def _pair(text_a: str, text_b: str, label: float, relation: str) -> Dict[str, object]:
    return {"text_a": text_a, "text_b": text_b, "label": label, "relation": relation}


def _entity_names(concept: IndustryConcept, count: int) -> tuple[str, ...]:
    names = ENTITY_NAMES.get(concept.canonical, ())
    if names:
        return names[:count]
    return tuple(f"{concept.canonical}{idx + 1}" for idx in range(count))


def build_compact_industry_pairs(
    *,
    contexts: Iterable[str] = COMPACT_CONTEXTS,
    entities_per_concept: int = 5,
    include_field_token_variants: bool = True,
) -> pd.DataFrame:
    context_list = tuple(contexts)
    pairs: List[Dict[str, object]] = []

    def format_text(
        *,
        entity_name: object,
        industry_name: object,
        context: object,
        include_field_tokens: bool = False,
    ) -> str:
        return compact_card_text(
            entity_name=entity_name,
            industry_name=industry_name,
            context=context,
            include_field_tokens=include_field_tokens,
        )

    concept_rows = []
    for concept in INDUSTRY_CONCEPTS:
        aliases = (concept.canonical,) + concept.aliases
        names = _entity_names(concept, entities_per_concept)
        for name_index, entity_name in enumerate(names):
            concept_rows.append((concept, entity_name, context_list[name_index % len(context_list)], aliases))

            for context in context_list:
                for left_alias, right_alias in combinations(aliases[: min(len(aliases), 5)], 2):
                    pairs.append(
                        _pair(
                            format_text(entity_name=entity_name, industry_name=left_alias, context=context),
                            format_text(entity_name=entity_name, industry_name=right_alias, context=context),
                            0.96,
                            "compact_same_entity_same_canonical_industry",
                        )
                    )
                    if include_field_token_variants and context == context_list[0]:
                        pairs.append(
                            _pair(
                                format_text(
                                    entity_name=entity_name,
                                    industry_name=left_alias,
                                    context=context,
                                    include_field_tokens=True,
                                ),
                                format_text(
                                    entity_name=entity_name,
                                    industry_name=right_alias,
                                    context=context,
                                    include_field_tokens=True,
                                ),
                                0.96,
                                "compact_labeled_tokens_same_entity_same_canonical_industry",
                            )
                        )

        for context in context_list:
            for left_name, right_name in combinations(names[: min(len(names), 4)], 2):
                pairs.append(
                    _pair(
                        format_text(entity_name=left_name, industry_name=aliases[0], context=context),
                        format_text(entity_name=right_name, industry_name=aliases[1], context=context),
                        0.88,
                        "compact_different_entity_same_canonical_industry",
                    )
                )

    for left, right in combinations(concept_rows, 2):
        left_concept, left_name, left_context, left_aliases = left
        right_concept, right_name, _, right_aliases = right
        if left_context != context_list[0]:
            continue

        if left_concept.parent == right_concept.parent and left_concept.canonical != right_concept.canonical:
            label = 0.55
            relation = "compact_same_parent_different_canonical_industry"
        elif left_concept.parent != right_concept.parent:
            label = 0.04
            relation = "compact_different_parent_industry_same_context"
        else:
            continue

        pairs.append(
            _pair(
                format_text(entity_name=left_name, industry_name=left_aliases[0], context=left_context),
                format_text(entity_name=right_name, industry_name=right_aliases[0], context=left_context),
                label,
                relation,
            )
        )

    return pd.DataFrame(pairs, columns=PAIR_COLUMNS).drop_duplicates().reset_index(drop=True)


def build_compact_hard_pairs(
    *,
    contexts: Iterable[str] = HARD_NEGATIVE_CONTEXTS,
) -> pd.DataFrame:
    context_list = tuple(contexts)
    pairs: List[Dict[str, object]] = []

    for context in context_list:
        for (left_entity, left_industry), (right_entity, right_industry) in HARD_NEGATIVE_PAIRS:
            pairs.append(
                _pair(
                    compact_card_text(entity_name=left_entity, industry_name=left_industry, context=context),
                    compact_card_text(entity_name=right_entity, industry_name=right_industry, context=context),
                    0.02,
                    "compact_hard_different_industry_same_context",
                )
            )
            pairs.append(
                _pair(
                    compact_card_text(entity_name=left_entity, industry_name=left_industry, context=context),
                    compact_card_text(entity_name=left_entity, industry_name=right_industry, context=context),
                    0.02,
                    "compact_same_entity_conflicting_industry",
                )
            )

        for (left_entity, left_industry), (right_entity, right_industry), label in HARD_MEDIUM_PAIRS:
            pairs.append(
                _pair(
                    compact_card_text(entity_name=left_entity, industry_name=left_industry, context=context),
                    compact_card_text(entity_name=right_entity, industry_name=right_industry, context=context),
                    label,
                    "compact_hard_related_industry_same_context",
                )
            )

    return pd.DataFrame(pairs, columns=PAIR_COLUMNS).drop_duplicates().reset_index(drop=True)


def _ksic_entity_name(industry_name: object, index: int) -> str:
    compact = re.sub(r"\s+", "", str(industry_name or "").strip())
    return f"{compact}사업체{index + 1}"


def build_compact_ksic_pairs(
    taxonomy: pd.DataFrame,
    *,
    contexts: Iterable[str] = COMPACT_CONTEXTS,
    max_sibling_pairs_per_group: int = 4,
    max_cousin_pairs_per_group: int = 4,
    max_negative_pairs: int = 4000,
) -> pd.DataFrame:
    if taxonomy.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    context_list = tuple(contexts)
    rows = taxonomy.drop_duplicates("ksic5_code").reset_index(drop=True)
    pairs: List[Dict[str, object]] = []

    def format_row(row: pd.Series | dict, *, entity_index: int, context: str) -> str:
        return compact_card_text(
            entity_name=_ksic_entity_name(row["ksic5_name"], entity_index),
            industry_name=row["ksic5_name"],
            context=context,
        )

    for idx, row in rows.iterrows():
        context = context_list[idx % len(context_list)]
        pairs.append(
            _pair(
                format_row(row, entity_index=0, context=context),
                format_row(row, entity_index=1, context=context),
                0.90,
                "compact_same_ksic5_different_entity",
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
                    format_row(left, entity_index=0, context=context),
                    format_row(right, entity_index=0, context=context),
                    0.70,
                    "compact_same_ksic4_sibling_ksic5",
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
                    format_row(left, entity_index=0, context=context),
                    format_row(right, entity_index=0, context=context),
                    0.50,
                    "compact_same_ksic3_different_ksic4",
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
                    format_row(left, entity_index=0, context=context),
                    format_row(right, entity_index=0, context=context),
                    0.04,
                    "compact_different_ksic1",
                )
            )
            negatives += 1
            if negatives >= max_negative_pairs:
                return pd.DataFrame(pairs, columns=PAIR_COLUMNS).drop_duplicates().reset_index(drop=True)

    return pd.DataFrame(pairs, columns=PAIR_COLUMNS).drop_duplicates().reset_index(drop=True)
