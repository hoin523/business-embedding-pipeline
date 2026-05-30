import pandas as pd

from bizembed.compact_pairs import (
    build_compact_hard_pairs,
    build_compact_industry_pairs,
    build_compact_ksic_pairs,
    compact_card_text,
)


def test_compact_card_text_uses_card_like_order_without_verbose_labels():
    text = compact_card_text(
        entity_name="보나비",
        industry_name="일반한식",
        context="내부회계관리팀 천호랄 출근 국내 평일",
    )

    assert text == "보나비 일반한식 내부회계관리팀 천호랄 출근 국내 평일"
    assert "표준업종명" not in text
    assert "|" not in text


def test_build_compact_industry_pairs_contains_hansik_alias_positive():
    pairs = build_compact_industry_pairs(contexts=("동일 문맥",), entities_per_concept=1)

    row = pairs[pairs["relation"].eq("compact_same_entity_same_canonical_industry")].iloc[0]

    assert row["label"] >= 0.9
    assert "가맹점명:" not in row["text_a"]
    assert "업종명:" not in row["text_a"]


def test_build_compact_ksic_pairs_uses_real_industry_name_only():
    taxonomy = pd.DataFrame(
        [
            {
                "ksic1_code": "I",
                "ksic1_name": "숙박 및 음식점업",
                "ksic2_code": "I56",
                "ksic2_name": "음식점 및 주점업",
                "ksic3_code": "I561",
                "ksic3_name": "음식점업",
                "ksic4_code": "I5611",
                "ksic4_name": "한식 음식점업",
                "ksic5_code": "I56111",
                "ksic5_name": "한식 일반 음식점업",
            },
            {
                "ksic1_code": "I",
                "ksic1_name": "숙박 및 음식점업",
                "ksic2_code": "I56",
                "ksic2_name": "음식점 및 주점업",
                "ksic3_code": "I561",
                "ksic3_name": "음식점업",
                "ksic4_code": "I5611",
                "ksic4_name": "한식 음식점업",
                "ksic5_code": "I56112",
                "ksic5_name": "한식 면 음식점업",
            },
        ]
    )

    pairs = build_compact_ksic_pairs(taxonomy, contexts=("동일 문맥",), max_negative_pairs=0)

    assert pairs["text_a"].str.contains("표준업종명", regex=False).sum() == 0
    assert pairs["text_a"].str.contains("한식 일반 음식점업", regex=False).any()


def test_build_compact_hard_pairs_contains_same_context_negative():
    pairs = build_compact_hard_pairs(contexts=("동일 문맥",))

    row = pairs[pairs["relation"].eq("compact_hard_different_industry_same_context")].iloc[0]

    assert row["label"] == 0.02
    assert row["text_a"].endswith("동일 문맥")
    assert row["text_b"].endswith("동일 문맥")
