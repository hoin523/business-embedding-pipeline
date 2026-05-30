import pandas as pd

from bizembed.pairs import generate_pairs


def test_generate_pairs_creates_same_industry_positive():
    records = pd.DataFrame(
        [
            {
                "entity_name": "보나비",
                "entity_type": "merchant",
                "industry_name": "일반한식",
                "industry_code": "I20101",
                "item_name": "",
                "source": "sbiz",
                "text": "가맹점명: 보나비 | 업종명: 일반한식",
            },
            {
                "entity_name": "본우리집밥",
                "entity_type": "merchant",
                "industry_name": "일반한식",
                "industry_code": "I20101",
                "item_name": "",
                "source": "sbiz",
                "text": "가맹점명: 본우리집밥 | 업종명: 일반한식",
            },
        ]
    )

    pairs = generate_pairs(records)

    assert list(pairs["relation"]) == ["same_industry"]
    assert pairs.loc[0, "label"] == 0.85


def test_generate_pairs_creates_same_item_positive_when_industry_is_missing():
    records = pd.DataFrame(
        [
            {
                "entity_name": "경원농자재",
                "entity_type": "supplier",
                "industry_name": "",
                "industry_code": "",
                "item_name": "온실설치공사",
                "source": "nara",
                "text": "공급업체명: 경원농자재 | 품목명: 온실설치공사",
            },
            {
                "entity_name": "그린온실",
                "entity_type": "supplier",
                "industry_name": "",
                "industry_code": "",
                "item_name": "온실설치공사",
                "source": "nara",
                "text": "공급업체명: 그린온실 | 품목명: 온실설치공사",
            },
        ]
    )

    pairs = generate_pairs(records)

    assert list(pairs["relation"]) == ["same_item"]
    assert pairs.loc[0, "label"] == 0.85


def test_generate_pairs_creates_hard_negative_for_same_name_different_industry():
    records = pd.DataFrame(
        [
            {
                "entity_name": "대성",
                "entity_type": "supplier",
                "industry_name": "식자재 도매업",
                "industry_code": "G463",
                "item_name": "급식재료",
                "source": "nara",
                "text": "공급업체명: 대성 | 업종명: 식자재 도매업 | 품목명: 급식재료",
            },
            {
                "entity_name": "대성",
                "entity_type": "supplier",
                "industry_name": "전기공사업",
                "industry_code": "F423",
                "item_name": "전기공사",
                "source": "nara",
                "text": "공급업체명: 대성 | 업종명: 전기공사업 | 품목명: 전기공사",
            },
        ]
    )

    pairs = generate_pairs(records)

    assert list(pairs["relation"]) == ["same_name_different_industry"]
    assert pairs.loc[0, "label"] == 0.2


def test_generate_pairs_applies_max_pairs_per_group_to_each_industry_group():
    records = pd.DataFrame(
        [
            {
                "entity_name": f"한식{i}",
                "entity_type": "merchant",
                "industry_name": "일반한식",
                "industry_code": "I20101",
                "item_name": "",
                "source": "sbiz",
                "text": f"가맹점명: 한식{i} | 업종명: 일반한식",
            }
            for i in range(4)
        ]
        + [
            {
                "entity_name": f"중식{i}",
                "entity_type": "merchant",
                "industry_name": "중식",
                "industry_code": "I20201",
                "item_name": "",
                "source": "sbiz",
                "text": f"가맹점명: 중식{i} | 업종명: 중식",
            }
            for i in range(4)
        ]
    )

    pairs = generate_pairs(records, max_pairs_per_group=2)

    positives = pairs[pairs["relation"] == "same_industry"]
    assert len(positives) == 4
