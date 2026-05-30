import pandas as pd

from bizembed.pair_filter import filter_industry_training_pairs, strip_labeled_segments


def test_strip_labeled_segments_removes_item_name_from_training_text():
    text = "공급업체명: 한국식품유통 | 업종명: 식자재 도매업 | 품목명: 급식재료"

    assert strip_labeled_segments(text, labels=("품목명:",)) == "공급업체명: 한국식품유통 | 업종명: 식자재 도매업"


def test_filter_industry_training_pairs_drops_item_only_pairs_and_strips_items():
    pairs = pd.DataFrame(
        [
            {
                "text_a": "공급업체명: 수호상사 | 품목명: 융복합안전조끼",
                "text_b": "공급업체명: 매드독캠프 | 품목명: 융복합안전조끼",
                "label": 0.85,
                "relation": "same_item",
            },
            {
                "text_a": "공급업체명: 한국식품유통 | 업종명: 식자재 도매업 | 품목명: 급식재료",
                "text_b": "공급업체명: 대한식자재 | 업종명: 식자재 도매업 | 품목명: 식자재",
                "label": 0.85,
                "relation": "same_industry",
            },
        ]
    )

    result = filter_industry_training_pairs(pairs)

    assert len(result) == 1
    assert result.loc[0, "text_a"] == "공급업체명: 한국식품유통 | 업종명: 식자재 도매업"
    assert "품목명:" not in result.loc[0, "text_b"]
