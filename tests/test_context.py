import pandas as pd

from bizembed.context import augment_pairs_with_context, hybridize_labeled_text


def test_hybridize_labeled_text_keeps_core_labels_and_unlabels_tail_fields():
    text = "공급업체명: 한국식품유통 | 업종명: 식자재 도매업 | 품목명: 급식재료"

    result = hybridize_labeled_text(text, tail="총무팀 정기구매")

    assert result == "공급업체명: 한국식품유통 | 업종명: 식자재 도매업 | 급식재료 총무팀 정기구매"


def test_augment_pairs_with_context_preserves_labels_and_marks_relation():
    pairs = pd.DataFrame(
        [
            {
                "text_a": "가맹점명: 보나비 | 업종명: 일반한식",
                "text_b": "가맹점명: 동해횟집 | 업종명: 일반한식",
                "label": 0.85,
                "relation": "same_industry",
            }
        ]
    )

    result = augment_pairs_with_context(pairs, variants_per_pair=2, seed=1)

    assert len(result) == 2
    assert result["label"].tolist() == [0.85, 0.85]
    assert result["relation"].tolist() == ["same_industry_context", "same_industry_context"]
    assert result["text_a"].str.startswith("가맹점명: 보나비 | 업종명: 일반한식 | ").all()
    assert "업종명:" in result.loc[0, "text_b"]
