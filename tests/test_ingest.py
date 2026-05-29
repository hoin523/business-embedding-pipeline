import pandas as pd

from bizembed.ingest import standardize_dataframe


def test_standardize_sbiz_store_columns():
    raw = pd.DataFrame(
        [
            {
                "상호명": "보나비",
                "상권업종소분류명": "일반한식",
                "상권업종소분류코드": "I20101",
            }
        ]
    )

    result = standardize_dataframe(raw, source="sbiz")

    assert result.loc[0, "entity_name"] == "보나비"
    assert result.loc[0, "entity_type"] == "merchant"
    assert result.loc[0, "industry_name"] == "일반한식"
    assert result.loc[0, "industry_code"] == "I20101"
    assert result.loc[0, "source"] == "sbiz"
    assert result.loc[0, "text"] == "가맹점명: 보나비 | 업종명: 일반한식"


def test_standardize_supplier_columns_with_item_name():
    raw = pd.DataFrame(
        [
            {
                "업체명": "한국식품유통",
                "업종명": "식자재 도매업",
                "업종코드": "G463",
                "공급물품명": "급식재료",
            }
        ]
    )

    result = standardize_dataframe(raw, source="nara")

    assert result.loc[0, "entity_name"] == "한국식품유통"
    assert result.loc[0, "entity_type"] == "supplier"
    assert result.loc[0, "industry_name"] == "식자재 도매업"
    assert result.loc[0, "item_name"] == "급식재료"
    assert result.loc[0, "text"] == "공급업체명: 한국식품유통 | 업종명: 식자재 도매업 | 품목명: 급식재료"
