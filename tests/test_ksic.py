import pandas as pd

from bizembed.ksic import build_ksic11_taxonomy, clean_ksic_name, format_ksic_semantic_text


def test_clean_ksic_name_removes_code_range_suffix():
    assert clean_ksic_name("숙박 및 음식점업(55~56)") == "숙박 및 음식점업"


def test_build_ksic11_taxonomy_keeps_real_hierarchy_columns():
    tree = pd.DataFrame(
        [
            {
                "ksic1_cd": "I",
                "ksic1_nm": "숙박 및 음식점업(55~56)",
                "ksic2_cd": "56",
                "ksic2_nm": "음식점 및 주점업",
                "ksic3_cd": "561",
                "ksic3_nm": "음식점업",
                "ksic4_cd": "5611",
                "ksic4_nm": "한식 음식점업",
                "ksic5_cd": "56111",
                "ksic5_nm": "한식 일반 음식점업",
                "ksic_C": "C11",
            }
        ]
    )

    result = build_ksic11_taxonomy(tree)

    assert result.loc[0, "industry_code"] == "56111"
    assert result.loc[0, "standard_industry_name"] == "한식 일반 음식점업"
    assert result.loc[0, "parent_industry_name"] == "음식점업"
    assert result.loc[0, "large_industry_name"] == "숙박 및 음식점업"
    assert result.loc[0, "path"] == "숙박 및 음식점업 > 음식점 및 주점업 > 음식점업 > 한식 음식점업 > 한식 일반 음식점업"


def test_format_ksic_semantic_text_uses_all_industry_levels():
    row = pd.Series(
        {
            "ksic1_name": "숙박 및 음식점업",
            "ksic2_name": "음식점 및 주점업",
            "ksic3_name": "음식점업",
            "ksic4_name": "한식 음식점업",
            "ksic5_name": "한식 일반 음식점업",
        }
    )

    text = format_ksic_semantic_text(entity_name="보나비", row=row, context="법인카드 점심")

    assert "표준업종명: 한식 일반 음식점업" in text
    assert "대분류명: 숙박 및 음식점업" in text
    assert "문맥: 법인카드 점심" in text
