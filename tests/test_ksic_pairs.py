import pandas as pd

from bizembed.ksic_pairs import build_ksic_semantic_pairs


def _taxonomy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ksic1_code": "I",
                "ksic1_name": "숙박 및 음식점업",
                "ksic2_code": "56",
                "ksic2_name": "음식점 및 주점업",
                "ksic3_code": "561",
                "ksic3_name": "음식점업",
                "ksic4_code": "5611",
                "ksic4_name": "한식 음식점업",
                "ksic5_code": "56111",
                "ksic5_name": "한식 일반 음식점업",
            },
            {
                "ksic1_code": "I",
                "ksic1_name": "숙박 및 음식점업",
                "ksic2_code": "56",
                "ksic2_name": "음식점 및 주점업",
                "ksic3_code": "561",
                "ksic3_name": "음식점업",
                "ksic4_code": "5611",
                "ksic4_name": "한식 음식점업",
                "ksic5_code": "56112",
                "ksic5_name": "한식 면 요리 전문점",
            },
            {
                "ksic1_code": "H",
                "ksic1_name": "운수 및 창고업",
                "ksic2_code": "51",
                "ksic2_name": "항공 운송업",
                "ksic3_code": "511",
                "ksic3_name": "항공 여객 운송업",
                "ksic4_code": "5110",
                "ksic4_name": "항공 여객 운송업",
                "ksic5_code": "51100",
                "ksic5_name": "항공 여객 운송업",
            },
        ]
    )


def test_build_ksic_semantic_pairs_uses_hierarchy_labels():
    pairs = build_ksic_semantic_pairs(_taxonomy(), max_negative_pairs=10)

    assert "same_ksic5_different_entity" in set(pairs["relation"])
    assert "same_ksic4_sibling_ksic5" in set(pairs["relation"])
    assert "different_ksic1" in set(pairs["relation"])
    assert pairs[pairs["relation"].eq("same_ksic4_sibling_ksic5")]["label"].iloc[0] == 0.72
