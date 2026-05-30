import pandas as pd

from bizembed.mcc import build_mcc_training_tables


def test_build_mcc_training_tables_adds_korean_alias_records_and_pairs():
    raw = pd.DataFrame(
        [
            {
                "mcc": "4511",
                "edited_description": "Airlines, Air Carriers",
                "combined_description": "Airlines, Air Carriers",
            },
            {
                "mcc": "5812",
                "edited_description": "Eating Places, Restaurants",
                "combined_description": "Eating Places, Restaurants",
            },
        ]
    )

    records, pairs = build_mcc_training_tables(raw)

    assert "업종명: 항공사" in set(records["text"])
    assert "업종명: 일반음식점" in set(records["text"])
    assert not records["text"].str.contains("품목명:", regex=False).any()
    assert (pairs["relation"] == "same_mcc_alias").any()
    assert (pairs["relation"] == "different_mcc_category").any()
