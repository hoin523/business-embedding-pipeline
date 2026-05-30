from bizembed.semantic_pairs import build_industry_semantic_pairs


def test_build_industry_semantic_pairs_contains_synonym_positive():
    pairs = build_industry_semantic_pairs(contexts=("동일 문맥",), entities_per_concept=1)

    row = pairs[pairs["relation"].eq("same_entity_same_canonical_industry")].iloc[0]

    assert "표준업종명:" in row["text_a"]
    assert "원업종명:" in row["text_a"]
    assert row["label"] == 0.95


def test_build_industry_semantic_pairs_contains_different_parent_negative():
    pairs = build_industry_semantic_pairs(contexts=("동일 문맥",), entities_per_concept=1)

    negatives = pairs[pairs["relation"].eq("different_parent_industry_same_context")]

    assert len(negatives) > 0
    assert negatives["label"].max() <= 0.10
