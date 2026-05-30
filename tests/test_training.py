import pandas as pd

from bizembed.training import prepare_training_pairs


def test_prepare_training_pairs_mnrl_keeps_positive_pairs_only():
    pairs = pd.DataFrame(
        [
            {"text_a": "a", "text_b": "b", "label": 0.85, "relation": "same"},
            {"text_a": "a", "text_b": "c", "label": 0.20, "relation": "different"},
        ]
    )

    result = prepare_training_pairs(pairs, loss_name="mnrl")

    assert result["label"].tolist() == [0.85]


def test_prepare_training_pairs_cosine_keeps_positive_and_negative_pairs():
    pairs = pd.DataFrame(
        [
            {"text_a": "a", "text_b": "b", "label": 0.85, "relation": "same"},
            {"text_a": "a", "text_b": "c", "label": 0.20, "relation": "different"},
        ]
    )

    result = prepare_training_pairs(pairs, loss_name="cosine")

    assert result["label"].tolist() == [0.85, 0.20]


def test_prepare_training_pairs_cosine_can_balance_negatives_by_positive_count():
    pairs = pd.DataFrame(
        [
            {"text_a": f"a{i}", "text_b": f"b{i}", "label": label, "relation": "r"}
            for i, label in enumerate([0.85, 0.90, 0.20, 0.20, 0.10, 0.20, 0.10])
        ]
    )

    result = prepare_training_pairs(pairs, loss_name="cosine", negative_ratio=2, seed=1)

    assert len(result[result["label"] >= 0.7]) == 2
    assert len(result[result["label"] < 0.7]) == 4
