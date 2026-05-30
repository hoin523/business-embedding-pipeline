from __future__ import annotations

import pandas as pd


def prepare_training_pairs(
    pairs: pd.DataFrame,
    *,
    loss_name: str = "mnrl",
    max_positive_pairs: int = 10000,
    max_pairs: int = 0,
    negative_ratio: float = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Select pairs appropriate for the requested sentence-transformer loss."""
    if loss_name == "mnrl":
        selected = pairs[pairs["label"] >= 0.7].copy()
        if max_positive_pairs and len(selected) > max_positive_pairs:
            selected = selected.sample(n=max_positive_pairs, random_state=seed)
        return selected.reset_index(drop=True)

    if loss_name == "cosine":
        if negative_ratio:
            positives = pairs[pairs["label"] >= 0.7].copy()
            negatives = pairs[pairs["label"] < 0.7].copy()
            max_negatives = int(len(positives) * negative_ratio)
            if max_negatives and len(negatives) > max_negatives:
                negatives = negatives.sample(n=max_negatives, random_state=seed)
            selected = pd.concat([positives, negatives], ignore_index=True)
            selected = selected.sample(frac=1, random_state=seed)
        else:
            selected = pairs.copy()
        if max_pairs and len(selected) > max_pairs:
            selected = selected.sample(n=max_pairs, random_state=seed)
        return selected.reset_index(drop=True)

    raise ValueError(f"Unsupported loss: {loss_name}")
