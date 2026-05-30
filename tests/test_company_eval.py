import pandas as pd

from bizembed.company_eval import (
    best_threshold,
    compact_records_from_frame,
    normalize_binary_label,
    threshold_metrics,
)


def test_compact_records_from_frame_uses_company_card_columns():
    frame = pd.DataFrame(
        [
            {
                "가맹점명": "보나비",
                "업종명": "일반한식",
                "기타텍스트": "내부회계관리팀 천호랄 출근 국내 평일",
            }
        ]
    )

    records = compact_records_from_frame(frame)

    assert records.loc[0, "compact_text"] == "보나비 일반한식 내부회계관리팀 천호랄 출근 국내 평일"


def test_normalize_binary_label_accepts_korean_labels():
    assert normalize_binary_label("유사") == 1
    assert normalize_binary_label("비유사") == 0
    assert normalize_binary_label("0.95") == 1
    assert normalize_binary_label("0.20") == 0


def test_threshold_metrics_selects_best_f1_threshold():
    metrics = threshold_metrics([95, 80, 30, 10], [1, 1, 0, 0], step=10)
    best = best_threshold(metrics)

    assert best["threshold"] <= 80
    assert best["f1"] == 1.0
