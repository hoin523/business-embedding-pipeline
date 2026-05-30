#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RECORD_ROWS = [
    {"가맹점명": "보나비", "업종명": "일반한식", "기타텍스트": "내부회계관리팀 천호랄 출근 국내 평일"},
    {"가맹점명": "대한항공", "업종명": "항공사", "기타텍스트": "내부회계관리팀 천호랄 출근 국내 평일"},
    {"가맹점명": "스타벅스", "업종명": "커피전문점", "기타텍스트": "재무팀 월말 결산 법인카드"},
    {"가맹점명": "이디야커피", "업종명": "카페", "기타텍스트": "재무팀 월말 결산 법인카드"},
]

PAIR_ROWS = [
    {
        "text_a": "보나비 일반한식 내부회계관리팀 천호랄 출근 국내 평일",
        "text_b": "보나비 한식음식점 내부회계관리팀 천호랄 출근 국내 평일",
        "label": "유사",
    },
    {
        "text_a": "보나비 일반한식 내부회계관리팀 천호랄 출근 국내 평일",
        "text_b": "대한항공 항공사 내부회계관리팀 천호랄 출근 국내 평일",
        "label": "비유사",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create company evaluation CSV templates.")
    parser.add_argument("--out-dir", default="data/templates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(RECORD_ROWS).to_csv(out_dir / "company_records_template.csv", index=False)
    pd.DataFrame(PAIR_ROWS).to_csv(out_dir / "company_pairs_template.csv", index=False)
    print(f"records_template={out_dir / 'company_records_template.csv'}")
    print(f"pairs_template={out_dir / 'company_pairs_template.csv'}")


if __name__ == "__main__":
    main()
