#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sentence_transformers import SentenceTransformer, util


CASES = [
    (
        "한식 alias high",
        "high",
        "보나비 일반한식 내부회계관리팀 천호랄 출근 국내 평일",
        "보나비 한식음식점 내부회계관리팀 천호랄 출근 국내 평일",
    ),
    (
        "한식 vs 항공 low",
        "low",
        "보나비 일반한식 내부회계관리팀 천호랄 출근 국내 평일",
        "대한항공 항공사 내부회계관리팀 천호랄 출근 국내 평일",
    ),
    (
        "한식 vs 음식점 related",
        "medium_high",
        "본우리집밥 한식 영업본부 고객사 방문 국내 평일",
        "맛나식당 일반음식점 영업본부 고객사 방문 국내 평일",
    ),
    (
        "카페 vs 베이커리 related",
        "medium",
        "스타벅스 커피전문점 재무팀 월말 결산 법인카드",
        "파리바게뜨 베이커리 재무팀 월말 결산 법인카드",
    ),
    (
        "식자재 유통 alias high",
        "high",
        "한국식품유통 식자재 도매업 급식재료 정기구매",
        "대한식자재유통 급식재료 유통 급식재료 정기구매",
    ),
    (
        "주유소 vs 카페 low",
        "low",
        "SK주유소 주유소 출장 차량 유류비",
        "이디야커피 카페 출장 차량 유류비",
    ),
    (
        "소프트웨어 related high",
        "high",
        "더존비즈온 소프트웨어 개발업 내부통제 점검",
        "안랩 정보보안 내부통제 점검",
    ),
    (
        "병원 vs 약국 related",
        "medium",
        "세브란스병원 종합병원 임직원 진료비",
        "온누리약국 약국 임직원 진료비",
    ),
    (
        "마트 vs 편의점 related",
        "medium",
        "이마트 대형마트 부서 공통 경비",
        "CU 편의점 부서 공통 경비",
    ),
    (
        "전기공사 alias high",
        "high",
        "대성전기 전기공사업 시설관리팀 유지보수",
        "한빛전기 전기설비공사 시설관리팀 유지보수",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compact card-like business text pairs.")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--tuned-model", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-csv", default="data/processed/compact_card_eval.csv")
    parser.add_argument("--out-html", default="data/processed/compact_card_eval.html")
    return parser.parse_args()


def _scores(model_name: str, left: list[str], right: list[str], device: str | None) -> list[float]:
    model = SentenceTransformer(model_name, device=device)
    emb_a = model.encode(left, convert_to_tensor=True, normalize_embeddings=True)
    emb_b = model.encode(right, convert_to_tensor=True, normalize_embeddings=True)
    return (util.cos_sim(emb_a, emb_b).diagonal().cpu().numpy() * 100).round(2).tolist()


def main() -> None:
    args = parse_args()
    rows = pd.DataFrame(CASES, columns=["case", "expected", "text_a", "text_b"])
    rows["baseline"] = _scores(args.baseline_model, rows["text_a"].tolist(), rows["text_b"].tolist(), args.device)
    rows["compact_tuned"] = _scores(args.tuned_model, rows["text_a"].tolist(), rows["text_b"].tolist(), args.device)
    rows["delta"] = (rows["compact_tuned"] - rows["baseline"]).round(2)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_csv, index=False)

    out_html = Path(args.out_html)
    out_html.write_text(rows.to_html(index=False, escape=False), encoding="utf-8")

    print(rows[["case", "expected", "baseline", "compact_tuned", "delta", "text_a", "text_b"]].to_string(index=False))
    print(f"csv={out_csv}")
    print(f"html={out_html}")


if __name__ == "__main__":
    main()
