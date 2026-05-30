#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from sentence_transformers import SentenceTransformer, util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bizembed.industry_taxonomy import build_semantic_text
from bizembed.semantic_pairs import build_industry_semantic_pairs


CONTEXT = "내부회계관리팀 천호랄 출근 국내 평일"
MODEL_COLUMNS = {
    "baseline": "이전 업종 모델",
    "semantic": "의미 보정 모델",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic industry similarity pairs.")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--out-csv", default="data/processed/semantic_100_comparison_eval.csv")
    parser.add_argument("--out-html", default="data/processed/semantic_100_comparison_eval.html")
    parser.add_argument("--summary-csv", default="data/processed/semantic_100_comparison_summary.csv")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _semantic(entity_name: str, industry: str, context: str = CONTEXT) -> str:
    return build_semantic_text(entity_name=entity_name, raw_industry=industry, context=context)


def _targeted_rows() -> list[dict[str, object]]:
    cases = [
        (
            "한식 동의어",
            _semantic("보나비", "일반한식"),
            _semantic("보나비", "한식음식점"),
            0.95,
            "same_entity_same_canonical_industry",
        ),
        (
            "한식 축약어",
            _semantic("보나비", "일반한식"),
            _semantic("보나비", "한식"),
            0.95,
            "same_entity_same_canonical_industry",
        ),
        (
            "한식 vs 음식점 상위어",
            _semantic("보나비", "일반한식"),
            _semantic("보나비", "일반음식점"),
            0.62,
            "same_parent_different_canonical_industry",
        ),
        (
            "한식 vs 항공사",
            _semantic("보나비", "일반한식"),
            _semantic("대한항공", "항공사"),
            0.08,
            "different_parent_industry_same_context",
        ),
        (
            "식자재 도매 동의어",
            _semantic("한국식품유통", "식자재 도매업"),
            _semantic("대한식자재유통", "급식재료 유통"),
            0.88,
            "different_entity_same_canonical_industry",
        ),
        (
            "카페 vs 제과점",
            _semantic("스타벅스", "커피전문점"),
            _semantic("파리바게뜨", "베이커리"),
            0.62,
            "same_parent_different_canonical_industry",
        ),
        (
            "병원 vs 약국",
            _semantic("세브란스병원", "종합병원"),
            _semantic("온누리약국", "약국"),
            0.62,
            "same_parent_different_canonical_industry",
        ),
        (
            "소프트웨어 vs 보안",
            _semantic("더존비즈온", "소프트웨어 개발업"),
            _semantic("안랩", "정보보안"),
            0.62,
            "same_parent_different_canonical_industry",
        ),
        (
            "마트 vs 편의점",
            _semantic("이마트", "대형마트"),
            _semantic("CU", "편의점"),
            0.62,
            "same_parent_different_canonical_industry",
        ),
        (
            "카페 vs 주유소",
            _semantic("이디야커피", "카페"),
            _semantic("SK주유소", "주유소"),
            0.08,
            "different_parent_industry_same_context",
        ),
    ]
    return [
        {"case": case, "text_a": text_a, "text_b": text_b, "label": label, "relation": relation}
        for case, text_a, text_b, label, relation in cases
    ]


def _sample_rows(limit: int) -> pd.DataFrame:
    pairs = build_industry_semantic_pairs(entities_per_concept=5)
    same = pairs[pairs["relation"].isin(["same_entity_same_canonical_industry", "different_entity_same_canonical_industry"])]
    parent = pairs[pairs["relation"] == "same_parent_different_canonical_industry"]
    negative = pairs[pairs["relation"] == "different_parent_industry_same_context"]

    target = pd.DataFrame(_targeted_rows())
    rest_limit = max(limit - len(target), 0)
    sampled = pd.concat(
        [
            same.head(rest_limit // 2),
            parent.head(rest_limit // 4),
            negative.head(rest_limit - (rest_limit // 2) - (rest_limit // 4)),
        ],
        ignore_index=True,
    )
    sampled.insert(0, "case", [f"자동평가 {idx + 1}" for idx in range(len(sampled))])
    return pd.concat([target, sampled], ignore_index=True).head(limit)


def _scores(model_path: str, texts_a: Iterable[str], texts_b: Iterable[str], device: str | None) -> list[float]:
    kwargs = {"device": device} if device else {}
    model = SentenceTransformer(model_path, **kwargs)
    emb_a = model.encode(list(texts_a), convert_to_tensor=True, normalize_embeddings=True)
    emb_b = model.encode(list(texts_b), convert_to_tensor=True, normalize_embeddings=True)
    return (util.cos_sim(emb_a, emb_b).diagonal().cpu().numpy() * 100).round(2).tolist()


def _bucket(label: float) -> str:
    if label >= 0.85:
        return "높아야 함"
    if label >= 0.50:
        return "중간이어야 함"
    return "낮아야 함"


def _write_html(rows: pd.DataFrame, out_path: Path) -> None:
    display_rows = []
    for _, row in rows.iterrows():
        display_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['case']))}</td>"
            f"<td>{html.escape(str(row['expected']))}</td>"
            f"<td>{html.escape(str(row['relation']))}</td>"
            f"<td>{row['이전 업종 모델']:.2f}</td>"
            f"<td>{row['의미 보정 모델']:.2f}</td>"
            f"<td>{row['변화']:+.2f}</td>"
            f"<td><div>{html.escape(str(row['text_a']))}</div><div class='b'>{html.escape(str(row['text_b']))}</div></td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Semantic Industry Similarity Evaluation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 10px; vertical-align: top; }}
    th {{ text-align: left; background: #f0f4f8; position: sticky; top: 0; }}
    .b {{ margin-top: 6px; color: #52606d; }}
  </style>
</head>
<body>
  <h1>Semantic Industry Similarity Evaluation</h1>
  <table>
    <thead>
      <tr>
        <th>케이스</th><th>기대</th><th>관계</th><th>이전 업종 모델</th><th>의미 보정 모델</th><th>변화</th><th>비교 텍스트</th>
      </tr>
    </thead>
    <tbody>
      {''.join(display_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = _sample_rows(args.limit)
    rows["expected"] = rows["label"].map(_bucket)
    rows[MODEL_COLUMNS["baseline"]] = _scores(args.baseline_model, rows["text_a"], rows["text_b"], args.device)
    rows[MODEL_COLUMNS["semantic"]] = _scores(args.semantic_model, rows["text_a"], rows["text_b"], args.device)
    rows["변화"] = rows[MODEL_COLUMNS["semantic"]] - rows[MODEL_COLUMNS["baseline"]]

    out_csv = Path(args.out_csv)
    out_html = Path(args.out_html)
    summary_csv = Path(args.summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_csv, index=False)
    _write_html(rows, out_html)

    summary = (
        rows.groupby("expected")[[MODEL_COLUMNS["baseline"], MODEL_COLUMNS["semantic"], "변화"]]
        .mean()
        .round(2)
        .reset_index()
    )
    summary.to_csv(summary_csv, index=False)

    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    print(f"wrote {summary_csv}")
    print(summary.to_string(index=False))
    print(rows.head(12)[["case", "expected", "relation", MODEL_COLUMNS["baseline"], MODEL_COLUMNS["semantic"], "변화"]].to_string(index=False))


if __name__ == "__main__":
    main()
