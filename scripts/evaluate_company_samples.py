#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from bizembed.company_eval import (
    best_threshold,
    compact_records_from_path,
    normalize_binary_label,
    threshold_metrics,
)
from bizembed.ingest import read_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate company card-like samples with a business embedding model.")
    parser.add_argument("--model", default="models/bge-m3-business-compact-card-context-v2")
    parser.add_argument("--input", required=True, help="CSV/XLSX/Parquet records or pair file.")
    parser.add_argument("--mode", choices=("records", "pairs"), default="records")
    parser.add_argument("--entity-column", default="")
    parser.add_argument("--industry-column", default="")
    parser.add_argument("--context-column", action="append", default=[])
    parser.add_argument("--text-a-column", default="text_a")
    parser.add_argument("--text-b-column", default="text_b")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="data/processed/company_eval")
    return parser.parse_args()


def _encode(model: SentenceTransformer, texts: list[str]):
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)


def evaluate_records(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = compact_records_from_path(
        args.input,
        entity_column=args.entity_column,
        industry_column=args.industry_column,
        context_columns=args.context_column,
    ).head(args.max_records)
    records.to_csv(out_dir / "records_compact.csv", index=False)
    if records.empty:
        raise SystemExit("No compact records were generated.")

    model = SentenceTransformer(args.model, device=args.device)
    embeddings = _encode(model, records["compact_text"].tolist())
    sims = util.cos_sim(embeddings, embeddings).cpu().numpy() * 100

    rows = []
    for idx, record in records.iterrows():
        candidates = sims[idx].argsort()[::-1]
        picked = 0
        for other_idx in candidates:
            if int(other_idx) == int(idx):
                continue
            other = records.iloc[int(other_idx)]
            rows.append(
                {
                    "record_id": record["record_id"],
                    "candidate_record_id": other["record_id"],
                    "score": round(float(sims[idx, other_idx]), 2),
                    "entity_name": record["entity_name"],
                    "industry_name": record["industry_name"],
                    "compact_text": record["compact_text"],
                    "candidate_entity_name": other["entity_name"],
                    "candidate_industry_name": other["industry_name"],
                    "candidate_compact_text": other["compact_text"],
                }
            )
            picked += 1
            if picked >= args.top_k:
                break
    pd.DataFrame(rows).to_csv(out_dir / "topk_candidates.csv", index=False)
    print(f"records={len(records)} -> {out_dir / 'records_compact.csv'}")
    print(f"topk={len(rows)} -> {out_dir / 'topk_candidates.csv'}")


def evaluate_pairs(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_table(args.input)
    if args.text_a_column not in pairs.columns or args.text_b_column not in pairs.columns:
        raise SystemExit(f"Pair file must contain {args.text_a_column!r} and {args.text_b_column!r}.")

    model = SentenceTransformer(args.model, device=args.device)
    emb_a = _encode(model, pairs[args.text_a_column].fillna("").astype(str).tolist())
    emb_b = _encode(model, pairs[args.text_b_column].fillna("").astype(str).tolist())
    pairs = pairs.copy()
    pairs["score"] = (util.cos_sim(emb_a, emb_b).diagonal().cpu().numpy() * 100).round(2)

    if args.label_column in pairs.columns:
        labels = pairs[args.label_column].map(normalize_binary_label)
        labeled = pairs[labels.notna()].copy()
        labeled["binary_label"] = labels[labels.notna()].astype(int)
        metrics = threshold_metrics(labeled["score"], labeled["binary_label"])
        metrics.to_csv(out_dir / "threshold_metrics.csv", index=False)
        best = best_threshold(metrics)
        if not best.empty:
            print(
                "best_threshold="
                f"{best['threshold']:.2f}, f1={best['f1']:.3f}, "
                f"precision={best['precision']:.3f}, recall={best['recall']:.3f}, accuracy={best['accuracy']:.3f}"
            )

    pairs.sort_values("score", ascending=False).to_csv(out_dir / "pair_scores.csv", index=False)
    print(f"pairs={len(pairs)} -> {out_dir / 'pair_scores.csv'}")


def main() -> None:
    args = parse_args()
    if args.mode == "pairs":
        evaluate_pairs(args)
    else:
        evaluate_records(args)


if __name__ == "__main__":
    main()
