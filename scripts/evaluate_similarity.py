#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sentence_transformers import SentenceTransformer, util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cosine similarity for business text pairs.")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pd.read_parquet(args.pairs) if args.pairs.endswith(".parquet") else pd.read_csv(args.pairs)
    pairs = pairs.head(args.limit)
    model = SentenceTransformer(args.model)
    emb_a = model.encode(pairs["text_a"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    emb_b = model.encode(pairs["text_b"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(emb_a, emb_b).diagonal().cpu().numpy()
    result = pairs.copy()
    result["cosine"] = scores
    print(result[["label", "cosine", "relation", "text_a", "text_b"]].to_string(index=False))


if __name__ == "__main__":
    main()
