#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bizembed.ksic_pairs import build_ksic_semantic_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic training pairs from the real KSIC hierarchy.")
    parser.add_argument("--taxonomy", default="data/processed/ksic11_industry_taxonomy.parquet")
    parser.add_argument("--out", default="data/processed/ksic11_semantic_pairs.parquet")
    parser.add_argument("--max-negative-pairs", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = pd.read_parquet(args.taxonomy) if args.taxonomy.endswith(".parquet") else pd.read_csv(args.taxonomy)
    pairs = build_ksic_semantic_pairs(taxonomy, max_negative_pairs=args.max_negative_pairs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out, index=False)
    pairs.to_csv(out.with_suffix(".csv"), index=False)
    print(f"pairs={len(pairs)}")
    print(pairs["relation"].value_counts().to_string())
    print(f"saved={out}")


if __name__ == "__main__":
    main()
