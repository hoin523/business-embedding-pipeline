#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bizembed.context import augment_pairs_with_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create card-like context variants from pair data.")
    parser.add_argument("--pairs", required=True, help="Input pair parquet or CSV.")
    parser.add_argument("--out", required=True, help="Output pair parquet or CSV.")
    parser.add_argument("--variants-per-pair", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=0, help="Optional input row sample before augmentation.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_pairs(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    pairs = read_pairs(args.pairs)
    if args.sample_size and len(pairs) > args.sample_size:
        pairs = pairs.sample(n=args.sample_size, random_state=args.seed)

    augmented = augment_pairs_with_context(
        pairs,
        variants_per_pair=args.variants_per_pair,
        seed=args.seed,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        augmented.to_parquet(output, index=False)
    else:
        augmented.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"pairs={len(augmented)} -> {output}")


if __name__ == "__main__":
    main()
