#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bizembed.compact_pairs import build_compact_hard_pairs, build_compact_industry_pairs, build_compact_ksic_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact merchant-industry-context training pairs.")
    parser.add_argument("--taxonomy", default="data/processed/ksic11_industry_taxonomy.parquet")
    parser.add_argument("--out", default="data/processed/compact_card_training_pairs.parquet")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--entities-per-concept", type=int, default=5)
    parser.add_argument("--max-negative-pairs", type=int, default=4000)
    parser.add_argument("--no-field-token-variants", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [
        build_compact_industry_pairs(
            entities_per_concept=args.entities_per_concept,
            include_field_token_variants=not args.no_field_token_variants,
        ),
        build_compact_hard_pairs(),
    ]

    taxonomy_path = Path(args.taxonomy)
    if taxonomy_path.exists():
        taxonomy = pd.read_parquet(taxonomy_path)
        frames.append(build_compact_ksic_pairs(taxonomy, max_negative_pairs=args.max_negative_pairs))

    pairs = pd.concat(frames, ignore_index=True).drop_duplicates().sample(frac=1, random_state=42).reset_index(drop=True)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output, index=False)
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.out_csv, index=False)
    print(f"pairs={len(pairs)} -> {output}")
    print(pairs["relation"].value_counts().to_string())


if __name__ == "__main__":
    main()
