#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bizembed.semantic_pairs import build_industry_semantic_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synonym and hierarchy-aware industry semantic pairs.")
    parser.add_argument("--out", default="data/processed/industry_semantic_pairs.parquet")
    parser.add_argument("--entities-per-concept", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = build_industry_semantic_pairs(entities_per_concept=args.entities_per_concept)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        pairs.to_parquet(output, index=False)
    else:
        pairs.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"pairs={len(pairs)} -> {output}")
    print(pairs["relation"].value_counts().to_string())


if __name__ == "__main__":
    main()
