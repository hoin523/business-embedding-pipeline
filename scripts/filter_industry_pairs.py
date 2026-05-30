#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bizembed.pair_filter import filter_industry_training_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep industry-bearing pairs and remove item-name segments.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_pairs(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    pairs = read_pairs(args.pairs)
    filtered = filter_industry_training_pairs(pairs)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        filtered.to_parquet(output, index=False)
    else:
        filtered.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"pairs={len(pairs)} filtered={len(filtered)} -> {output}")


if __name__ == "__main__":
    main()
