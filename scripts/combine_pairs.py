#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine embedding pair parquet files.")
    parser.add_argument("--input", action="append", required=True, help="Input pair parquet file. Repeatable.")
    parser.add_argument("--out", required=True, help="Output combined parquet path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_parquet(path) for path in args.input]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    print(f"pairs={len(combined)} -> {output}")


if __name__ == "__main__":
    main()
