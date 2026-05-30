#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import requests

from bizembed.mcc import build_mcc_training_tables


DEFAULT_URL = "https://raw.githubusercontent.com/greggles/mcc-codes/master/mcc_codes.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public MCC codes and build Korean card-category training data.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-out", default="data/raw/mcc_codes.csv")
    parser.add_argument("--records-out", default="data/processed/mcc_records.parquet")
    parser.add_argument("--pairs-out", default="data/processed/mcc_pairs.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_out = Path(args.raw_out)
    raw_out.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(args.url, timeout=30)
    response.raise_for_status()
    raw_out.write_bytes(response.content)

    raw = pd.read_csv(raw_out, dtype=str)
    records, pairs = build_mcc_training_tables(raw)

    records_out = Path(args.records_out)
    pairs_out = Path(args.pairs_out)
    records_out.parent.mkdir(parents=True, exist_ok=True)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(records_out, index=False)
    pairs.to_parquet(pairs_out, index=False)
    print(f"raw={len(raw)} -> {raw_out}")
    print(f"records={len(records)} -> {records_out}")
    print(f"pairs={len(pairs)} -> {pairs_out}")


if __name__ == "__main__":
    main()
