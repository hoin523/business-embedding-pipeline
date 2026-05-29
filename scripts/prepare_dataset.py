#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bizembed.ingest import read_table, standardize_dataframe
from bizembed.pairs import generate_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Korean business embedding records and pairs.")
    parser.add_argument("--source", required=True, help="Source name: sbiz, nara, license, ksic, or custom.")
    parser.add_argument("--input", action="append", required=True, help="Input CSV/XLSX/Parquet/ZIP file. Repeatable.")
    parser.add_argument("--records-out", required=True, help="Output canonical records parquet path.")
    parser.add_argument("--pairs-out", required=True, help="Output pair parquet path.")
    parser.add_argument("--max-pairs-per-group", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for input_path in args.input:
        raw = read_table(input_path)
        records.append(standardize_dataframe(raw, source=args.source))

    all_records = pd.concat(records, ignore_index=True).drop_duplicates()
    pairs = generate_pairs(all_records, max_pairs_per_group=args.max_pairs_per_group)

    records_out = Path(args.records_out)
    pairs_out = Path(args.pairs_out)
    records_out.parent.mkdir(parents=True, exist_ok=True)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    all_records.to_parquet(records_out, index=False)
    pairs.to_parquet(pairs_out, index=False)

    print(f"records={len(all_records)} -> {records_out}")
    print(f"pairs={len(pairs)} -> {pairs_out}")


if __name__ == "__main__":
    main()
