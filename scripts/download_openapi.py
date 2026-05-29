#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bizembed.collect import DataGoKrClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a paginated data.go.kr OpenAPI endpoint.")
    parser.add_argument("--endpoint", required=True, help="Full endpoint URL.")
    parser.add_argument("--out", required=True, help="Output CSV or parquet path.")
    parser.add_argument("--service-key-env", default="DATA_GO_KR_SERVICE_KEY")
    parser.add_argument("--num-rows", type=int, default=1000)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--param", action="append", default=[], help="Extra query parameter as key=value. Repeatable.")
    return parser.parse_args()


def parse_extra(values: list[str]) -> dict[str, str]:
    extra = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--param must be key=value, got: {value}")
        key, raw = value.split("=", 1)
        extra[key] = raw
    return extra


def main() -> None:
    args = parse_args()
    service_key = os.environ.get(args.service_key_env)
    if not service_key:
        raise SystemExit(f"Missing service key env: {args.service_key_env}")

    client = DataGoKrClient(service_key=service_key)
    rows = []
    for page in client.iter_pages(
        args.endpoint,
        num_rows=args.num_rows,
        max_pages=args.max_pages,
        extra=parse_extra(args.param),
    ):
        rows.extend(page)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if output.suffix == ".parquet":
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"rows={len(df)} -> {output}")


if __name__ == "__main__":
    main()
