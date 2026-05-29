#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bizembed.collect import DataGoKrClient


SBIZ_BASE_URL = "https://apis.data.go.kr/B553077/api/open/sdsc2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SBIZ store data by industry code.")
    parser.add_argument("--industry-code", action="append", required=True, help="Industry code. Repeatable.")
    parser.add_argument("--div-id", default="indsSclsCd", help="indsLclsCd, indsMclsCd, or indsSclsCd.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--service-key-env", default="DATA_GO_KR_SERVICE_KEY")
    parser.add_argument("--num-rows", type=int, default=1000)
    parser.add_argument("--max-pages", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service_key = os.environ.get(args.service_key_env)
    if not service_key:
        raise SystemExit(f"Missing service key env: {args.service_key_env}")

    client = DataGoKrClient(service_key=service_key)
    endpoint = f"{SBIZ_BASE_URL}/storeListInUpjong"
    rows = []
    for code in args.industry_code:
        for page in client.iter_pages(
            endpoint,
            num_rows=args.num_rows,
            max_pages=args.max_pages,
            extra={"divId": args.div_id, "key": code},
        ):
            rows.extend(page)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).drop_duplicates()
    if output.suffix == ".parquet":
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"rows={len(df)} -> {output}")


if __name__ == "__main__":
    main()
