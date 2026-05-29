#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bizembed.collect import DataGoKrClient, detect_industry_codes


SBIZ_BASE_URL = "https://apis.data.go.kr/B553077/api/open/sdsc2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all SBIZ store data by discovering small industry codes.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--service-key-env", default="DATA_GO_KR_SERVICE_KEY")
    parser.add_argument("--num-rows", type=int, default=1000)
    parser.add_argument("--max-pages-per-code", type=int)
    parser.add_argument("--max-codes", type=int, help="Debug limit for smoke tests.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between industry codes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service_key = os.environ.get(args.service_key_env)
    if not service_key:
        raise SystemExit(f"Missing service key env: {args.service_key_env}")

    client = DataGoKrClient(service_key=service_key)
    code_endpoint = f"{SBIZ_BASE_URL}/smallUpjongList"
    code_rows = client.fetch_page(code_endpoint, page_no=1, num_rows=1000)
    codes = detect_industry_codes(code_rows)
    if args.max_codes:
        codes = codes[: args.max_codes]
    if not codes:
        raise SystemExit("No SBIZ small industry codes were returned from smallUpjongList.")

    store_endpoint = f"{SBIZ_BASE_URL}/storeListInUpjong"
    rows = []
    for index, code in enumerate(codes, start=1):
        print(f"[{index}/{len(codes)}] {code}")
        for page in client.iter_pages(
            store_endpoint,
            num_rows=args.num_rows,
            max_pages=args.max_pages_per_code,
            extra={"divId": "indsSclsCd", "key": code},
        ):
            rows.extend(page)
        time.sleep(args.sleep)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).drop_duplicates()
    if output.suffix == ".parquet":
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"codes={len(codes)} rows={len(df)} -> {output}")


if __name__ == "__main__":
    main()
