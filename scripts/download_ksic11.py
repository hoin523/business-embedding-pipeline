#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bizembed.ksic import build_ksic11_taxonomy


KSIC_PACKAGE_URL = "https://urbanjj.r-universe.dev/src/contrib/KSIC_1.0.2.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize real 11th-revision KSIC industry hierarchy.")
    parser.add_argument("--url", default=KSIC_PACKAGE_URL)
    parser.add_argument("--raw-dir", default="data/raw/ksic11")
    parser.add_argument("--out", default="data/processed/ksic11_industry_taxonomy.parquet")
    return parser.parse_args()


def _download(url: str, target: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)


def _extract_rda_with_rscript(package_dir: Path, raw_dir: Path) -> None:
    if not shutil.which("Rscript"):
        raise SystemExit("Rscript is required to extract KSIC .rda files from the R package.")

    script = f"""
load("{package_dir / 'data' / 'ksicDB.rda'}")
load("{package_dir / 'data' / 'ksicTreeDB.rda'}")
write.csv(ksicDB, "{raw_dir / 'ksicDB.csv'}", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(ksicTreeDB, "{raw_dir / 'ksicTreeDB.csv'}", row.names = FALSE, fileEncoding = "UTF-8")
"""
    subprocess.run(["Rscript", "-e", script], check=True)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "KSIC.tar.gz"
        _download(args.url, archive)
        with tarfile.open(archive, "r:gz") as package:
            package.extractall(tmp_path)
        _extract_rda_with_rscript(tmp_path / "KSIC", raw_dir)

    tree = pd.read_csv(raw_dir / "ksicTreeDB.csv", dtype=str)
    taxonomy = build_ksic11_taxonomy(tree)
    taxonomy.to_parquet(out_path, index=False)
    taxonomy.to_csv(out_path.with_suffix(".csv"), index=False)

    print(f"source={args.url}")
    print(f"raw={raw_dir}")
    print(f"taxonomy={out_path}")
    print(f"rows={len(taxonomy)}")
    print(taxonomy.groupby(['ksic1_code', 'ksic1_name']).size().reset_index(name='ksic5_count').to_string(index=False))


if __name__ == "__main__":
    main()
