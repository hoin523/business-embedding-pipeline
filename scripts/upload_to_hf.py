#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload prepared data or model artifacts to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=["dataset", "model"], required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(Path(args.path)),
        path_in_repo=".",
    )
    print(f"uploaded={args.repo_type}:{args.repo_id}")


if __name__ == "__main__":
    main()
