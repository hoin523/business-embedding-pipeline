#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BGE-M3 on Korean business similarity pairs.")
    parser.add_argument("--pairs", required=True, help="Parquet or CSV with text_a, text_b, label columns.")
    parser.add_argument("--output", required=True, help="Model output directory.")
    parser.add_argument("--base-model", default="BAAI/bge-m3")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-positive-pairs", type=int, default=10000)
    parser.add_argument("--device", default=None, help="Optional torch device, for example mps, cuda, or cpu.")
    return parser.parse_args()


def read_pairs(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    pairs = read_pairs(args.pairs)
    pairs = pairs[pairs["label"] >= 0.7].copy()
    if args.max_positive_pairs and len(pairs) > args.max_positive_pairs:
        pairs = pairs.sample(n=args.max_positive_pairs, random_state=42)
    examples = [
        InputExample(texts=[row.text_a, row.text_b])
        for row in pairs.itertuples(index=False)
    ]
    if not examples:
        raise SystemExit("No positive pairs found. Generate pairs before training.")

    model = SentenceTransformer(args.base_model, device=args.device)
    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        optimizer_params={"lr": args.learning_rate},
        show_progress_bar=True,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
