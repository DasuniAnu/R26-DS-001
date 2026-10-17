import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/eval/test CSV splits for Sinhala SLR52 Whisper fine-tuning."
    )
    parser.add_argument("--metadata", default="slr52_metadata.csv")
    parser.add_argument("--output-dir", default="metadata")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit for quick smoke tests. 0 uses all rows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()
    metadata_path = root / args.metadata
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path, encoding="utf-8")
    required = {"audio_path", "transcription", "speaker_id", "utterance_id", "part"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required column(s): {', '.join(missing)}")

    df["audio_path"] = df["audio_path"].astype(str)
    df["transcription"] = df["transcription"].astype(str)
    df["speaker_id"] = df["speaker_id"].astype(str)
    df["utterance_id"] = df["utterance_id"].astype(str)

    exists = df["audio_path"].map(lambda p: (root / p).exists())
    if not exists.all():
        bad = df.loc[~exists, "audio_path"].head(10).tolist()
        raise SystemExit("Some audio files do not exist:\n" + "\n".join(bad))

    if args.limit and args.limit > 0:
        df = df.sample(frac=1.0, random_state=args.seed).head(args.limit)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_total = len(df)
    n_test = int(round(n_total * args.test_ratio))
    n_eval = int(round(n_total * args.eval_ratio))
    n_train = n_total - n_eval - n_test
    if n_train <= 0:
        raise SystemExit("Split ratios leave no training rows.")

    train = df.iloc[:n_train].copy()
    eval_df = df.iloc[n_train : n_train + n_eval].copy()
    test = df.iloc[n_train + n_eval :].copy()

    train.to_csv(output_dir / "train.csv", index=False, encoding="utf-8")
    eval_df.to_csv(output_dir / "eval.csv", index=False, encoding="utf-8")
    test.to_csv(output_dir / "test.csv", index=False, encoding="utf-8")

    print(f"rows: total={n_total} train={len(train)} eval={len(eval_df)} test={len(test)}")
    print(f"wrote: {output_dir / 'train.csv'}")
    print(f"wrote: {output_dir / 'eval.csv'}")
    print(f"wrote: {output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
