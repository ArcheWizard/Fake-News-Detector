"""Extract sample test examples for manual testing."""
import argparse
import json
import random

from fnd.data.datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Extract test samples")
    parser.add_argument("--dataset", default="kaggle_fake_real", help="Dataset name")
    parser.add_argument("--data_dir", required=True, help="Data directory")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples per class")
    parser.add_argument("--out_file", default="test_samples.json", help="Output JSON file")
    args = parser.parse_args()

    bundle = load_dataset(args.dataset, args.data_dir)
    test_df = bundle.test_df

    # Sample both classes
    fake_samples = test_df[test_df["label"] == 1].sample(n=min(args.num_samples, sum(test_df["label"] == 1)))
    real_samples = test_df[test_df["label"] == 0].sample(n=min(args.num_samples, sum(test_df["label"] == 0)))

    samples = {
        "fake": [
            {"text": row["text"], "label": "fake"}
            for _, row in fake_samples.iterrows()
        ],
        "real": [
            {"text": row["text"], "label": "real"}
            for _, row in real_samples.iterrows()
        ]
    }

    with open(args.out_file, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Extracted {len(samples['fake'])} fake and {len(samples['real'])} real samples to {args.out_file}")


if __name__ == "__main__":
    main()
