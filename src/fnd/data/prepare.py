import argparse
import os

import pandas as pd


def normalize_kaggle_fake_real(in_dir: str) -> pd.DataFrame:
    true_csv = os.path.join(in_dir, "True.csv")
    fake_csv = os.path.join(in_dir, "Fake.csv")
    if not (os.path.isfile(true_csv) and os.path.isfile(fake_csv)):
        raise FileNotFoundError(
            f"Expected True.csv and Fake.csv in {in_dir}. Found: True.csv={os.path.isfile(true_csv)}, Fake.csv={os.path.isfile(fake_csv)}"
        )

    df_true_raw = pd.read_csv(true_csv)
    df_fake_raw = pd.read_csv(fake_csv)

    def normalize(df_raw: pd.DataFrame, label_value: int) -> pd.DataFrame:
        title = (
            df_raw["title"].fillna("")
            if "title" in df_raw.columns
            else pd.Series([""] * len(df_raw))
        )
        text = (
            df_raw["text"].fillna("")
            if "text" in df_raw.columns
            else pd.Series([""] * len(df_raw))
        )
        combined = (title.astype(str) + "\n\n" + text.astype(str)).str.strip()
        return pd.DataFrame({"text": combined, "label": label_value})

    df_true = normalize(df_true_raw, 0)
    df_fake = normalize(df_fake_raw, 1)
    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)
    return df


def write_processed(df: pd.DataFrame, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "dataset.csv")
    df[["text", "label"]].to_csv(out_csv, index=False)
    return out_csv


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Fake News Detector"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["kaggle_fake_real"], help="Dataset name"
    )
    parser.add_argument(
        "--in_dir", required=True, help="Input directory containing raw files"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for processed dataset"
    )
    args = parser.parse_args()

    if args.dataset == "kaggle_fake_real":
        df = normalize_kaggle_fake_real(args.in_dir)
        out_csv = write_processed(df, args.out_dir)
        print(f"Wrote processed dataset to {out_csv} (rows={len(df)})")


if __name__ == "__main__":
    main()
