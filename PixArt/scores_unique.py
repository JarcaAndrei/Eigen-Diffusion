import os
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True,
                    help="inference directories that were scored.")
    ap.add_argument("--metric", choices=["pickscore", "biomedclip"],
                    default="pickscore", help="which score CSV to read.")
    args = ap.parse_args()

    for path in args.paths:
        csv_path = f"{args.metric}_{path}.csv"
        df = pd.read_csv(csv_path)
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

        avg_scores = {name: df[f"{name}_score"].mean() for name in folders}
        avg_scores = dict(sorted(avg_scores.items(),
                                 key=lambda x: x[1], reverse=True))

        print(f"\n=== {csv_path} ===")
        for name, score in avg_scores.items():
            print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()
