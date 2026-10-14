# Selects a coreset baseline once per dataset and seed, with k = n1 + ratio * n1 as in the eigen run.
# The trainers load the saved indices through --strategy.
import io
import os
import re
import json
import time
import argparse
import contextlib
import numpy as np

from baselines import get_baseline_indexes

FEATURES = {"kcenter": "pixels32", "facility_location": "pixels32", "kmeans++": "pixels32", "semdedup": "clip_vitb16"}


def parse_args():
    parser = argparse.ArgumentParser(description="Coreset baseline selection.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="pick | hpdv2 | rapidata | roco")
    parser.add_argument("--strategy", type=str, required=True, choices=list(FEATURES))
    parser.add_argument("--first_eigen", type=int, default=1024, help="Stage-1 budget n1 of the eigen run.")
    parser.add_argument("--ratio", type=float, default=6)
    parser.add_argument("--features_dir", type=str, default="baseline_features")
    parser.add_argument("--out_dir", type=str, default="baseline_indices")
    return parser.parse_args()


args = parse_args()
features = FEATURES[args.strategy]
X = np.load(os.path.join(args.features_dir, f"{features}_{args.dataset_name}_{args.seed}.npz"))["X"]
k = args.first_eigen + int(args.first_eigen * args.ratio)

log = io.StringIO()
started = time.strftime("%Y-%m-%d %H:%M:%S")
t0 = time.perf_counter()
with contextlib.redirect_stdout(log):
    indices = get_baseline_indexes(args.strategy, X, k, args.seed)
wall_seconds = time.perf_counter() - t0
print(log.getvalue(), end="")

record = {"strategy": args.strategy, "dataset": args.dataset_name, "seed": args.seed, "features": features,
          "n": int(X.shape[0]), "k": k, "selected": len(indices), "unique": len(set(indices)),
          "selection_seconds": float(re.search(r"selection_seconds=([\d.]+)", log.getvalue()).group(1)),
          "wall_seconds": round(wall_seconds, 2), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
          "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "log": log.getvalue().strip()}
os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir, f"{args.strategy}_{args.dataset_name}_{args.seed}.json")
with open(out_path + ".tmp", "w") as f:
    json.dump({**record, "indices": [int(i) for i in indices]}, f)
os.replace(out_path + ".tmp", out_path)
with open(os.path.join(args.out_dir, "timings.jsonl"), "a") as f:
    f.write(json.dumps(record) + "\n")
print(f"saved {out_path}  selected {record['selected']} ({record['unique']} unique) of n={record['n']}, k={k}")
