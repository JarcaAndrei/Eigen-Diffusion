import io
import os
import json
import time
import argparse
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset

IMG_SIZE = 224
# keep fixed: the batch size changes the embeddings slightly, and the selection with them
BATCH_SIZE = 512


def parse_args():
    parser = argparse.ArgumentParser(description="EVA-CLIP embedding builder.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="pick | hpdv2 | rapidata | roco")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache dir (Pick-a-Pic).")
    parser.add_argument("--out_dir", type=str, default="all_embeddings")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--x_only", action="store_true",
                        help="Embed the chosen image only: enough for --sft, not for DPO.")
    return parser.parse_args()


class Pairs(Dataset):
    def __init__(self, dataset, has_rejected, want_rejected, preprocess):
        self.rows = dataset.select_columns(["jpg_0", "label_0"] + (["jpg_1"] if has_rejected else []))
        self.has_rejected, self.want_rejected, self.preprocess = has_rejected, want_rejected, preprocess

    def __len__(self):
        return len(self.rows)

    def image(self, raw):
        return self.preprocess(Image.open(io.BytesIO(raw)).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

    def __getitem__(self, i):
        row = self.rows[i]
        if not self.has_rejected:
            return self.image(row["jpg_0"])
        chosen, rejected = (row["jpg_0"], row["jpg_1"]) if int(row["label_0"]) == 1 else (row["jpg_1"], row["jpg_0"])
        if not self.want_rejected:
            return self.image(chosen)
        return self.image(chosen), self.image(rejected)


def normalise(emb):
    return np.asarray([v / (np.linalg.norm(v) + 1e-8) for v in emb])


def main():
    args = parse_args()

    t_load = time.perf_counter()
    device = "cuda"
    fixed_seed = args.seed
    max_samples = 100000
    dataset_name = args.dataset_name

    if dataset_name == "hpdv2":
        dataset = load_from_disk(os.path.join(args.data_dir, "hpdv2_pickapic_format"))
    elif dataset_name == "pick":
        dataset = load_dataset("pickapic-anonymous/pickapic_v1", None,
                               cache_dir=args.cache_dir, data_dir=None)["train"]
    elif dataset_name == "rapidata":
        dataset = load_from_disk(os.path.join(args.data_dir, "processed_rapidata_dataset_bytes"))
    elif dataset_name == "roco":
        dataset = load_from_disk(os.path.join(args.data_dir, "roco_dataset"))
    else:
        raise ValueError(f"unknown dataset {dataset_name}")

    has_rejected = "jpg_1" in dataset.column_names

    keep_cols = ["jpg_0", "label_0", "caption"] + (["jpg_1"] if has_rejected else [])
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])

    not_split_idx = [i for i, label_0 in enumerate(dataset['label_0'])
                     if label_0 in (0, 1)]
    dataset = dataset.select(not_split_idx)

    if dataset_name in ["roco", "rapidata"]:
        dataset = dataset.shuffle(seed=fixed_seed)
    else:
        dataset = dataset.shuffle(seed=fixed_seed).select(range(max_samples))
    load_seconds = time.perf_counter() - t_load

    model, _, preprocess = open_clip.create_model_and_transforms(
        "EVA02-L-14",
        pretrained="merged2b_s4b_b131k",
        device=device,
        precision="fp32",
    )
    model = model.to(device).eval()

    want_rejected = has_rejected and not args.x_only
    loader = DataLoader(Pairs(dataset, has_rejected, want_rejected, preprocess), batch_size=BATCH_SIZE,
                        num_workers=args.num_workers, shuffle=False, pin_memory=True,
                        prefetch_factor=1 if args.num_workers else None)
    t0 = time.perf_counter()
    X, Y = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            chosen, rejected = batch if want_rejected else (batch, None)
            X.append(normalise(model.encode_image(chosen.to(device, non_blocking=True)).cpu().numpy()))
            if rejected is not None:
                Y.append(normalise(model.encode_image(rejected.to(device, non_blocking=True)).cpu().numpy()))
    X = np.concatenate(X).astype(np.float32)
    Y = np.concatenate(Y).astype(np.float32) if Y else None
    seconds = time.perf_counter() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"embeddings_{dataset_name}_{fixed_seed}.npz")
    if Y is not None:
        np.savez_compressed(out_path, X=X, Y=Y)
    else:
        np.savez_compressed(out_path, X=X)
    images = len(X) + (len(Y) if Y is not None else 0)
    record = {"features": "eva02_l14", "dataset": dataset_name, "seed": fixed_seed, "n": int(X.shape[0]),
              "dim": int(X.shape[1]), "x_only": Y is None, "images": images, "seconds": round(seconds, 2),
              "load_seconds": round(load_seconds, 2), "num_workers": args.num_workers, "batch_size": BATCH_SIZE,
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out_dir, "timings.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"saved {out_path}  X={X.shape}" + (f" Y={Y.shape}" if Y is not None else "") +
          f"  {seconds:.1f}s ({images / seconds:.0f} img/s)")


if __name__ == "__main__":
    main()
