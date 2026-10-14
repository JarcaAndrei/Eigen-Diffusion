# Chosen-image features for the coreset baselines, on the same rows as get_embeddings.py:
# 32x32 pixels for k-center, facility location and k-means++, CLIP ViT-B/16 for SemDeDup.
import io
import os
import json
import time
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline feature builder.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="pick | hpdv2 | rapidata | roco")
    parser.add_argument("--features", type=str, required=True, choices=["pixels", "clip"])
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache dir (Pick-a-Pic).")
    parser.add_argument("--out_dir", type=str, default="baseline_features")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


args = parse_args()

t_load = time.perf_counter()
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

# chosen image of each row
labels = dataset["label_0"]
columns = {c: dataset.select_columns([c]) for c in (["jpg_0", "jpg_1"] if has_rejected else ["jpg_0"])}

if args.features == "clip":
    import open_clip
    # the OpenAI weights expect QuickGELU
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="openai", device="cuda", precision="fp32")
    model = model.eval()


class ChosenImages(Dataset):
    def __len__(self):
        return len(labels)

    def __getitem__(self, i):
        col = "jpg_0" if not has_rejected or int(labels[i]) == 1 else "jpg_1"
        img = Image.open(io.BytesIO(columns[col][i][col])).convert("RGB")
        if args.features == "pixels":
            return torch.from_numpy(np.asarray(img.resize((32, 32)), dtype=np.uint8).reshape(-1).copy())
        return preprocess(img)


loader = DataLoader(ChosenImages(), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
t0 = time.perf_counter()
chunks = []
with torch.no_grad():
    for batch in tqdm(loader):
        if args.features == "pixels":
            chunks.append(batch.numpy())
        else:
            emb = model.encode_image(batch.to("cuda", non_blocking=True))
            chunks.append(torch.nn.functional.normalize(emb, dim=1).cpu().numpy().astype(np.float32))
X = np.concatenate(chunks)
seconds = time.perf_counter() - t0

name = "pixels32" if args.features == "pixels" else "clip_vitb16"
os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir, f"{name}_{dataset_name}_{fixed_seed}.npz")
np.savez_compressed(out_path, X=X)
record = {"features": name, "dataset": dataset_name, "seed": fixed_seed, "n": int(X.shape[0]), "dim": int(X.shape[1]),
          "seconds": round(seconds, 2), "load_seconds": round(load_seconds, 2), "num_workers": args.num_workers,
          "batch_size": args.batch_size, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(os.path.join(args.out_dir, "timings.jsonl"), "a") as f:
    f.write(json.dumps(record) + "\n")
print(f"saved {out_path}  X={X.shape} {X.dtype}  {seconds:.1f}s ({X.shape[0] / seconds:.0f} img/s)")
