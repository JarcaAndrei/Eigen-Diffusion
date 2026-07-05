import io
import os
import argparse
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="EVA-CLIP embedding builder.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="pick | hpdv2 | rapidata | roco")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache dir (Pick-a-Pic).")
    parser.add_argument("--out_dir", type=str, default="all_embeddings")
    return parser.parse_args()


args = parse_args()

device = "cuda"
fixed_seed = args.seed
IMG_SIZE = 224
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


model, _, preprocess = open_clip.create_model_and_transforms(
    "EVA02-L-14",
    pretrained="merged2b_s4b_b131k",
    device=device,
    precision="fp32",
)
model = model.to(device).eval()


def load_pair(examples):
    chosen_images = []
    rejected_images = []
    for idx in range(len(examples["jpg_0"])):
        im0, label = examples["jpg_0"][idx], examples["label_0"][idx]
        if has_rejected:
            im1 = examples["jpg_1"][idx]
            chosen, rejected = (im0, im1) if int(label) == 1 else (im1, im0)
        else:
            chosen = im0
        chosen_img = Image.open(io.BytesIO(chosen)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        chosen_images.append(preprocess(chosen_img))
        if has_rejected:
            rejected_img = Image.open(io.BytesIO(rejected)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            rejected_images.append(preprocess(rejected_img))
    out = {"chosen": chosen_images}
    if has_rejected:
        out["rejected"] = rejected_images
    return out


dataset = dataset.with_transform(load_pair)

batch_size = 512
chosen_batch = []
rejected_batch = []
cache = []


def flush_batch():
    global cache, chosen_batch, rejected_batch
    if len(chosen_batch) == 0:
        return
    img_chosen = torch.stack(chosen_batch).to(device, non_blocking=True)
    with torch.no_grad():
        chosen_emb = model.encode_image(img_chosen).cpu().numpy()
        if has_rejected:
            img_rejected = torch.stack(rejected_batch).to(device, non_blocking=True)
            rejected_emb = model.encode_image(img_rejected).cpu().numpy()
    for i in range(len(chosen_emb)):
        cache.append((chosen_emb[i], rejected_emb[i] if has_rejected else None))
    chosen_batch.clear()
    rejected_batch.clear()


for sample in tqdm(dataset):
    chosen_batch.append(sample["chosen"])
    if has_rejected:
        rejected_batch.append(sample["rejected"])
    if len(chosen_batch) == batch_size:
        flush_batch()

flush_batch()

X, Y = [], []
for chosen_emb, rejected_emb in cache:
    chosen_emb = chosen_emb / (np.linalg.norm(chosen_emb) + 1e-8)
    X.append(chosen_emb)
    if has_rejected:
        rejected_emb = rejected_emb / (np.linalg.norm(rejected_emb) + 1e-8)
        Y.append(rejected_emb)
X = np.asarray(X)

os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir, f"embeddings_{dataset_name}_{fixed_seed}.npz")
if has_rejected:
    np.savez_compressed(out_path, X=X.astype(np.float32),
                        Y=np.asarray(Y).astype(np.float32))
else:
    np.savez_compressed(out_path, X=X.astype(np.float32))
print(f"saved {out_path}  X={X.shape}" + (f" Y={np.asarray(Y).shape}" if has_rejected else ""))

torch.cuda.empty_cache()