import os
import re
import csv
import json
import argparse

import torch
from tqdm import tqdm
from datasets import load_dataset
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from transformers import T5Tokenizer
from peft import PeftModel

BASE_MODEL = "PixArt-alpha/PixArt-XL-2-512x512"
GUIDANCE_SCALE = 4.5          # PixArt default (vs 7.5 for SD1.5)
NUM_INFERENCE_STEPS = 20      # PixArt needs fewer steps than SD1.5
T5_MAX_LEN = 120              # T5 max length used during training


def load_prompts(dataset, cache_dir=None):
    if dataset == "pick":
        ds = load_dataset("pickapic-anonymous/pickapic_v1", None,
                          cache_dir=cache_dir)["validation"]
        prompts = ds.unique(column="caption")
    elif dataset == "hpdv2":
        with open("data/test.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [item["prompt"] for item in data if "prompt" in item]
    elif dataset == "rapidata":
        with open("test_rapidata.txt", "r", encoding="utf-8") as f:
            prompts = f.read().splitlines()
    elif dataset == "roco":
        with open("roco_validation.txt", "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"unknown dataset {dataset}")
    print(f"{dataset}: {len(prompts)} prompts")
    return prompts


def resolve_checkpoint(run_dir, checkpoint):
    if checkpoint is not None:
        ckpt = f"checkpoint-{checkpoint}"
    else:
        dirs = [d for d in os.listdir(run_dir) if re.match(r"checkpoint-\d+$", d)]
        if not dirs:
            return None
        ckpt = max(dirs, key=lambda x: int(re.search(r"\d+", x).group()))
    path = os.path.join(run_dir, ckpt)
    return path if os.path.isdir(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["pick", "hpdv2", "rapidata", "roco"])
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--checkpoint", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=None, help="HF cache dir (Pick-a-Pic).")
    ap.add_argument("--out_prefix", default="generated")
    args = ap.parse_args()

    prompts = load_prompts(args.dataset, cache_dir=args.cache_dir)
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")

    def is_truncated(prompt):
        return len(tokenizer(prompt, truncation=False)["input_ids"]) > T5_MAX_LEN

    run_names = [f for f in os.listdir(args.exp_dir)
                 if os.path.isdir(os.path.join(args.exp_dir, f))]

    for run in run_names:
        run_dir = os.path.join(args.exp_dir, run)
        ckpt_path = resolve_checkpoint(run_dir, args.checkpoint)
        if ckpt_path is None:
            print(f"[skip] no checkpoint in {run_dir}")
            continue
        print("loading", ckpt_path)

        base_transformer = Transformer2DModel.from_pretrained(
            BASE_MODEL, subfolder="transformer", torch_dtype=torch.float16)
        transformer = PeftModel.from_pretrained(base_transformer, ckpt_path)
        transformer = transformer.merge_and_unload()  # fuse LoRA for faster inference

        pipeline = PixArtAlphaPipeline.from_pretrained(
            BASE_MODEL, transformer=transformer, torch_dtype=torch.float16).to("cuda")

        output_dir = f"{args.out_prefix}_{os.path.basename(args.exp_dir)}/generated_{run}"
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, "metadata.csv")

        rows = []
        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts),
                                desc=f"{run}"):
            seed = idx * 100 + args.seed
            generator = torch.Generator(device="cuda").manual_seed(seed)
            image = pipeline(prompt, num_inference_steps=NUM_INFERENCE_STEPS,
                             guidance_scale=GUIDANCE_SCALE,
                             generator=generator).images[0]
            fname = f"{idx:05d}_s{seed:02d}.png"
            image_path = os.path.join(output_dir, fname)
            image.save(image_path)
            rows.append({
                "prompt_idx": idx, "caption": prompt, "seed": seed,
                "image_path": image_path, "guidance_scale": GUIDANCE_SCALE,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "truncated": is_truncated(prompt),
            })

        if not rows:
            print(f"[skip] no prompts for {run}")
            del pipeline, transformer, base_transformer
            torch.cuda.empty_cache()
            continue

        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {len(rows)} images -> {output_dir}")

        del pipeline, transformer, base_transformer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
