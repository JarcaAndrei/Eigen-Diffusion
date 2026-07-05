import os
import re
import csv
import json
import argparse

import torch
import safetensors.torch
from tqdm import tqdm
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50


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
    path = os.path.join(run_dir, ckpt, "lora", "pytorch_lora_weights.safetensors")
    return path if os.path.isfile(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["pick", "hpdv2", "rapidata", "roco"])
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--checkpoint", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=None, help="HF cache dir (Pick-a-Pic).")
    ap.add_argument("--out_prefix", default="inference")
    args = ap.parse_args()

    prompts = load_prompts(args.dataset, cache_dir=args.cache_dir)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def is_truncated(prompt):
        tokens = tokenizer(prompt, truncation=False, add_special_tokens=True)["input_ids"]
        return len(tokens) > 77

    run_names = [f for f in os.listdir(args.exp_dir)
                 if os.path.isdir(os.path.join(args.exp_dir, f))]

    for run in run_names:
        run_dir = os.path.join(args.exp_dir, run)
        ckpt_path = resolve_checkpoint(run_dir, args.checkpoint)
        if ckpt_path is None:
            print(f"[skip] no checkpoint in {run_dir}")
            continue
        print("loading", ckpt_path)

        pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16).to("cuda")
        pipeline.safety_checker = None

        state_dict = safetensors.torch.load_file(ckpt_path)
        new_state = {k.replace("unet.base_model.model", "unet"): v
                     for k, v in state_dict.items()}
        pipeline.load_lora_weights(new_state, adapter_name="dpo")
        pipeline.set_adapters(["dpo"])
        pipeline.fuse_lora()

        output_dir = f"{args.out_prefix}_{os.path.basename(args.exp_dir)}/generated_{run}"
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, "metadata.csv")

        rows = []
        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts),
                                desc=f"{run}"):
            seed = idx * 100 + args.seed
            generator = torch.manual_seed(seed)
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
            del pipeline
            torch.cuda.empty_cache()
            continue

        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {len(rows)} images -> {output_dir}")

        del pipeline
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
