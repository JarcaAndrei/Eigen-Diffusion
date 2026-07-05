import os
import argparse

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

DEVICE = "cuda"
PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL = "yuvalkirstain/PickScore_v1"


def calc_pickscore(processor, model, prompt, images):
    image_inputs = processor(images=images, padding=True, truncation=True,
                             max_length=77, return_tensors="pt").to(DEVICE)
    text_inputs = processor(text=prompt, padding=True, truncation=True,
                            max_length=77, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_embs = model.get_image_features(**image_inputs).pooler_output
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs).pooler_output
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        probs = torch.softmax(scores, dim=-1)
    return scores.cpu().tolist(), probs.cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True,
                    help="inference directories to score.")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(PROCESSOR)
    model = AutoModel.from_pretrained(MODEL).eval().to(DEVICE)

    for path in args.paths:
        folders = sorted(f for f in os.listdir(path)
                         if os.path.isdir(os.path.join(path, f)))
        if not folders:
            print(f"[skip] no run folders in {path}")
            continue
        dfs = [pd.read_csv(f"{path}/{folder}/metadata.csv") for folder in folders]
        model_names = [os.path.basename(f) for f in folders]

        results = []
        for i in tqdm(range(len(dfs[0])), desc=path):
            prompt = dfs[0].iloc[i]["caption"]
            images = [Image.open(dfs[k].iloc[i]["image_path"]).convert("RGB")
                      for k in range(len(folders))]
            scores, probs = calc_pickscore(processor, model, prompt, images)
            winner = model_names[int(torch.tensor(probs).argmax())]
            row = {"prompt_idx": i, "caption": prompt, "winner": winner}
            for j, name in enumerate(model_names):
                row[f"{name}_score"] = scores[j]
                row[f"{name}_prob"] = probs[j]
            results.append(row)

        out_path = f"pickscore_{path}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
