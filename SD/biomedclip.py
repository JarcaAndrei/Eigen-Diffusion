import os
import argparse

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer

BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CONTEXT_LENGTH = 256
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_biomedclip():
    model, preprocess = create_model_from_pretrained(BIOMEDCLIP_MODEL)
    tokenizer = get_tokenizer(BIOMEDCLIP_MODEL)
    model.to(device).eval()
    return model, preprocess, tokenizer


def calc_biomedclip_scores(model, preprocess, tokenizer, caption, images):
    image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
    text_tokens = tokenizer([caption], context_length=CONTEXT_LENGTH).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(image_tensors, text_tokens)
        # features are already normalized by the model
        scores = (logit_scale * image_features @ text_features.T)[:, 0]  # (N,)
        probs = torch.softmax(scores, dim=-1)

    return scores.cpu().tolist(), probs.cpu().tolist()


def run(paths):
    model, preprocess, tokenizer = load_biomedclip()

    for path in paths:
        folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
        if not folders:
            print(f"[skip] no run folders in {path}")
            continue
        model_names = folders
        dfs = [pd.read_csv(os.path.join(path, f, "metadata.csv")) for f in folders]

        results = []

        for i in tqdm(range(len(dfs[0])), desc=path):
            caption = dfs[0].iloc[i]["caption"]
            images = [
                Image.open(dfs[k].iloc[i]["image_path"]).convert("RGB")
                for k in range(len(folders))
            ]

            scores, probs = calc_biomedclip_scores(model, preprocess, tokenizer, caption, images)

            winner_idx = int(torch.tensor(probs).argmax())
            winner_name = model_names[winner_idx]

            row = {"prompt_idx": i, "caption": caption, "winner": winner_name}
            for j, name in enumerate(model_names):
                row[f"{name}_score"] = scores[j]
                row[f"{name}_prob"] = probs[j]

            results.append(row)

        df_results = pd.DataFrame(results)
        out_path = f"biomedclip_{path}.csv"
        df_results.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True,
                    help="inference directories to score.")
    args = ap.parse_args()
    run(args.paths)
