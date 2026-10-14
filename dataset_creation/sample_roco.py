import math
import pandas as pd
from datasets import load_dataset
from transformers import CLIPTokenizer


OUT_TXT = "roco_validation.txt"
TARGET_N = 500
MIN_WORDS = 6
MAX_CLIP_TOKENS = 77


def clip_token_count(tokenizer, text):
    return len(tokenizer.encode(str(text)))


def main():
    print("Loading ROCO-radiology test split...")
    ds = load_dataset("mdwiratathya/ROCO-radiology", split="test")

    df = ds.to_pandas()[["caption"]]
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    df["wc"] = df["caption"].apply(lambda t: len(str(t).split()))
    df["tc"] = df["caption"].apply(lambda t: clip_token_count(tokenizer, t))

    before = len(df)
    df = df[(df["wc"] >= MIN_WORDS) & (df["tc"] <= MAX_CLIP_TOKENS)].reset_index(drop=True)
    print(f"After filter [>= {MIN_WORDS} words, <= {MAX_CLIP_TOKENS} CLIP tokens]: {len(df)} / {before}")

    max_wc = df["wc"].max()
    bin_edges = list(range(MIN_WORDS, max_wc + 1, 10)) + [max_wc + 1]
    labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges) - 1)]
    df["bin"] = pd.cut(df["wc"], bins=bin_edges, labels=labels, right=False)

    bin_counts = df["bin"].value_counts().sort_index()
    allocation = (bin_counts / len(df) * TARGET_N).apply(math.floor)
    remainder = TARGET_N - allocation.sum()
    fractions = (bin_counts / len(df) * TARGET_N) - allocation
    for lbl in fractions.nlargest(remainder).index:
        allocation[lbl] += 1

    sampled = pd.concat([
        df[df["bin"] == lbl].sample(n=min(n, (df["bin"] == lbl).sum()), random_state=42)
        for lbl, n in allocation.items() if n > 0
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(sampled["caption"].tolist()))

    wcs = sampled["wc"]
    print(f"Saved {len(sampled)} captions to {OUT_TXT}")
    print(f"Word count: mean {wcs.mean():.1f}, median: {wcs.median():.0f}, min: {wcs.min()}, max: {wcs.max()}")


if __name__ == "__main__":
    main()
