from collections import Counter
from datasets import load_dataset

dataset = load_dataset("Rapidata/human-style-preferences-images", split="train")
prompts = dataset["prompt"]
prompt_counts = Counter(prompts)

test_prompts = {p for p, c in prompt_counts.items() if c < 240}
train_prompts = {p for p, c in prompt_counts.items() if c >= 240}

with open("test_rapidata.txt", "w", encoding="utf-8") as f:
    for prompt in sorted(test_prompts):
        f.write(prompt + "\n")

train_indices = [idx for idx, prompt in enumerate(prompts)
                 if prompt in train_prompts]
with open("train_rapidata.txt", "w", encoding="utf-8") as f:
    for idx in train_indices:
        f.write(str(idx) + "\n")

print(f"Unique test prompts (<240): {len(test_prompts)}")
print(f"Train prompts (>=240): {len(train_prompts)}")
print(f"Total train rows written: {len(train_indices)}")
