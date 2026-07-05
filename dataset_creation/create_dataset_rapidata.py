import io
from PIL import Image
from datasets import load_dataset, Dataset, Features, Value


dataset = load_dataset("Rapidata/human-style-preferences-images", split="train")

with open("train_rapidata.txt", "r") as f:
    train_indices = [int(line.strip()) for line in f if line.strip()]

dataset = dataset.select(train_indices)
dataset = dataset.rename_columns({
    "prompt": "caption",
    "image1": "jpg_0",
    "image2": "jpg_1",
    "votes_image1": "label_0",
    "votes_image2": "label_1",
})
dataset = dataset.remove_columns(
    [col for col in dataset.column_names
     if col not in {"caption", "jpg_0", "jpg_1", "label_0", "label_1"}]
)

features = Features({
    "jpg_0": Value("binary"),
    "jpg_1": Value("binary"),
    "label_0": Value("int64"),
    "label_1": Value("int64"),
    "caption": Value("string"),
})


def normalize_to_jpeg_bytes(img: Image.Image) -> bytes:
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def gen(dataset):
    for row in dataset:
        if row["label_0"] == row["label_1"]:
            continue
        if row["label_0"] > row["label_1"]:
            l0, l1 = 1, 0
        else:
            l0, l1 = 0, 1
        yield {
            "jpg_0": normalize_to_jpeg_bytes(row["jpg_0"]),
            "jpg_1": normalize_to_jpeg_bytes(row["jpg_1"]),
            "label_0": l0,
            "label_1": l1,
            "caption": row["caption"],
        }


print("Building dataset without loading everything in RAM...")
dataset_bytes = Dataset.from_generator(lambda: gen(dataset), features=features)
dataset_bytes.save_to_disk("./processed_rapidata_dataset_bytes")
