import io
import os
import json

from PIL import Image
from datasets import Dataset, Features, Value

TRAIN_JSON = "data/train/train.json"
IMAGE_DIR = "data/train/train"
corrupt_images = []

features = Features({
    "jpg_0": Value("binary"),
    "jpg_1": Value("binary"),
    "label_0": Value("int8"),
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


def load_and_normalize(path):
    try:
        return normalize_to_jpeg_bytes(Image.open(path))
    except Exception:
        corrupt_images.append(path)
        return None


def gen():
    with open(TRAIN_JSON, "r") as f:
        data = json.load(f)

    for sample in data:
        paths = sample["image_path"]
        pref  = sample["human_preference"]

        img0_path = os.path.join(IMAGE_DIR, paths[0])
        img1_path = os.path.join(IMAGE_DIR, paths[1])

        # Decide preferred image
        if pref[0] == 1:
            left_path  = img0_path
            right_path = img1_path
        else:
            left_path  = img1_path
            right_path = img0_path

        # Load images
        jpg0 = load_and_normalize(left_path)
        jpg1 = load_and_normalize(right_path)

        # Skip sample if either image failed
        if jpg0 is None or jpg1 is None:
            continue

        yield {
            "jpg_0": jpg0,   # preferred
            "jpg_1": jpg1,   # non-preferred
            "label_0": 1,
            "caption": sample["prompt"],
        }


dataset = Dataset.from_generator(gen, features=features)
dataset.save_to_disk("./hpdv2_pickapic_format")

print("Corrupt images found:", corrupt_images)
print(dataset)