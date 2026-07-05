import io
from PIL import Image
from datasets import load_dataset, Dataset, Features, Value

ASPECT_MIN = 0.7
ASPECT_MAX = 1.35
TARGET_SIZE = (512, 512)

features = Features({
    "jpg_0": Value("binary"),
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


def encode_jpg0(img: Image.Image) -> bytes:
    return normalize_to_jpeg_bytes(img.resize(TARGET_SIZE, Image.BICUBIC))


def keep_long_captions(example):
    return len(example["caption"].strip().split()) > 5


def main():
    orig = load_dataset("mdwiratathya/ROCO-radiology", split="train")
    orig = orig.filter(keep_long_captions)

    def generator():
        for o in orig:
            img = o["image"]
            w, h = img.size
            if not (ASPECT_MIN <= w / h <= ASPECT_MAX):
                continue
            yield {
                "jpg_0": encode_jpg0(img),
                "label_0": 1,
                "caption": o["caption"],
            }

    ds = Dataset.from_generator(generator, features=features)
    ds.save_to_disk("roco_dataset")
    print(f"Saved {len(ds)} ROCO SFT examples to roco_dataset")


if __name__ == "__main__":
    main()
