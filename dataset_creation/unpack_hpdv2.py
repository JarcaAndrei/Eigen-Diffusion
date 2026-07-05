import os
import tarfile

train_tar = "data/train.tar.gz"
test_tar  = "data/test.tar.gz"

train_dir = "data/train"
test_dir  = "data/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

with tarfile.open(train_tar, "r:gz") as tar:
    print("Extracting train...")
    tar.extractall(path=train_dir)

with tarfile.open(test_tar, "r:gz") as tar:
    print("Extracting test...")
    tar.extractall(path=test_dir)

print("Done.")