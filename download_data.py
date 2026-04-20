"""
download_data.py
================
Downloads a small working subset of the PlantVillage dataset
directly — no Kaggle account needed.

Uses the publicly available subset from GitHub.
Downloads ~250 images across 5 disease classes.

Run: python download_data.py
"""

import os
import json
import urllib.request

CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___healthy",
    "Potato___Early_blight",
]

IMAGES_PER_CLASS_TRAIN = 40
IMAGES_PER_CLASS_VAL   = 10

GITHUB_API_BASE = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"

def get_image_list(class_name):
    url = f"{GITHUB_API_BASE}/{class_name}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        files = json.loads(r.read())
    return [f for f in files if f["name"].lower().endswith((".jpg", ".jpeg", ".png"))]

def download_file(download_url, dest_path):
    req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        with open(dest_path, "wb") as f:
            f.write(r.read())

def make_synthetic(class_name, train_dir, val_dir, n_train, n_val):
    from PIL import Image, ImageDraw
    import random

    base_colors = {
        "healthy":      (60, 180, 75),
        "Early_blight": (200, 130, 40),
        "Late_blight":  (150, 50,  50),
    }
    key   = class_name.split("___")[-1]
    color = base_colors.get(key, (120, 120, 120))

    for directory, n in [(train_dir, n_train), (val_dir, n_val)]:
        for i in range(n):
            img  = Image.new("RGB", (224, 224), color)
            draw = ImageDraw.Draw(img)
            for _ in range(random.randint(8, 20)):
                x, y = random.randint(0, 224), random.randint(0, 224)
                r    = random.randint(4, 28)
                c2   = tuple(max(0, min(255, c + random.randint(-50, 50))) for c in color)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=c2)
            img.save(os.path.join(directory, f"img_{i:04d}.jpg"), "JPEG")


def main():
    print("=" * 55)
    print("  PlantScan AI — Dataset Setup")
    print("=" * 55)

    for class_name in CLASSES:
        train_dir = os.path.join("data", "train", class_name)
        val_dir   = os.path.join("data", "valid", class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir,   exist_ok=True)

        print(f"\n  [{class_name}]")
        try:
            print("    Fetching file list from GitHub...")
            image_files = get_image_list(class_name)
            needed      = IMAGES_PER_CLASS_TRAIN + IMAGES_PER_CLASS_VAL
            image_files = image_files[:needed]

            for i, finfo in enumerate(image_files):
                dest = os.path.join(
                    train_dir if i < IMAGES_PER_CLASS_TRAIN else val_dir,
                    finfo["name"]
                )
                if os.path.exists(dest):
                    print(f"    Skipping {finfo['name']} (already downloaded)", end="\r")
                    continue
                print(f"    Downloading image {i+1}/{len(image_files)}...    ", end="\r")
                try:
                    download_file(finfo["download_url"], dest)
                except Exception as e:
                    print(f"\n    Skipped {finfo['name']}: {e}")

            t = len(os.listdir(train_dir))
            v = len(os.listdir(val_dir))
            print(f"    ✅ Downloaded — train: {t}  val: {v}            ")

        except Exception as e:
            print(f"    ⚠️  GitHub download failed: {e}")
            print("    Generating synthetic placeholder images instead...")
            make_synthetic(class_name, train_dir, val_dir,
                           IMAGES_PER_CLASS_TRAIN, IMAGES_PER_CLASS_VAL)
            print(f"    ✅ Synthetic images created")

    print("\n" + "=" * 55)
    total_t = sum(
        len(os.listdir(os.path.join("data","train",c))) for c in CLASSES
    )
    total_v = sum(
        len(os.listdir(os.path.join("data","valid",c))) for c in CLASSES
    )
    print(f"  Total training   : {total_t} images")
    print(f"  Total validation : {total_v} images")
    print("\n  ✅ Done! Now run:  python train.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
