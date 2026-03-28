"""
download_data.py — Auto-downloads MovieLens 100K dataset
Run this once before train.py and app.py
"""

import urllib.request, zipfile, os
from pathlib import Path

URL      = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path("data")
DEST_ZIP = DATA_DIR / "ml-100k.zip"
DEST_DIR = DATA_DIR / "ml-100k"

def download():
    DATA_DIR.mkdir(exist_ok=True)

    if DEST_DIR.exists() and (DEST_DIR / "u.data").exists():
        print("✅ Dataset already present at data/ml-100k/ — nothing to do.")
        return

    print(f"⬇  Downloading MovieLens 100K from:\n   {URL}")
    print("   (size ~5 MB, should take a few seconds)\n")

    def progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r   [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(URL, DEST_ZIP, reporthook=progress)
    print("\n\n📦 Extracting...")

    with zipfile.ZipFile(DEST_ZIP, "r") as z:
        z.extractall(DATA_DIR)

    DEST_ZIP.unlink()  # clean up zip

    files = list(DEST_DIR.iterdir())
    print(f"✅ Extracted {len(files)} files to {DEST_DIR}/\n")
    print("Key files:")
    for fname in ["u.data", "u.item", "u.user", "u.genre"]:
        fp = DEST_DIR / fname
        if fp.exists():
            print(f"   {fname:15s}  ({fp.stat().st_size / 1024:.0f} KB)")

    print("\n🚀 Now run:  python train.py")

if __name__ == "__main__":
    download()
