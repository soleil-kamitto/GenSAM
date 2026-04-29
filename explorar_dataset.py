import os
from pathlib import Path

dataset_root = Path.home() / ".deepcell/datasets/cellsam_v1.2"

for split in ["train", "val", "test"]:
    split_path = dataset_root / split
    if not split_path.exists():
        print(f"\n[{split}] no encontrado")
        continue

    print(f"\n{'='*50}")
    print(f"  SPLIT: {split.upper()}")
    print(f"{'='*50}")

    total = 0
    for folder in sorted(split_path.iterdir()):
        if not folder.is_dir():
            continue
        count = len(list(folder.glob("*X.npy")))
        total += count
        print(f"  {folder.name:<35} {count:>5} imágenes")

    print(f"  {'TOTAL':<35} {total:>5} imágenes")
