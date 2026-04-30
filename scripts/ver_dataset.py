"""
Muestra un mosaico de imágenes de cualquier carpeta del dataset.
Uso: python ver_dataset.py <nombre_dataset> <split>
Ejemplo: python ver_dataset.py bact_phase train
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

dataset_name = sys.argv[1] if len(sys.argv) > 1 else "bact_phase"
split        = sys.argv[2] if len(sys.argv) > 2 else "train"

dataset_dir = Path.home() / f".deepcell/datasets/cellsam_v1.2/{split}/{dataset_name}"

if not dataset_dir.exists():
    print(f"No existe: {dataset_dir}")
    sys.exit(1)

archivos = sorted(dataset_dir.glob("*X.npy"))
n_mostrar = min(12, len(archivos))
print(f"Dataset: {dataset_name} ({split}) — {len(archivos)} imágenes totales, mostrando {n_mostrar}")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle(f"Dataset: {dataset_name} — {split}", fontsize=16, fontweight="bold")
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i >= n_mostrar:
        ax.axis("off")
        continue

    img_raw = np.load(archivos[i])
    mask_raw = np.load(str(archivos[i]).replace("X.npy", "y.npy"))

    # Normalizar imagen para visualizar
    if img_raw.ndim == 3 and img_raw.shape[0] in [1, 2, 3]:
        img = img_raw[-1]  # último canal
    else:
        img = img_raw

    img = img.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    mask = mask_raw[0] if mask_raw.ndim == 3 else mask_raw
    n_cells = mask.max()

    ax.imshow(img, cmap="gray")
    ax.set_title(f"{archivos[i].name.split('.')[0]}\n{n_cells} células", fontsize=8)
    ax.axis("off")

plt.tight_layout()
output = f"resultados/dataset/dataset_{dataset_name}_{split}.png"
plt.savefig(output, dpi=120, bbox_inches="tight")
print(f"Guardado: {output}")
