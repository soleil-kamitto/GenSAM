import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from cellSAM import get_model, segment_cellular_image

# Cargar imagen y máscara real (ground truth)
ecoli_dir = Path.home() / ".deepcell/datasets/cellsam_v1.2/train/2c_e_coli"
img_path  = ecoli_dir / "pos0_frame_56.b0.X.npy"
mask_path = ecoli_dir / "pos0_frame_56.b0.y.npy"

img_raw  = np.load(img_path)   # shape (3, H, W) — 3 canales
mask_gt  = np.load(mask_path)  # shape (1, H, W) — máscara real

print(f"Imagen shape: {img_raw.shape}")
print(f"Máscara GT shape: {mask_gt.shape}")

# El modelo espera (H, W) o (H, W, C) — convertimos
img = img_raw[2]  # canal 2 (el que usa CellSAM internamente para bacterias)
print(f"Imagen canal usado: {img.shape}, min={img.min():.2f}, max={img.max():.2f}")

# Cargar modelo y segmentar
print("\nCargando modelo...")
model = get_model()

print("Segmentando...")
mask_pred, _, _ = segment_cellular_image(img, model=model, device="cpu")
mask_real = mask_gt[0]  # quitar dim de canal

n_pred = mask_pred.max()
n_real = mask_real.max()
print(f"Células detectadas por CellSAM:  {n_pred}")
print(f"Células reales (ground truth):   {n_real}")

# Visualización: imagen | ground truth | predicción
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("E. coli — CellSAM vs Ground Truth", fontsize=14, fontweight="bold")

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Imagen original (E. coli)")
axes[0].axis("off")

axes[1].imshow(mask_real, cmap="tab20")
axes[1].set_title(f"Ground Truth ({n_real} células reales)")
axes[1].axis("off")

axes[2].imshow(mask_pred, cmap="tab20")
axes[2].set_title(f"CellSAM predijo ({n_pred} células)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("resultado_ecoli.png", dpi=150)
print("\nGuardado: resultado_ecoli.png")
