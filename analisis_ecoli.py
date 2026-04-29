"""
Análisis completo de E. coli — igual que analisis_completo.py
pero usando imágenes reales del dataset con ground truth incluido.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from cellSAM import get_model, segment_cellular_image

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
ecoli_dir = Path.home() / ".deepcell/datasets/cellsam_v1.2/train/2c_e_coli"
img_path  = ecoli_dir / "pos0_frame_56.b0.X.npy"
mask_gt_path = ecoli_dir / "pos0_frame_56.b0.y.npy"

# ── 1. CARGAR DATOS ───────────────────────────────────────────────────────────
print("\n[1] Cargando imagen E. coli del dataset...")
img_raw  = np.load(img_path)    # shape (3, H, W)
mask_gt  = np.load(mask_gt_path)[0]  # shape (H, W)

img = img_raw[-1].astype(np.float32)  # canal que usa CellSAM
if img.max() > img.min():
    img_vis = (img - img.min()) / (img.max() - img.min())
else:
    img_vis = img
img_uint8 = (img_vis * 255).astype(np.uint8)

print(f"    Shape imagen: {img_raw.shape}")
print(f"    Bacterias reales (GT): {mask_gt.max()}")

# ── 2. CARGAR MODELO ──────────────────────────────────────────────────────────
print("\n[2] Cargando modelo CellSAM...")
model = get_model()
print("    Modelo listo.")

# ── 3. SEGMENTACIÓN SIN POSTPROCESAMIENTO ─────────────────────────────────────
print("\n[3] Segmentando sin postprocesamiento...")
mask_pred, _, bboxes = segment_cellular_image(img_uint8, model=model, device="cpu")
n_pred = mask_pred.max()
n_real = mask_gt.max()
print(f"    CellSAM detectó: {n_pred} bacterias")
print(f"    Ground truth:    {n_real} bacterias")

# ── 4. SEGMENTACIÓN CON POSTPROCESAMIENTO ─────────────────────────────────────
print("\n[4] Segmentando con postprocesamiento...")
mask_post, _, _ = segment_cellular_image(img_uint8, model=model, postprocess=True, device="cpu")
n_post = mask_post.max()
print(f"    CellSAM (postprocesado): {n_post} bacterias")

# ── 5. MÉTRICAS POR BACTERIA ──────────────────────────────────────────────────
print("\n[5] Calculando métricas...")
props = regionprops(mask_pred, intensity_image=img_uint8)
metricas = []
for cell in props:
    metricas.append({
        "id":               cell.label,
        "area_px":          cell.area,
        "perimetro_px":     cell.perimeter,
        "excentricidad":    cell.eccentricity,
        "solidez":          cell.solidity,
        "diametro_equiv":   cell.equivalent_diameter_area,
        "intensidad_media": cell.intensity_mean,
        "centroide_x":      cell.centroid[1],
        "centroide_y":      cell.centroid[0],
        "bbox_x1":          cell.bbox[1],
        "bbox_y1":          cell.bbox[0],
        "bbox_x2":          cell.bbox[3],
        "bbox_y2":          cell.bbox[2],
    })

df = pd.DataFrame(metricas)

print(f"\n    {'Métrica':<25} {'Media':>10} {'Min':>10} {'Max':>10}")
print("    " + "-"*57)
for col in ["area_px", "perimetro_px", "excentricidad", "solidez", "diametro_equiv"]:
    print(f"    {col:<25} {df[col].mean():>10.2f} {df[col].min():>10.2f} {df[col].max():>10.2f}")

df.to_csv("resultados_ecoli.csv", index=False)
print(f"\n    CSV guardado: resultados_ecoli.csv")

# ── 6. ANÁLISIS COMPLETO (6 paneles) ─────────────────────────────────────────
print("\n[6] Generando analisis_ecoli_completo.png...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Análisis E. coli — CellSAM", fontsize=16, fontweight="bold")

# Imagen original
axes[0, 0].imshow(img_vis, cmap="gray")
axes[0, 0].set_title("Imagen original (E. coli)")
axes[0, 0].axis("off")

# Predicción CellSAM
axes[0, 1].imshow(mask_pred, cmap="tab20")
axes[0, 1].set_title(f"CellSAM predijo ({n_pred} bacterias)")
axes[0, 1].axis("off")

# Ground truth
axes[0, 2].imshow(mask_gt, cmap="tab20")
axes[0, 2].set_title(f"Ground Truth ({n_real} bacterias reales)")
axes[0, 2].axis("off")

# Contornos sobre imagen
boundaries = find_boundaries(mask_pred, mode="outer")
img_contornos = np.stack([img_vis, img_vis, img_vis], axis=-1)
img_contornos[boundaries] = [1, 0, 0]
axes[1, 0].imshow(img_contornos)
axes[1, 0].set_title("Contornos detectados")
axes[1, 0].axis("off")

# Distribución de áreas
axes[1, 1].hist(df["area_px"], bins=15, color="steelblue", edgecolor="white")
axes[1, 1].axvline(df["area_px"].mean(), color="red", linestyle="--",
                   label=f"Media: {df['area_px'].mean():.0f}")
axes[1, 1].set_xlabel("Área (píxeles)")
axes[1, 1].set_ylabel("N° de bacterias")
axes[1, 1].set_title("Distribución de tamaños")
axes[1, 1].legend()

# Excentricidad vs Área
sc = axes[1, 2].scatter(df["area_px"], df["excentricidad"],
                        c=df["intensidad_media"], cmap="plasma", alpha=0.7, s=50)
plt.colorbar(sc, ax=axes[1, 2], label="Intensidad media")
axes[1, 2].set_xlabel("Área (píxeles)")
axes[1, 2].set_ylabel("Excentricidad (0=círculo, 1=alargada)")
axes[1, 2].set_title("Forma vs Tamaño")

plt.tight_layout()
plt.savefig("analisis_ecoli_completo.png", dpi=150, bbox_inches="tight")
print("    Guardado: analisis_ecoli_completo.png")

# ── 7. COMPARACIÓN POSTPROCESAMIENTO ──────────────────────────────────────────
print("\n[7] Generando comparacion_ecoli_postprocesamiento.png...")
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("E. coli — Ground Truth vs Sin/Con postprocesamiento", fontsize=13)

axes2[0].imshow(mask_gt, cmap="tab20")
axes2[0].set_title(f"Ground Truth ({n_real} bacterias)")
axes2[0].axis("off")

axes2[1].imshow(mask_pred, cmap="tab20")
axes2[1].set_title(f"Sin postprocesamiento ({n_pred} bacterias)")
axes2[1].axis("off")

axes2[2].imshow(mask_post, cmap="tab20")
axes2[2].set_title(f"Con postprocesamiento ({n_post} bacterias)")
axes2[2].axis("off")

plt.tight_layout()
plt.savefig("comparacion_ecoli_postprocesamiento.png", dpi=150)
print("    Guardado: comparacion_ecoli_postprocesamiento.png")

# ── RESUMEN ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("RESUMEN E. coli")
print("="*50)
print(f"  Bacterias reales (GT):        {n_real}")
print(f"  CellSAM sin postprocesar:     {n_pred}")
print(f"  CellSAM con postprocesar:     {n_post}")
print(f"  Área promedio:                {df['area_px'].mean():.1f} px²")
print(f"  Diámetro equivalente medio:   {df['diametro_equiv'].mean():.1f} px")
print(f"  Excentricidad media:          {df['excentricidad'].mean():.3f}  (0=círculo)")
print(f"  Solidez media:                {df['solidez'].mean():.3f}  (1=sin huecos)")
