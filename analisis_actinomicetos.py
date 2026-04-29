"""
Análisis de Actinomyces israelii con CellSAM.
Compara resultados con E. coli para ver diferencias morfológicas.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from cellSAM import get_model, segment_cellular_image

# ── 1. CARGAR IMAGEN ──────────────────────────────────────────────────────────
print("\n[1] Cargando imagen Actinomyces israelii...")
img_raw = imread("images/Actinomyces.israeli_0020.tif")
print(f"    Shape original: {img_raw.shape}, dtype: {img_raw.dtype}")

if img_raw.ndim == 3:
    img = (rgb2gray(img_raw) * 255).astype(np.uint8)
else:
    img = img_raw.astype(np.uint8)

img_vis = img.astype(np.float32)
img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
print(f"    Shape procesada: {img.shape}")

# ── 2. CARGAR MODELO ──────────────────────────────────────────────────────────
print("\n[2] Cargando modelo...")
model = get_model()
print("    Listo.")

# ── 3. SEGMENTACIÓN ───────────────────────────────────────────────────────────
print("\n[3] Segmentando sin postprocesamiento...")
mask, _, _ = segment_cellular_image(img, model=model, device="cpu")
n = mask.max()
print(f"    Objetos detectados: {n}")

print("\n[4] Segmentando con postprocesamiento...")
mask_post, _, _ = segment_cellular_image(img, model=model, postprocess=True, device="cpu")
n_post = mask_post.max()
print(f"    Objetos detectados (postprocesado): {n_post}")

# ── 4. MÉTRICAS ───────────────────────────────────────────────────────────────
print("\n[5] Calculando métricas...")
props = regionprops(mask, intensity_image=img)
metricas = []
for cell in props:
    metricas.append({
        "id":             cell.label,
        "area_px":        cell.area,
        "perimetro_px":   cell.perimeter,
        "excentricidad":  cell.eccentricity,
        "solidez":        cell.solidity,
        "diametro_equiv": cell.equivalent_diameter_area,
        "intensidad_media": cell.intensity_mean,
    })
df = pd.DataFrame(metricas)

print(f"\n    {'Métrica':<25} {'Media':>10} {'Min':>10} {'Max':>10}")
print("    " + "-"*57)
for col in ["area_px", "perimetro_px", "excentricidad", "solidez", "diametro_equiv"]:
    print(f"    {col:<25} {df[col].mean():>10.2f} {df[col].min():>10.2f} {df[col].max():>10.2f}")

df.to_csv("resultados_actinomicetos.csv", index=False)
print(f"\n    CSV: resultados_actinomicetos.csv")

# ── 5. ANÁLISIS COMPLETO ──────────────────────────────────────────────────────
print("\n[6] Generando analisis_actinomicetos_completo.png...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Análisis Actinomyces israelii — CellSAM", fontsize=16, fontweight="bold")

axes[0, 0].imshow(img_vis, cmap="gray")
axes[0, 0].set_title("Imagen original (A. israelii)")
axes[0, 0].axis("off")

axes[0, 1].imshow(mask, cmap="tab20")
axes[0, 1].set_title(f"Segmentación ({n} objetos)")
axes[0, 1].axis("off")

boundaries = find_boundaries(mask, mode="outer")
img_contornos = np.stack([img_vis, img_vis, img_vis], axis=-1)
img_contornos[boundaries] = [1, 0, 0]
axes[0, 2].imshow(img_contornos)
axes[0, 2].set_title("Contornos detectados")
axes[0, 2].axis("off")

axes[1, 0].hist(df["area_px"], bins=15, color="darkorange", edgecolor="white")
axes[1, 0].axvline(df["area_px"].mean(), color="red", linestyle="--",
                   label=f"Media: {df['area_px'].mean():.0f}")
axes[1, 0].set_xlabel("Área (píxeles)")
axes[1, 0].set_ylabel("N° de objetos")
axes[1, 0].set_title("Distribución de tamaños")
axes[1, 0].legend()

axes[1, 1].hist(df["excentricidad"], bins=15, color="purple", edgecolor="white")
axes[1, 1].axvline(df["excentricidad"].mean(), color="red", linestyle="--",
                   label=f"Media: {df['excentricidad'].mean():.3f}")
axes[1, 1].set_xlabel("Excentricidad (0=círculo, 1=alargado)")
axes[1, 1].set_ylabel("N° de objetos")
axes[1, 1].set_title("Distribución de forma")
axes[1, 1].legend()

sc = axes[1, 2].scatter(df["area_px"], df["excentricidad"],
                        c=df["solidez"], cmap="RdYlGn", alpha=0.7, s=50,
                        vmin=0.5, vmax=1.0)
plt.colorbar(sc, ax=axes[1, 2], label="Solidez")
axes[1, 2].set_xlabel("Área (píxeles)")
axes[1, 2].set_ylabel("Excentricidad")
axes[1, 2].set_title("Forma vs Tamaño (color=solidez)")

plt.tight_layout()
plt.savefig("analisis_actinomicetos_completo.png", dpi=150, bbox_inches="tight")
print("    Guardado: analisis_actinomicetos_completo.png")

# ── 6. COMPARACIÓN POSTPROCESAMIENTO ──────────────────────────────────────────
print("\n[7] Generando comparacion_actinomicetos_postprocesamiento.png...")
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("A. israelii — Original vs Sin/Con postprocesamiento", fontsize=13)

axes2[0].imshow(img_vis, cmap="gray")
axes2[0].set_title("Imagen original")
axes2[0].axis("off")

axes2[1].imshow(mask, cmap="tab20")
axes2[1].set_title(f"Sin postprocesamiento ({n} objetos)")
axes2[1].axis("off")

axes2[2].imshow(mask_post, cmap="tab20")
axes2[2].set_title(f"Con postprocesamiento ({n_post} objetos)")
axes2[2].axis("off")

plt.tight_layout()
plt.savefig("comparacion_actinomicetos_postprocesamiento.png", dpi=150)
print("    Guardado: comparacion_actinomicetos_postprocesamiento.png")

# ── RESUMEN COMPARATIVO ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN COMPARATIVO")
print("="*60)
print(f"{'Métrica':<30} {'A. israelii':>15} {'E. coli (ref)':>15}")
print("-"*60)
print(f"{'Objetos detectados':<30} {n:>15} {'83':>15}")
print(f"{'Área media (px²)':<30} {df['area_px'].mean():>15.1f} {'~pequeña':>15}")
print(f"{'Excentricidad media':<30} {df['excentricidad'].mean():>15.3f} {'0.967':>15}")
print(f"{'Solidez media':<30} {df['solidez'].mean():>15.3f} {'0.852':>15}")
