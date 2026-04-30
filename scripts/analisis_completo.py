"""
Análisis completo con CellSAM
==============================
Cubre: segmentación, conteo, métricas por célula, bounding boxes,
distribuciones, postprocesamiento, y exportación a CSV.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import find_boundaries

from cellSAM import get_model, segment_cellular_image


# ── 1. CARGAR MODELO ──────────────────────────────────────────────────────────
print("\n[1] Cargando modelo CellSAM...")
model = get_model()
print("    Modelo listo.")

# ── 2. CARGAR IMAGEN ──────────────────────────────────────────────────────────
print("\n[2] Cargando imagen...")
img_raw = imread("images/celulas.png.png")
if img_raw.ndim == 3:
    img = (rgb2gray(img_raw) * 255).astype(np.uint8)
else:
    img = img_raw
print(f"    Tamaño: {img.shape[1]}x{img.shape[0]} píxeles")

# ── 3. SEGMENTACIÓN BÁSICA ────────────────────────────────────────────────────
print("\n[3] Segmentando (sin postprocesamiento)...")
mask, embedding, bboxes = segment_cellular_image(img, model=model, device="cpu")
# mask  → array 2D donde cada célula tiene un número único (1, 2, 3, ...)
# embedding → representación interna de la imagen aprendida por el modelo
# bboxes → coordenadas [x1, y1, x2, y2] de cada célula detectada
n_cells = mask.max()
print(f"    Células detectadas: {n_cells}")

# ── 4. SEGMENTACIÓN CON POSTPROCESAMIENTO ─────────────────────────────────────
print("\n[4] Segmentando (con postprocesamiento)...")
mask_post, _, _ = segment_cellular_image(
    img, model=model, postprocess=True, device="cpu"
)
# postprocess=True aplica:
#   - binary_opening/closing para suavizar bordes
#   - filtro gaussiano para bordes más suaves
#   - remove_small_regions para eliminar ruido
n_cells_post = mask_post.max()
print(f"    Células detectadas (postprocesado): {n_cells_post}")

# ── 5. MÉTRICAS POR CÉLULA ────────────────────────────────────────────────────
print("\n[5] Calculando métricas por célula...")
# regionprops analiza cada región (célula) de la máscara por separado
props = regionprops(mask, intensity_image=img)

metricas = []
for cell in props:
    metricas.append({
        "id":               cell.label,
        "area_px":          cell.area,                    # área en píxeles
        "perimetro_px":     cell.perimeter,               # perímetro en píxeles
        "excentricidad":    cell.eccentricity,            # 0=círculo, 1=línea
        "solidez":          cell.solidity,                # área/área_convexa (1=sólido)
        "diametro_equiv":   cell.equivalent_diameter_area,# diámetro si fuera círculo
        "intensidad_media": cell.intensity_mean,          # brillo medio dentro
        "intensidad_min":   cell.intensity_min,
        "intensidad_max":   cell.intensity_max,
        "centroide_x":      cell.centroid[1],
        "centroide_y":      cell.centroid[0],
        "bbox_x1":          cell.bbox[1],
        "bbox_y1":          cell.bbox[0],
        "bbox_x2":          cell.bbox[3],
        "bbox_y2":          cell.bbox[2],
    })

df = pd.DataFrame(metricas)

# Estadísticas resumen
print(f"\n    {'Métrica':<25} {'Media':>10} {'Min':>10} {'Max':>10}")
print("    " + "-" * 57)
for col in ["area_px", "perimetro_px", "excentricidad", "solidez", "diametro_equiv"]:
    print(f"    {col:<25} {df[col].mean():>10.2f} {df[col].min():>10.2f} {df[col].max():>10.2f}")

# ── 6. EXPORTAR A CSV ─────────────────────────────────────────────────────────
csv_path = "resultados/celulas/resultados_celulas.csv"
df.to_csv(csv_path, index=False)
print(f"\n[6] Métricas exportadas a: {csv_path}")

# ── 7. VISUALIZACIONES ────────────────────────────────────────────────────────
print("\n[7] Generando visualizaciones...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Análisis CellSAM — Resultado Completo", fontsize=16, fontweight="bold")

# 7a. Imagen original
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Imagen original")
axes[0, 0].axis("off")

# 7b. Máscara de segmentación
axes[0, 1].imshow(mask, cmap="tab20")
axes[0, 1].set_title(f"Segmentación ({n_cells} células)")
axes[0, 1].axis("off")

# 7c. Contornos sobre la imagen original
boundaries = find_boundaries(mask, mode="outer")
img_contornos = np.stack([img, img, img], axis=-1)  # RGB
img_contornos[boundaries] = [255, 0, 0]             # contornos en rojo
axes[0, 2].imshow(img_contornos)
axes[0, 2].set_title("Contornos de células")
axes[0, 2].axis("off")

# 7d. Bounding boxes sobre la imagen
axes[1, 0].imshow(img, cmap="gray")
axes[1, 0].set_title("Bounding boxes")
axes[1, 0].axis("off")
for _, row in df.iterrows():
    w = row["bbox_x2"] - row["bbox_x1"]
    h = row["bbox_y2"] - row["bbox_y1"]
    rect = mpatches.Rectangle(
        (row["bbox_x1"], row["bbox_y1"]), w, h,
        linewidth=0.8, edgecolor="lime", facecolor="none"
    )
    axes[1, 0].add_patch(rect)

# 7e. Distribución de áreas
axes[1, 1].hist(df["area_px"], bins=20, color="steelblue", edgecolor="white")
axes[1, 1].axvline(df["area_px"].mean(), color="red", linestyle="--", label=f"Media: {df['area_px'].mean():.0f}")
axes[1, 1].set_xlabel("Área (píxeles)")
axes[1, 1].set_ylabel("N° de células")
axes[1, 1].set_title("Distribución de tamaños")
axes[1, 1].legend()

# 7f. Excentricidad vs Área (forma de las células)
sc = axes[1, 2].scatter(
    df["area_px"], df["excentricidad"],
    c=df["intensidad_media"], cmap="plasma", alpha=0.7, s=40
)
plt.colorbar(sc, ax=axes[1, 2], label="Intensidad media")
axes[1, 2].set_xlabel("Área (píxeles)")
axes[1, 2].set_ylabel("Excentricidad (0=círculo, 1=alargada)")
axes[1, 2].set_title("Forma vs Tamaño")

plt.tight_layout()
plt.savefig("resultados/celulas/analisis_completo.png", dpi=150, bbox_inches="tight")
print("    Guardado: resultados/celulas/analisis_completo.png")

# ── 8. COMPARACIÓN CON/SIN POSTPROCESAMIENTO ─────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Comparación: con vs sin postprocesamiento", fontsize=13)

axes2[0].imshow(mask, cmap="tab20")
axes2[0].set_title(f"Sin postprocesamiento ({n_cells} células)")
axes2[0].axis("off")

axes2[1].imshow(mask_post, cmap="tab20")
axes2[1].set_title(f"Con postprocesamiento ({n_cells_post} células)")
axes2[1].axis("off")

plt.tight_layout()
plt.savefig("resultados/celulas/comparacion_postprocesamiento.png", dpi=150)
print("    Guardado: resultados/celulas/comparacion_postprocesamiento.png")

print("\n" + "=" * 50)
print("RESUMEN FINAL")
print("=" * 50)
print(f"  Células detectadas:        {n_cells}")
print(f"  Área promedio:             {df['area_px'].mean():.1f} px²")
print(f"  Diámetro equivalente med:  {df['diametro_equiv'].mean():.1f} px")
print(f"  Excentricidad media:       {df['excentricidad'].mean():.3f}  (0=círculo)")
print(f"  Solidez media:             {df['solidez'].mean():.3f}  (1=sin huecos)")
print(f"  Archivos generados:")
print(f"    - resultados/celulas/analisis_completo.png")
print(f"    - resultados/celulas/comparacion_postprocesamiento.png")
print(f"    - resultados/celulas/resultados_celulas.csv")
