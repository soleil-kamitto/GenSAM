"""
Script de prueba: descarga el modelo CellSAM y segmenta una imagen de ejemplo.
La imagen de ejemplo viene de skimage (no requiere S3 ni datos externos).
El resultado se guarda como 'resultado_segmentacion.png'.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # sin ventana de GUI
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray

from cellSAM import get_model, segment_cellular_image


def main():
    print("=" * 50)
    print("1. Cargando modelo CellSAM...")
    print("   (si es la primera vez, descarga ~varios GB de ~/.deepcell)")
    print("=" * 50)
    model = get_model()
    print("   Modelo cargado.\n")

    print("2. Cargando imagen real de celulas...")
    img_raw = imread("images/celulas.png.png")
    if img_raw.ndim == 3:
        img = (rgb2gray(img_raw) * 255).astype(np.uint8)
    else:
        img = img_raw
    print(f"   Forma de la imagen: {img.shape}\n")

    print("3. Segmentando...")
    mask, _, bboxes = segment_cellular_image(img, model=model, device="cpu")
    n_cells = mask.max()
    print(f"   Segmentacion completada. Celulas detectadas: {n_cells}\n")

    print("4. Guardando resultado en 'resultado_segmentacion.png'...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="tab20")
    axes[1].set_title(f"Mascara CellSAM ({n_cells} celulas)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("resultados/celulas/resultado_segmentacion.png", dpi=150)
    print("   Guardado: resultados/celulas/resultado_segmentacion.png")
    print("\nListo!")


if __name__ == "__main__":
    main()
