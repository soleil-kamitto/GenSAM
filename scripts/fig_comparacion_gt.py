"""
Genera la figura de proceso para el capitulo: el recorrido completo de una
placa por el pipeline (original, corregida, deteccion con colores) mas el
conteo manual de referencia, en un solo panel de 4 imagenes, para las 15
placas propias.

Reutiliza las funciones del pipeline en scripts/contar_mis_fotos.py, para que
el recorte y la deteccion sean identicos a los que se usaron en la corrida
final documentada en el capitulo. El panel de ground truth reutiliza la
imagen ya generada por scripts/exportar_gt_imagenes.py.

Uso:
    python scripts/fig_comparacion_gt.py            # las 15 placas
    python scripts/fig_comparacion_gt.py MC73-A      # solo una placa

Guarda:
    docs/figuras/<placa>_proceso.png
"""
import csv
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from contar_mis_fotos import (
    detect_plate, crop_plate, flat_field, scout, segment, draw_overlay,
)
from cellSAM import get_model

IMGS_DIR = Path('images/mis_fotos')
PUNTOS_DIR = Path('results/ground_truth_mis_fotos')
GT_IMG_DIR = Path('results/ground_truth_mis_fotos/imagenes')
OUT_DIR = Path('docs/figuras')


def contar_marcas(stem):
    """Total de marcas del conteo manual registradas para la placa."""
    with open(PUNTOS_DIR / f'{stem}_puntos.csv', newline='') as f:
        return sum(1 for _ in csv.DictReader(f))


def procesar(model, stem):
    img = cv2.imread(str(IMGS_DIR / f'{stem}.jpg'))
    cx, cy, r = detect_plate(img)
    crop, plate_mask = crop_plate(img, cx, cy, r)
    flat = flat_field(crop, plate_mask)
    n_adapt, thr, razon = scout(flat, plate_mask)
    seg_mask, valid, n_ink, n_rim = segment(model, flat, crop, plate_mask, thr)
    n_sistema = len(valid)
    overlay = draw_overlay(crop, seg_mask, valid)

    n_manual = contar_marcas(stem)
    # fotografia completa con las marcas, tal como la guarda la herramienta de
    # conteo: sin recortar, para no perder ninguna marca del operador
    img_gt = np.asarray(Image.open(GT_IMG_DIR / f'{stem}_gt.png'))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.5))

    axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title('1. Placa original\n(recorte, sin corregir)', fontsize=11)

    axes[1].imshow(cv2.cvtColor(flat, cv2.COLOR_BGR2RGB))
    axes[1].set_title('2. Corregida\n(flat-field + CLAHE, entrada al modelo)',
                      fontsize=11)

    axes[2].imshow(overlay)
    axes[2].set_title(f'3. Detección CellSAM\n(bbox_threshold={thr:.2f})   '
                      f'n = {n_sistema}', fontsize=11)

    axes[3].imshow(img_gt)
    axes[3].set_title(f'4. Conteo manual de referencia\n(fotografía completa)   '
                      f'n = {n_manual}', fontsize=11)

    for ax in axes:
        ax.axis('off')

    fig.suptitle(f'Placa {stem}', fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()
    salida = OUT_DIR / f'{stem}_proceso.png'
    plt.savefig(salida, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'{stem:<14} n_manual={n_manual:>4}  n_sistema={n_sistema:>4}'
          f'  -> {salida}')


def main():
    if len(sys.argv) > 1:
        placas = [sys.argv[1]]
    else:
        placas = sorted(p.name.replace('_puntos.csv', '')
                        for p in PUNTOS_DIR.glob('*_puntos.csv'))

    print(f'Cargando modelo CellSAM... ({len(placas)} placas)')
    model = get_model()

    for stem in placas:
        procesar(model, stem)

    print(f'\nListo. Figuras en {OUT_DIR}/')


if __name__ == '__main__':
    main()
