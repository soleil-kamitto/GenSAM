"""
Experimento 1: efecto de bajar el umbral de deteccion de CellSAM.

Motivacion: el desglose de filtros sobre NRC73-A-2.5 mostro que de 186 regiones
propuestas por el modelo solo se descartan 5 (tinta), o sea que el conteo no
esta limitado por el post-filtrado sino por cuantos candidatos genera el
detector. Bajar bbox_threshold deberia aumentar esa cantidad.

El riesgo es que aparezcan falsos positivos, asi que la figura por umbral
permite revisar visualmente si lo detectado de mas son colonias reales.

Uso:
    python scripts/exp_umbral_lote2.py                      # NRC73-A-2.5
    python scripts/exp_umbral_lote2.py NRC73-B-2.5          # otra placa

Guarda:
    results/colonias/experimentos/10_umbral_lote2/<placa>_thr<valor>.png
    results/colonias/experimentos/10_umbral_lote2/resultados.csv
"""
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import regionprops

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from contar_mis_fotos import (
    detect_plate, crop_plate, flat_field, is_ink, draw_overlay,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY,
)
from cellSAM import get_model, segment_cellular_image

IMGS_DIR = Path('images/mis_fotos_lote2')
OUT_DIR = Path('results/colonias/experimentos/10_umbral_lote2')
UMBRALES = [0.40, 0.30, 0.25, 0.20]


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else 'NRC73-A-2.5-'
    ruta = IMGS_DIR / f'{stem}.jpg'
    if not ruta.exists():
        print(f'No existe {ruta}')
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Cargando modelo CellSAM...')
    model = get_model()
    print('Listo.\n')

    img = cv2.imread(str(ruta))
    cx, cy, r = detect_plate(img)
    crop, plate_mask = crop_plate(img, cx, cy, r)
    flat = flat_field(crop, plate_mask)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    print(f'{"umbral":>7} {"propuestas":>11} {"tinta":>6} {"conteo":>7}')
    print('-' * 34)

    filas = []
    for thr in UMBRALES:
        try:
            seg, _, _ = segment_cellular_image(
                cv2.cvtColor(flat, cv2.COLOR_BGR2RGB), model=model,
                normalize=True, postprocess=True,
                bbox_threshold=thr, device='cpu')
            if seg is None:
                seg = np.zeros(flat.shape[:2], dtype=np.int32)
        except (AttributeError, TypeError, ValueError):
            seg = np.zeros(flat.shape[:2], dtype=np.int32)
        seg = seg.copy()
        seg[plate_mask == 0] = 0

        props = regionprops(seg)
        base = [p for p in props
                if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                and p.solidity >= MIN_SOLIDITY]
        validas = [p for p in base if not is_ink(hsv, seg, p.label)]
        n_tinta = len(base) - len(validas)

        print(f'{thr:>7.2f} {len(props):>11} {n_tinta:>6} {len(validas):>7}')
        filas.append({'placa': stem, 'umbral': thr,
                      'propuestas': len(props), 'tinta': n_tinta,
                      'conteo': len(validas)})

        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        fig.suptitle(f'{stem}   umbral = {thr:.2f}   n = {len(validas)}',
                     fontsize=13, fontweight='bold')
        axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Placa original')
        axes[1].imshow(draw_overlay(crop, seg, validas))
        axes[1].set_title(f'Detección  n = {len(validas)}')
        for a in axes:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{stem}_thr{thr:.2f}.png', dpi=120,
                    bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(filas)
    csv = OUT_DIR / 'resultados.csv'
    if csv.exists():
        previo = pd.read_csv(csv)
        previo = previo[previo['placa'] != stem]
        df = pd.concat([previo, df], ignore_index=True)
    df.to_csv(csv, index=False)
    print(f'\nResultados en {OUT_DIR}/')


if __name__ == '__main__':
    main()
