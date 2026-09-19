"""
Experimento 2: segmentacion por mosaico (tiling).

Motivacion: CellSAM reescala internamente la imagen de entrada a 1024x1024. Con
la placa completa, una colonia pequena queda reducida a pocos pixeles y el
detector no llega a proponerla. Si la placa se divide en cuadrantes y cada uno
se procesa por separado, esa misma colonia ocupa el doble de lado dentro de la
entrada del modelo, y deberia volverse detectable.

El precio es que hay que unir los resultados con cuidado, porque una colonia
partida por una linea de corte aparece como dos fragmentos. Se resuelve con un
solape entre cuadrantes y una fusion por cercania de centroides.

Uso:
    python scripts/exp_mosaico_lote2.py                 # NRC73-A-2.5, 2x2
    python scripts/exp_mosaico_lote2.py NRC73-B-2.5 3   # otra placa, 3x3

Guarda:
    results/colonias/experimentos/11_mosaico_lote2/<placa>_<n>x<n>.png
    results/colonias/experimentos/11_mosaico_lote2/resultados.csv
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
    detect_plate, crop_plate, flat_field, is_ink,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY,
)
from cellSAM import get_model, segment_cellular_image

IMGS_DIR = Path('images/mis_fotos_lote2')
OUT_DIR = Path('results/colonias/experimentos/11_mosaico_lote2')
UMBRAL = 0.40
SOLAPE = 0.10        # fraccion de solape entre cuadrantes vecinos
DIST_FUSION = 0.6    # fusiona centroides mas cercanos que esta fraccion del
                     # radio equivalente medio, para no contar dos veces una
                     # colonia partida por una linea de corte


def detectar_en_cuadrante(model, tile_flat, tile_original, tile_mask, thr):
    """
    Devuelve los centroides y areas de las colonias en un cuadrante.

    El filtro de tinta se aplica aqui, sobre los pixeles exactos de cada region
    y no sobre un parche alrededor del centroide, igual que en el pipeline
    principal. Muestrear un parche mete pixeles de agar en la mediana, y como
    en estas fotografias el agar tiene un tono cercano al umbral de la tinta,
    eso descartaba colonias reales de forma masiva.
    """
    try:
        seg, _, _ = segment_cellular_image(
            cv2.cvtColor(tile_flat, cv2.COLOR_BGR2RGB), model=model,
            normalize=True, postprocess=True,
            bbox_threshold=thr, device='cpu')
        if seg is None:
            return []
    except (AttributeError, TypeError, ValueError):
        return []
    seg = seg.copy()
    seg[tile_mask == 0] = 0
    hsv = cv2.cvtColor(tile_original, cv2.COLOR_BGR2HSV)
    salida = []
    for p in regionprops(seg):
        if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA \
                and p.solidity >= MIN_SOLIDITY \
                and not is_ink(hsv, seg, p.label):
            salida.append({'cy': p.centroid[0], 'cx': p.centroid[1],
                           'area': p.area})
    return salida


def fusionar(detecciones):
    """
    Une detecciones duplicadas de las zonas de solape. Dos detecciones se
    consideran la misma colonia si sus centroides estan mas cerca que
    DIST_FUSION veces el radio equivalente medio de ambas.
    """
    fusionadas = []
    for d in sorted(detecciones, key=lambda x: -x['area']):
        rd = np.sqrt(d['area'] / np.pi)
        duplicada = False
        for f in fusionadas:
            rf = np.sqrt(f['area'] / np.pi)
            dist = np.hypot(d['cx'] - f['cx'], d['cy'] - f['cy'])
            if dist < DIST_FUSION * (rd + rf):
                duplicada = True
                break
        if not duplicada:
            fusionadas.append(d)
    return fusionadas


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else 'NRC73-A-2.5-'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ruta = IMGS_DIR / f'{stem}.jpg'
    if not ruta.exists():
        print(f'No existe {ruta}')
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Cargando modelo CellSAM...')
    model = get_model()
    print(f'Listo. Mosaico {n}x{n} sobre {stem}\n')

    img = cv2.imread(str(ruta))
    cx, cy, r = detect_plate(img)
    crop, plate_mask = crop_plate(img, cx, cy, r)
    flat = flat_field(crop, plate_mask)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    alto, ancho = crop.shape[:2]

    paso_y, paso_x = alto / n, ancho / n
    margen_y, margen_x = paso_y * SOLAPE, paso_x * SOLAPE

    todas = []
    for i in range(n):
        for j in range(n):
            y0 = int(max(0, i * paso_y - margen_y))
            y1 = int(min(alto, (i + 1) * paso_y + margen_y))
            x0 = int(max(0, j * paso_x - margen_x))
            x1 = int(min(ancho, (j + 1) * paso_x + margen_x))

            tile_flat = flat[y0:y1, x0:x1]
            tile_orig = crop[y0:y1, x0:x1]
            tile_mask = plate_mask[y0:y1, x0:x1]
            if tile_mask.sum() == 0:
                continue

            dets = detectar_en_cuadrante(model, tile_flat, tile_orig,
                                         tile_mask, UMBRAL)
            for d in dets:                      # a coordenadas de la placa
                d['cy'] += y0
                d['cx'] += x0
            todas.extend(dets)
            print(f'  cuadrante ({i},{j}): {len(dets)} detecciones')

    antes = len(todas)
    todas = fusionar(todas)
    print(f'\ntotal bruto: {antes}   tras fusionar solapes: {len(todas)}')

    # el filtro de tinta ya se aplico dentro de cada cuadrante, sobre los
    # pixeles exactos de cada region
    finales = todas

    # figura
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.suptitle(f'{stem}   mosaico {n}x{n}   n = {len(finales)}',
                 fontsize=13, fontweight='bold')
    axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Placa original')
    axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if finales:
        axes[1].scatter([d['cx'] for d in finales], [d['cy'] for d in finales],
                        s=45, facecolors='none', edgecolors='red', linewidths=1.4)
    for k in range(1, n):                        # lineas de corte, de referencia
        axes[1].axhline(k * paso_y, color='cyan', lw=0.8, alpha=0.6)
        axes[1].axvline(k * paso_x, color='cyan', lw=0.8, alpha=0.6)
    axes[1].set_title(f'Detección por mosaico  n = {len(finales)}')
    for a in axes:
        a.axis('off')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{stem}_{n}x{n}.png', dpi=130, bbox_inches='tight')
    plt.close()

    fila = pd.DataFrame([{'placa': stem, 'mosaico': f'{n}x{n}',
                          'bruto': antes, 'tras_fusion': len(todas),
                          'conteo': len(finales)}])
    csv = OUT_DIR / 'resultados.csv'
    if csv.exists():
        fila = pd.concat([pd.read_csv(csv), fila], ignore_index=True)
    fila.to_csv(csv, index=False)
    print(f'\nResultados en {OUT_DIR}/')


if __name__ == '__main__':
    main()
