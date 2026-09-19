"""
Combina mosaico y umbral bajo sobre las 15 placas con conteo manual.

Los dos experimentos anteriores, hechos sobre una placa del segundo lote,
dieron mejoras parciales y del mismo orden. El mosaico 2x2 subio el conteo de
181 a 191, y bajar el umbral de 0.40 a 0.25 lo subio a 195. Ambas estrategias
atacan el mismo cuello de botella, que es cuantas colonias propone el detector,
pero por vias distintas, asi que podrian sumarse.

Aqui se mide la combinacion sobre las placas que si tienen conteo manual, que es
donde el resultado se puede evaluar de verdad. Las placas del segundo lote
quedan intactas como prueba ciega.

Uso:
    python scripts/exp_mosaico_umbral.py

Guarda:
    results/colonias/experimentos/15_mosaico_umbral/resultados.csv
"""
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from contar_mis_fotos import (
    detect_plate, crop_plate, flat_field, is_ink,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY,
)
from cellSAM import get_model, segment_cellular_image

IMGS_DIR = Path('images/mis_fotos')
GT_CSV = Path('results/ground_truth_mis_fotos/ground_truth.csv')
OUT_DIR = Path('results/colonias/experimentos/15_mosaico_umbral')
TNTC = 250
SOLAPE = 0.10
DIST_FUSION = 0.6
UMBRAL = 0.25       # el mejor del barrido, sin llegar a 0.20
MOSAICO = 2


def detectar(model, tile_flat, tile_orig, tile_mask, thr):
    try:
        seg, _, _ = segment_cellular_image(
            cv2.cvtColor(tile_flat, cv2.COLOR_BGR2RGB), model=model,
            normalize=True, postprocess=True, bbox_threshold=thr, device='cpu')
        if seg is None:
            return []
    except (AttributeError, TypeError, ValueError):
        return []
    seg = seg.copy()
    seg[tile_mask == 0] = 0
    hsv = cv2.cvtColor(tile_orig, cv2.COLOR_BGR2HSV)
    return [{'cy': p.centroid[0], 'cx': p.centroid[1], 'area': p.area}
            for p in regionprops(seg)
            if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
            and p.solidity >= MIN_SOLIDITY
            and not is_ink(hsv, seg, p.label)]


def fusionar(dets):
    salida = []
    for d in sorted(dets, key=lambda x: -x['area']):
        rd = np.sqrt(d['area'] / np.pi)
        if not any(np.hypot(d['cx'] - f['cx'], d['cy'] - f['cy'])
                   < DIST_FUSION * (rd + np.sqrt(f['area'] / np.pi))
                   for f in salida):
            salida.append(d)
    return salida


def contar_mosaico(model, crop, flat, plate_mask, n, thr):
    alto, ancho = crop.shape[:2]
    py, px = alto / n, ancho / n
    my, mx = py * SOLAPE, px * SOLAPE
    todas = []
    for i in range(n):
        for j in range(n):
            y0, y1 = int(max(0, i * py - my)), int(min(alto, (i + 1) * py + my))
            x0, x1 = int(max(0, j * px - mx)), int(min(ancho, (j + 1) * px + mx))
            tm = plate_mask[y0:y1, x0:x1]
            if tm.sum() == 0:
                continue
            for d in detectar(model, flat[y0:y1, x0:x1], crop[y0:y1, x0:x1],
                              tm, thr):
                d['cy'] += y0
                d['cx'] += x0
                todas.append(d)
    return len(fusionar(todas))


def contar_entero(model, crop, flat, plate_mask, thr):
    return len(detectar(model, flat, crop, plate_mask, thr))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(GT_CSV).set_index('image')['count'].to_dict()

    print('Cargando CellSAM...')
    model = get_model()
    print(f'Listo. Mosaico {MOSAICO}x{MOSAICO}, umbral {UMBRAL}\n')

    print(f'{"placa":<14} {"manual":>7} {"entero040":>10} {"entero025":>10} '
          f'{"mosaico025":>11}')
    print('-' * 56)

    filas = []
    for ruta in sorted(IMGS_DIR.glob('*.jpg')):
        stem = ruta.stem
        img = cv2.imread(str(ruta))
        cx, cy, r = detect_plate(img)
        crop, plate_mask = crop_plate(img, cx, cy, r)
        flat = flat_field(crop, plate_mask)

        n_040 = contar_entero(model, crop, flat, plate_mask, 0.40)
        n_025 = contar_entero(model, crop, flat, plate_mask, UMBRAL)
        n_mos = contar_mosaico(model, crop, flat, plate_mask, MOSAICO, UMBRAL)

        print(f'{stem:<14} {gt.get(stem, 0):>7} {n_040:>10} {n_025:>10} '
              f'{n_mos:>11}')
        filas.append({'placa': stem, 'manual': gt.get(stem, 0),
                      'entero_040': n_040, 'entero_025': n_025,
                      'mosaico_025': n_mos})
        pd.DataFrame(filas).to_csv(OUT_DIR / 'resultados.csv', index=False)

    df = pd.DataFrame(filas)
    contables = df[df['manual'] <= TNTC]
    print('\n' + '=' * 58)
    print(f'{"variante":<14} {"MAE contable":>13} {"MAE global":>11} {"acierto":>9}')
    print('-' * 58)
    for col in ['entero_040', 'entero_025', 'mosaico_025']:
        ec = (contables[col] - contables['manual']).abs()
        eg = (df[col] - df['manual']).abs()
        ac = 1 - eg.sum() / df['manual'].sum()
        print(f'{col:<14} {ec.mean():>13.2f} {eg.mean():>11.2f} {ac*100:>8.1f}%')
    print('=' * 58)


if __name__ == '__main__':
    main()
