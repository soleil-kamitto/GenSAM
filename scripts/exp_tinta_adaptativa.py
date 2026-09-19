"""
Filtro de tinta adaptativo, y su efecto sobre el conteo.

Problema detectado: el filtro actual descarta como rotulacion toda region cuyo
tono caiga entre 60 y 140. Ese rango se fijo midiendo el primer lote de fotos,
donde las colonias estaban en tono 39 a 44 y la tinta en 71 a 105. En el segundo
lote el balance de color de la camara cambio y las colonias subieron a una
mediana de 56 con percentil 90 en 66, es decir pegadas al umbral. El filtro
quedo sin margen y podria empezar a descartar colonias reales.

Solucion propuesta: en lugar de un rango fijo, medir el tono del agar en cada
fotografia y considerar tinta lo que se aparta de el. El agar ocupa la mayor
parte de la placa, asi que su tono es simplemente la mediana de la imagen dentro
del recorte. Una colonia se parece al agar en tono, aunque sea mas oscura,
mientras que la tinta azul se aparta mucho.

Compara tres variantes sobre las 15 placas con conteo manual
  sin_filtro    ninguna region se descarta por color
  fijo          el filtro actual, tono entre 60 y 140
  adaptativo    tono que se aparta del agar mas de DESVIO_TINTA grados

Uso:
    python scripts/exp_tinta_adaptativa.py

Guarda:
    results/colonias/experimentos/14_tinta_adaptativa/resultados.csv
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
    detect_plate, crop_plate, flat_field, scout,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY,
    INK_HUE_MIN, INK_HUE_MAX, INK_SAT_MIN,
)
from cellSAM import get_model, segment_cellular_image

IMGS_DIR = Path('images/mis_fotos')
GT_CSV = Path('results/ground_truth_mis_fotos/ground_truth.csv')
OUT_DIR = Path('results/colonias/experimentos/14_tinta_adaptativa')
TNTC = 250
DESVIO_TINTA = 18    # grados de tono de separacion respecto al agar
SAT_MINIMA = 40


def tono_del_agar(hsv, plate_mask):
    """El agar domina la placa, asi que su tono es la mediana dentro del disco."""
    dentro = plate_mask > 0
    return float(np.median(hsv[..., 0][dentro]))


def es_tinta_fija(h, s):
    return INK_HUE_MIN <= h <= INK_HUE_MAX and s >= INK_SAT_MIN


def es_tinta_adaptativa(h, s, tono_agar):
    return abs(h - tono_agar) > DESVIO_TINTA and s >= SAT_MINIMA


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(GT_CSV).set_index('image')['count'].to_dict()

    print('Cargando CellSAM...')
    model = get_model()
    print('Listo.\n')

    print(f'{"placa":<14} {"manual":>7} {"sin filtro":>11} {"fijo":>6} '
          f'{"adapt":>6} {"tono agar":>10}')
    print('-' * 60)

    filas = []
    for ruta in sorted(IMGS_DIR.glob('*.jpg')):
        stem = ruta.stem
        img = cv2.imread(str(ruta))
        cx, cy, r = detect_plate(img)
        crop, plate_mask = crop_plate(img, cx, cy, r)
        flat = flat_field(crop, plate_mask)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        agar = tono_del_agar(hsv, plate_mask)

        _, thr, _ = scout(flat, plate_mask)
        try:
            seg, _, _ = segment_cellular_image(
                cv2.cvtColor(flat, cv2.COLOR_BGR2RGB), model=model,
                normalize=True, postprocess=True,
                bbox_threshold=thr, device='cpu')
            if seg is None:
                seg = np.zeros(flat.shape[:2], np.int32)
        except (AttributeError, TypeError, ValueError):
            seg = np.zeros(flat.shape[:2], np.int32)
        seg = seg.copy()
        seg[plate_mask == 0] = 0

        base = [p for p in regionprops(seg)
                if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                and p.solidity >= MIN_SOLIDITY]

        n_sin, n_fijo, n_adapt = len(base), 0, 0
        for p in base:
            region = seg == p.label
            h = float(np.median(hsv[..., 0][region]))
            s = float(np.median(hsv[..., 1][region]))
            if not es_tinta_fija(h, s):
                n_fijo += 1
            if not es_tinta_adaptativa(h, s, agar):
                n_adapt += 1

        print(f'{stem:<14} {gt.get(stem, 0):>7} {n_sin:>11} {n_fijo:>6} '
              f'{n_adapt:>6} {agar:>10.1f}')
        filas.append({'placa': stem, 'manual': gt.get(stem, 0),
                      'sin_filtro': n_sin, 'fijo': n_fijo,
                      'adaptativo': n_adapt, 'tono_agar': agar})

    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / 'resultados.csv', index=False)

    contables = df[df['manual'] <= TNTC]
    print('\n' + '=' * 58)
    print(f'{"variante":<14} {"MAE contable":>13} {"MAE global":>11} {"acierto":>9}')
    print('-' * 58)
    for col in ['sin_filtro', 'fijo', 'adaptativo']:
        ec = (contables[col] - contables['manual']).abs()
        eg = (df[col] - df['manual']).abs()
        ac = 1 - eg.sum() / df['manual'].sum()
        print(f'{col:<14} {ec.mean():>13.2f} {eg.mean():>11.2f} {ac*100:>8.1f}%')
    print('=' * 58)
    print(f'\nResultados en {OUT_DIR}/')


if __name__ == '__main__':
    main()
