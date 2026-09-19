"""
Compara Cellpose contra CellSAM sobre las 15 placas con conteo manual.

Cellpose es otra familia de modelos de segmentacion celular, anterior a CellSAM
y base de Omnipose, que el capitulo ya cita. Interesa saber si rinde mejor sobre
placas fotografiadas, ya que ambos fueron entrenados con microscopia.

Se prueban dos entradas, la placa recortada tal cual y la corregida por
flat-field, porque la correccion de iluminacion resulto decisiva para CellSAM y
conviene ver si tambien lo es aqui.

Uso:
    python scripts/exp_cellpose.py

Guarda:
    results/colonias/experimentos/13_cellpose/resultados.csv
    results/colonias/experimentos/13_cellpose/<placa>_cellpose.png
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
    detect_plate, crop_plate, flat_field,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY,
)

IMGS_DIR = Path('images/mis_fotos')
GT_CSV = Path('results/ground_truth_mis_fotos/ground_truth.csv')
OUT_DIR = Path('results/colonias/experimentos/13_cellpose')
TNTC = 250
DIAMETRO = 40    # diametro tipico de colonia en el recorte de 1200 px


def contar(masks, plate_mask):
    masks = masks.copy()
    masks[plate_mask == 0] = 0
    return [p for p in regionprops(masks)
            if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
            and p.solidity >= MIN_SOLIDITY]


def main():
    from cellpose import models

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(GT_CSV).set_index('image')['count'].to_dict()

    print('Cargando Cellpose...')
    modelo = models.CellposeModel(gpu=False)
    print('Listo.\n')

    print(f'{"placa":<14} {"manual":>7} {"original":>9} {"corregida":>10}')
    print('-' * 43)

    filas = []
    for ruta in sorted(IMGS_DIR.glob('*.jpg')):
        stem = ruta.stem
        img = cv2.imread(str(ruta))
        cx, cy, r = detect_plate(img)
        crop, plate_mask = crop_plate(img, cx, cy, r)
        flat = flat_field(crop, plate_mask)

        conteos = {}
        mascaras = {}
        for nombre, entrada in [('original', crop), ('corregida', flat)]:
            try:
                gris = cv2.cvtColor(entrada, cv2.COLOR_BGR2GRAY)
                res = modelo.eval(gris, diameter=DIAMETRO)
                masks = res[0]
                validas = contar(masks, plate_mask)
                conteos[nombre] = len(validas)
                mascaras[nombre] = (masks, validas)
            except Exception as e:
                print(f'  {stem} {nombre}: fallo ({type(e).__name__}: {e})')
                conteos[nombre] = 0
                mascaras[nombre] = (np.zeros(crop.shape[:2], np.int32), [])

        print(f'{stem:<14} {gt.get(stem, 0):>7} {conteos["original"]:>9} '
              f'{conteos["corregida"]:>10}')
        filas.append({'placa': stem, 'manual': gt.get(stem, 0),
                      'cellpose_original': conteos['original'],
                      'cellpose_corregida': conteos['corregida']})

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        fig.suptitle(f'{stem}   manual = {gt.get(stem, 0)}',
                     fontsize=12, fontweight='bold')
        axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Placa')
        for k, nombre in enumerate(['original', 'corregida'], 1):
            masks, validas = mascaras[nombre]
            over = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            rng = np.random.default_rng(42)
            for p in validas:
                over[masks == p.label] = (over[masks == p.label] * 0.3
                                          + rng.uniform(0.2, 1, 3) * 0.7)
            axes[k].imshow(over)
            axes[k].set_title(f'Cellpose sobre {nombre}\nn = {conteos[nombre]}')
        for a in axes:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{stem}_cellpose.png', dpi=110,
                    bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / 'resultados.csv', index=False)

    contables = df[df['manual'] <= TNTC]
    print('\n' + '=' * 58)
    print(f'{"metodo":<22} {"MAE contable":>13} {"MAE global":>11} {"acierto":>9}')
    print('-' * 58)
    for col in ['cellpose_original', 'cellpose_corregida']:
        ec = (contables[col] - contables['manual']).abs()
        eg = (df[col] - df['manual']).abs()
        ac = 1 - eg.sum() / df['manual'].sum()
        print(f'{col:<22} {ec.mean():>13.2f} {eg.mean():>11.2f} {ac*100:>8.1f}%')
    print(f'{"cellsam (referencia)":<22} {4.08:>13.2f} {11.33:>11.2f} {80.1:>8.1f}%')
    print('=' * 58)


if __name__ == '__main__':
    main()
