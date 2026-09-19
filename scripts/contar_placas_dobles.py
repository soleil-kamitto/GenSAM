"""
Aplica el pipeline actual al conjunto de referencia de placas dobles.

El conjunto images/placas contiene 8 fotografias con dos placas cada una, 16 en
total, y es el que tiene conteo manual desde el principio del proyecto. Su mejor
resultado historico fue un MAE de 5.94 colonias por placa.

Desde entonces el pipeline incorporo tres mejoras, todas desarrolladas y
ajustadas sobre el otro conjunto, el de fotografias propias de una placa por
imagen. Eso convierte a este conjunto en una validacion casi independiente: si
las mejoras funcionan tambien aqui, no son un ajuste a las peculiaridades de
aquellas imagenes.

Las mejoras son
  correccion de iluminacion de campo plano, que resulto decisiva en las fotos
  con contraluz
  recorte al 92 % del radio en lugar del 86 %, elegido midiendo que fraccion de
  las colonias reales queda dentro del area analizada
  filtro de color que rechaza la rotulacion a marcador, cuya contribucion se
  midio en algo mas de cuatro puntos de acierto

La diferencia de forma con el otro conjunto es que aqui cada fotografia trae dos
placas, de modo que la deteccion clasica tiene que localizar y separar ambas
antes de segmentar. La imagen se parte por la mitad, en horizontal o en vertical
segun su orientacion, y se busca un circulo en cada mitad.

Uso:
    python scripts/contar_placas_dobles.py
    python scripts/contar_placas_dobles.py --sin-flat-field

La segunda forma desactiva la correccion de iluminacion. Sirve para comprobar si
es ella la causa del sobreconteo observado en este conjunto, ya que se diseño
para corregir el gradiente del contraluz y estas fotografias tienen iluminacion
directa, sin ese gradiente que corregir.

Guarda:
    results/colonias/placas_dobles/summary.csv
    results/colonias/placas_dobles/comparacion.csv
    results/colonias/placas_dobles/<imagen>_count.png
"""
import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from contar_mis_fotos import (
    crop_plate, flat_field, scout, segment, draw_overlay, SHRINK, SUPPORTED,
)
from cellSAM import get_model

IMGS_DIR = Path('images/placas')
GT_CSV = IMGS_DIR / 'ground_truth.csv'
OUT_DIR = Path('results/colonias/placas_dobles')


def detectar_dos_placas(img_bgr):
    """
    Localiza las dos placas de la fotografia.

    Se parte la imagen por la mitad segun su orientacion y se busca un circulo
    en cada mitad. Si Hough no encuentra nada en alguna, se asume que la placa
    ocupa el centro de esa mitad, que es lo que ocurre en la practica porque las
    fotografias estan encuadradas de forma regular.
    """
    h, w = img_bgr.shape[:2]
    vertical = h > w
    placas = []
    for i in range(2):
        if vertical:
            y0, y1 = i * h // 2, (i + 1) * h // 2
            mitad = img_bgr[y0:y1, :]
            ox, oy = 0, y0
        else:
            x0, x1 = i * w // 2, (i + 1) * w // 2
            mitad = img_bgr[:, x0:x1]
            ox, oy = x0, 0

        hh, hw = mitad.shape[:2]
        gris = cv2.cvtColor(mitad, cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(gris, (21, 21), 0)
        circulos = cv2.HoughCircles(
            suave, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(hh, hw), param1=60, param2=25,
            minRadius=int(min(hh, hw) * 0.30),
            maxRadius=int(min(hh, hw) * 0.52),
        )
        if circulos is not None:
            mejor = np.round(circulos[0][0]).astype(int)
            cx, cy, r = int(mejor[0]) + ox, int(mejor[1]) + oy, int(mejor[2])
        else:
            cx, cy = hw // 2 + ox, hh // 2 + oy
            r = int(min(hh, hw) * 0.43)
        placas.append((cx, cy, r))
    return placas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sin-flat-field', action='store_true',
                    help='no aplicar la correccion de iluminacion')
    ap.add_argument('--umbral', type=float, default=None,
                    help='umbral fijo, en vez del que elige el scout')
    args = ap.parse_args()
    usar_flat = not args.sin_flat_field

    sufijo = '' if usar_flat else '_sin_flat'
    if args.umbral is not None:
        sufijo += f'_thr{args.umbral:.2f}'.replace('.', '')
    out_dir = OUT_DIR.parent / (OUT_DIR.name + sufijo)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = {}
    if GT_CSV.exists():
        for _, fila in pd.read_csv(GT_CSV).iterrows():
            gt[Path(fila['image']).stem] = (int(fila['plate_A']),
                                            int(fila['plate_B']))

    imagenes = sorted(p for p in IMGS_DIR.iterdir()
                      if p.suffix.lower() in SUPPORTED
                      and not p.name.startswith('ground'))
    print(f'Imagenes: {len(imagenes)}  (dos placas cada una)')
    print(f'Recorte: {SHRINK} del radio, '
          f'{"con" if usar_flat else "SIN"} correccion de iluminacion\n')
    print('Cargando CellSAM...')
    model = get_model()
    print('Listo.\n')

    cab = (f'{"imagen":<18} {"A manual":>9} {"A sist":>7} {"B manual":>9} '
           f'{"B sist":>7}')
    print(cab)
    print('-' * len(cab))

    filas = []
    for ruta in imagenes:
        img = cv2.imread(str(ruta))
        placas = detectar_dos_placas(img)
        gt_a, gt_b = gt.get(ruta.stem, (None, None))

        conteos, paneles = [], []
        for i, (cx, cy, r) in enumerate(placas):
            crop, mascara = crop_plate(img, cx, cy, r)
            plano = flat_field(crop, mascara) if usar_flat else crop
            if args.umbral is not None:
                umbral = args.umbral
            else:
                _, umbral, _ = scout(plano, mascara)
            etiquetas, validas, n_tinta, n_borde = segment(
                model, plano, crop, mascara, umbral)
            conteos.append(len(validas))
            paneles.append((crop, plano, etiquetas, validas, umbral))

        ca, cb = conteos
        print(f'{ruta.stem:<18} {str(gt_a):>9} {ca:>7} {str(gt_b):>9} {cb:>7}')
        filas.append({'imagen': ruta.stem,
                      'manual_A': gt_a, 'sistema_A': ca,
                      'manual_B': gt_b, 'sistema_B': cb})

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'{ruta.name}   manual: A={gt_a}  B={gt_b}',
                     fontsize=13, fontweight='bold')
        for i, (crop, plano, etiquetas, validas, umbral) in enumerate(paneles):
            letra = 'AB'[i]
            axes[i, 0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title(f'Placa {letra}, recorte', fontsize=10)
            axes[i, 1].imshow(cv2.cvtColor(plano, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title('Corregida, entrada al modelo', fontsize=10)
            axes[i, 2].imshow(draw_overlay(crop, etiquetas, validas))
            axes[i, 2].set_title(f'Detección  n={len(validas)}  '
                                 f'umbral={umbral:.2f}', fontsize=10)
            for a in axes[i]:
                a.axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f'{ruta.stem}_count.png', dpi=110,
                    bbox_inches='tight')
        plt.close()
        pd.DataFrame(filas).to_csv(out_dir / 'summary.csv', index=False)

    df = pd.DataFrame(filas)
    df.to_csv(out_dir / 'summary.csv', index=False)

    # metricas por placa, tratando A y B como observaciones independientes
    con_gt = df.dropna(subset=['manual_A', 'manual_B'])
    if con_gt.empty:
        print('\nSin conteo manual, no se calculan metricas.')
        return

    manual = pd.concat([con_gt['manual_A'], con_gt['manual_B']],
                       ignore_index=True)
    sistema = pd.concat([con_gt['sistema_A'], con_gt['sistema_B']],
                        ignore_index=True)
    error = (sistema - manual)
    comparacion = pd.DataFrame({'manual': manual, 'sistema': sistema,
                                'error': error,
                                'abs_error': error.abs()})
    comparacion.to_csv(out_dir / 'comparacion.csv', index=False)

    mae = comparacion['abs_error'].mean()
    acierto = 1 - comparacion['abs_error'].sum() / comparacion['manual'].sum()
    sesgo = error.mean()

    print('\n' + '=' * 56)
    print(f'{"placas evaluadas":<34} {len(comparacion):>8}')
    print(f'{"MAE":<34} {mae:>8.2f} colonias/placa')
    print(f'{"sesgo medio":<34} {sesgo:>+8.2f}')
    print(f'{"acierto agregado":<34} {acierto*100:>7.1f}%')
    print('-' * 56)
    print(f'{"referencia historica de este conjunto":<34} {5.94:>8.2f} de MAE')
    print(f'{"pipeline actual en el otro conjunto":<34} {4.08:>8.2f} de MAE, '
          f'80.1% de acierto')
    print('=' * 56)
    print(f'\nResultados en {OUT_DIR}/')


if __name__ == '__main__':
    main()
