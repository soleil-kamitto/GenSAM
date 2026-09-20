"""
Conteo por mosaico, para recuperar las colonias que la reduccion de escala borra.

Problema. CellSAM redimensiona internamente cualquier entrada a 1024 x 1024. Una
placa se fotografia entera, asi que en una con muchas colonias pequenas cada una
llega al modelo reducida a unos pocos pixeles, por debajo de lo que el detector
resuelve. El efecto no se arregla subiendo la resolucion de entrada, porque el
modelo la vuelve a bajar; hay que cambiar lo que se le entrega.

Solucion. Se parte el disco en baldosas solapadas y se segmenta cada una por
separado. Una baldosa cubre una fraccion del disco, de modo que al escalarla a
1024 las colonias conservan su tamano aparente o incluso lo ganan. Es la misma
idea que la inferencia por rebanadas descrita por Akyon y colaboradores en 2022
para deteccion de objetos pequenos, aplicada aqui a segmentacion de colonias.

Tres detalles que hay que cuidar, y que son donde suele fallar este esquema.

1. Una colonia partida por el corte se veria dos veces, una mitad en cada
   baldosa. Por eso las baldosas se solapan, y una deteccion que toca un corte
   interior se descarta: la baldosa vecina la contiene entera y la aporta ella.
   Los cortes que coinciden con el borde de la placa no cuentan como interiores,
   porque alli no hay vecina que aporte nada.

2. Aun con lo anterior, la zona de solape puede dar dos detecciones de la misma
   colonia. Se fusionan las que caen mas cerca que una fraccion de sus radios,
   con el mismo criterio que usa el conteo por consenso.

3. Los filtros de area estan calibrados sobre el recorte reducido a 1200 px. Al
   trabajar con mas resolucion las areas crecen con el cuadrado de la escala, asi
   que los limites se reescalan en la misma proporcion. Olvidarlo descartaria
   como demasiado grandes a casi todas las colonias.

Uso:
    python scripts/contar_mosaico.py images/mis_fotos_lote3
    python scripts/contar_mosaico.py images/mis_fotos_lote3 --rejilla 3
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
from skimage.measure import regionprops

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from contar_mis_fotos import (
    detect_plate, flat_field, SHRINK,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY, SUPPORTED,
)
from autocalibrar import calibrar, es_tinta
from quitar_rotulacion import limpiar_rotulacion
from cellSAM import get_model, segment_cellular_image

DIM_MOSAICO = 2400    # lado del recorte con el que se trabaja, en pixeles
SOLAPE = 0.25         # fraccion de baldosa que comparte con su vecina
MARGEN_CORTE = 6      # pixeles desde el corte interior para darlo por tocado
DIST_MISMA = 0.6      # fraccion del radio para fusionar dos detecciones


def recortar(img_bgr, cx, cy, r, dim, shrink=SHRINK):
    """Recorte circular de la placa, llevado a una dimension mayor que la usual."""
    rr = int(r * shrink)
    x0, y0 = max(0, cx - rr), max(0, cy - rr)
    x1, y1 = min(img_bgr.shape[1], cx + rr), min(img_bgr.shape[0], cy + rr)
    crop = img_bgr[y0:y1, x0:x1].copy()

    mascara = np.zeros(crop.shape[:2], np.uint8)
    cv2.circle(mascara, (cx - x0, cy - y0), rr, 255, -1)

    largo = max(crop.shape[:2])
    if largo != dim:
        escala = dim / largo
        nuevo = (int(crop.shape[1] * escala), int(crop.shape[0] * escala))
        crop = cv2.resize(crop, nuevo, interpolation=cv2.INTER_AREA)
        mascara = cv2.resize(mascara, nuevo, interpolation=cv2.INTER_NEAREST)
    crop[mascara == 0] = 0
    return crop, mascara


def cortes(largo, n, solape):
    """Posiciones de inicio y fin de cada baldosa en un eje."""
    if n <= 1:
        return [(0, largo)]
    paso = largo / (n - (n - 1) * solape)
    avance = paso * (1 - solape)
    tramos = []
    for i in range(n):
        a = int(round(i * avance))
        b = int(round(a + paso))
        if i == n - 1:
            a, b = largo - int(round(paso)), largo
        tramos.append((max(0, a), min(largo, b)))
    return tramos


def detectar_baldosa(model, entrada, original, mascara, umbral, hue_agar,
                     sat_minima, lim_area, en_borde_placa):
    """
    Detecciones de una baldosa, en coordenadas de la baldosa.

    Se descarta lo que toca un corte interior, porque la baldosa vecina contiene
    esa colonia entera y la aportara ella. `en_borde_placa` dice, para cada lado,
    si ese corte coincide con el limite del recorte, en cuyo caso no es interior
    y lo que lo toca se conserva.
    """
    try:
        seg, _, _ = segment_cellular_image(
            cv2.cvtColor(entrada, cv2.COLOR_BGR2RGB), model=model,
            normalize=True, postprocess=True,
            bbox_threshold=umbral, device='cpu')
        if seg is None:
            return []
    except (AttributeError, TypeError, ValueError):
        return []
    seg = seg.copy()
    seg[mascara == 0] = 0

    alto, ancho = seg.shape
    arriba, abajo, izq, der = en_borde_placa
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    salida = []
    for p in regionprops(seg):
        if not (lim_area[0] <= p.area <= lim_area[1]):
            continue
        if p.solidity < MIN_SOLIDITY:
            continue
        y0, x0, y1, x1 = p.bbox
        if (not arriba and y0 <= MARGEN_CORTE) or \
           (not abajo and y1 >= alto - MARGEN_CORTE) or \
           (not izq and x0 <= MARGEN_CORTE) or \
           (not der and x1 >= ancho - MARGEN_CORTE):
            continue
        if es_tinta(hsv, seg, p.label, hue_agar, sat_minima):
            continue
        salida.append({'cy': p.centroid[0], 'cx': p.centroid[1],
                       'area': float(p.area)})
    return salida


def fusionar(detecciones):
    """Une las detecciones repetidas en la zona de solape."""
    grupos = []
    for d in sorted(detecciones, key=lambda x: -x['area']):
        rd = np.sqrt(d['area'] / np.pi)
        unida = False
        for g in grupos:
            rg = np.sqrt(g['area'] / np.pi)
            if np.hypot(d['cx'] - g['cx'], d['cy'] - g['cy']) \
                    < DIST_MISMA * (rd + rg):
                unida = True
                break
        if not unida:
            grupos.append(dict(d))
    return grupos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('imagenes', nargs='?', default='images/mis_fotos_lote3')
    ap.add_argument('--rejilla', type=int, default=2,
                    help='baldosas por lado')
    ap.add_argument('--dim', type=int, default=DIM_MOSAICO)
    args = ap.parse_args()

    origen = Path(args.imagenes)
    salida = Path('results/colonias') / f'{origen.name}_mosaico{args.rejilla}'
    salida.mkdir(parents=True, exist_ok=True)

    rutas = sorted(p for p in origen.iterdir()
                   if p.suffix.lower() in SUPPORTED
                   and not p.name.startswith('ground'))

    # los filtros de area se calibraron sobre recortes de 1200 px; al trabajar
    # con mas resolucion el area de una misma colonia crece con el cuadrado de
    # la escala, y ademas cada baldosa se amplia respecto del disco completo
    escala = args.dim / 1200.0
    lim_area = (MIN_COLONY_AREA * escala ** 2, MAX_COLONY_AREA * escala ** 2)

    print(f'Imagenes: {len(rutas)}   salida: {salida}')
    print(f'Recorte a {args.dim} px, rejilla {args.rejilla}x{args.rejilla}, '
          f'solape {SOLAPE:.0%}')
    print(f'Area valida reescalada: {lim_area[0]:.0f} a {lim_area[1]:.0f} px\n')
    print('Cargando CellSAM...')
    model = get_model()
    print('Listo.\n')

    cab = (f'{"placa":<12} {"umbral":>7} {"baldosas":>9} {"brutas":>7} '
           f'{"fusion":>7} {"final":>6}')
    print(cab)
    print('-' * len(cab))

    filas = []
    for ruta in rutas:
        img = cv2.imread(str(ruta))
        px, py, r = detect_plate(img)
        crop, mascara = recortar(img, px, py, r, args.dim)
        par = calibrar(crop, mascara)
        limpio, _, mascara = limpiar_rotulacion(crop, mascara, par['hue_agar'])
        base = flat_field(limpio, mascara) if par['aplicar_flat'] else limpio

        alto, ancho = base.shape[:2]
        filas_y = cortes(alto, args.rejilla, SOLAPE)
        filas_x = cortes(ancho, args.rejilla, SOLAPE)

        brutas = []
        usadas = 0
        for i, (y0, y1) in enumerate(filas_y):
            for j, (x0, x1) in enumerate(filas_x):
                sub_m = mascara[y0:y1, x0:x1]
                if (sub_m > 0).sum() < 500:
                    continue        # baldosa casi toda fuera del disco
                usadas += 1
                sub = base[y0:y1, x0:x1]
                sub_o = limpio[y0:y1, x0:x1]
                en_borde = (i == 0, i == len(filas_y) - 1,
                            j == 0, j == len(filas_x) - 1)
                for d in detectar_baldosa(model, sub, sub_o, sub_m,
                                          par['umbral'], par['hue_agar'],
                                          par['sat_minima'], lim_area,
                                          en_borde):
                    brutas.append({'cy': d['cy'] + y0, 'cx': d['cx'] + x0,
                                   'area': d['area']})

        finales = fusionar(brutas)

        print(f'{ruta.stem:<12} {par["umbral"]:>7.2f} {usadas:>9} '
              f'{len(brutas):>7} {len(brutas)-len(finales):>7} '
              f'{len(finales):>6}')
        filas.append({'placa': ruta.stem, 'conteo_mosaico': len(finales),
                      'brutas': len(brutas),
                      'fusionadas': len(brutas) - len(finales),
                      'baldosas': usadas, 'umbral': round(par['umbral'], 3)})
        pd.DataFrame(filas).to_csv(salida / 'summary.csv', index=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        fig.suptitle(f'{ruta.name}   mosaico {args.rejilla}x{args.rejilla}, '
                     f'{len(finales)} colonias',
                     fontsize=13, fontweight='bold')
        axes[0].imshow(cv2.cvtColor(limpio, cv2.COLOR_BGR2RGB))
        for y0, y1 in filas_y:
            axes[0].axhline(y0, color='orange', lw=0.8, ls='--')
        for x0, x1 in filas_x:
            axes[0].axvline(x0, color='orange', lw=0.8, ls='--')
        axes[0].set_title('Placa limpia y cortes del mosaico', fontsize=11)
        axes[1].imshow(cv2.cvtColor(limpio, cv2.COLOR_BGR2RGB))
        for g in finales:
            axes[1].add_patch(plt.Circle(
                (g['cx'], g['cy']), np.sqrt(g['area'] / np.pi) * 1.3,
                fill=False, edgecolor='lime', linewidth=1.2))
        axes[1].set_title(f'Detección  n = {len(finales)}', fontsize=11)
        for a in axes:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(salida / f'{ruta.stem}_mosaico.png', dpi=110,
                    bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(filas)
    df.to_csv(salida / 'summary.csv', index=False)
    print('-' * len(cab))
    print(f'\nTotal: {df["conteo_mosaico"].sum()} colonias en {len(df)} placas')
    print(f'Rango: {df["conteo_mosaico"].min()} a {df["conteo_mosaico"].max()}')
    print(f'Detecciones fusionadas por solape: {df["fusionadas"].sum()}')
    print(f'\nResultados en {salida}/')


if __name__ == '__main__':
    main()
