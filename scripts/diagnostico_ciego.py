"""
Diagnosticos de calidad que no necesitan conteo manual.

Motivo. La investigadora conserva el conteo de referencia sin comunicarlo, de
modo que no se puede validar mirando el error. Hace falta entonces juzgar una
corrida por propiedades que deberian cumplirse si las detecciones son colonias,
y que se rompen de forma caracteristica cuando son artefactos.

Se miden dos cosas.

1. Distribucion radial. Las colonias crecen donde cayo el inoculo, y ese reparto
   no tiene motivo para preferir el perimetro de la placa. Si la densidad de
   detecciones por unidad de area se dispara en el anillo exterior, lo mas
   probable es que se esten contando artefactos del recorte o de la
   reconstruccion de la rotulacion, que es justo el fallo que se observo en
   RC73-8. Se compara la densidad del anillo exterior con la del resto.

2. Distribucion de tamanos. Una poblacion de colonias del mismo cultivo crece de
   forma parecida, asi que sus areas se agrupan. Los artefactos no siguen esa
   ley y aparecen como una cola de objetos muy pequenos. Se informa la mediana,
   la dispersion relativa y la fraccion de detecciones con area menor que la
   mitad de la mediana.

Ninguna de las dos da una cifra de acierto. Sirven para comparar dos variantes
del metodo entre si y para senalar placas sospechosas que conviene mirar.

Uso:
    python scripts/diagnostico_ciego.py images/mis_fotos_lote3
    python scripts/diagnostico_ciego.py images/mis_fotos_lote3 --sin-limpieza
"""
import argparse
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
    detect_plate, crop_plate, flat_field,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY, SUPPORTED,
)
from autocalibrar import calibrar, es_tinta
from quitar_rotulacion import limpiar_rotulacion
from preproceso_fisico import densidad_optica
from cellSAM import get_model, segment_cellular_image

FRACCION_ANILLO = 0.85   # lo que queda fuera de este radio es el anillo exterior


def radiales(centroides, cx, cy, radio):
    """
    Densidad de detecciones por unidad de area dentro y fuera del anillo.

    Se normaliza por area porque el anillo exterior, aun siendo estrecho,
    abarca una fraccion apreciable del disco, y compararlo por conteo bruto
    seria enganoso.
    """
    if not centroides:
        return float('nan'), 0, 0
    r_corte = radio * FRACCION_ANILLO
    d = np.array([np.hypot(x - cx, y - cy) for y, x in centroides])
    fuera = int((d > r_corte).sum())
    dentro = len(d) - fuera

    area_dentro = np.pi * r_corte ** 2
    area_anillo = np.pi * (radio ** 2 - r_corte ** 2)
    dens_dentro = dentro / max(area_dentro, 1.0)
    dens_fuera = fuera / max(area_anillo, 1.0)
    if dens_dentro <= 0:
        return (float('inf') if fuera else float('nan')), dentro, fuera
    return dens_fuera / dens_dentro, dentro, fuera


def tamanos(areas):
    """Mediana, dispersion relativa y cola de objetos anormalmente pequenos."""
    if len(areas) == 0:
        return float('nan'), float('nan'), float('nan')
    a = np.asarray(areas, dtype=float)
    med = float(np.median(a))
    # se usa la desviacion absoluta mediana, robusta frente a los propios
    # artefactos que se quiere detectar
    mad = float(np.median(np.abs(a - med)))
    return med, mad / max(med, 1e-6), float((a < med * 0.5).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('imagenes', nargs='?', default='images/mis_fotos_lote3')
    ap.add_argument('--densidad-optica', action='store_true')
    ap.add_argument('--sin-limpieza', action='store_true',
                    help='no eliminar la rotulacion, para comparar')
    args = ap.parse_args()

    origen = Path(args.imagenes)
    salida = Path('results/colonias/experimentos/20_diagnostico_ciego')
    salida.mkdir(parents=True, exist_ok=True)

    rutas = sorted(p for p in origen.iterdir()
                   if p.suffix.lower() in SUPPORTED
                   and not p.name.startswith('ground'))
    print(f'Imagenes: {len(rutas)}')
    print(f'Anillo exterior: fuera del {FRACCION_ANILLO:.0%} del radio\n')
    print('Cargando CellSAM...')
    model = get_model()
    print('Listo.\n')

    cab = (f'{"placa":<12} {"conteo":>7} {"borde":>6} {"centro":>7} '
           f'{"razon":>7} {"area med":>9} {"disp":>6} {"peq":>6}')
    print(cab)
    print('-' * len(cab))

    filas = []
    for ruta in rutas:
        img = cv2.imread(str(ruta))
        px, py, r = detect_plate(img)
        crop, mascara = crop_plate(img, px, py, r)
        par = calibrar(crop, mascara)

        if args.sin_limpieza:
            limpio = crop
        else:
            limpio, _, mascara = limpiar_rotulacion(crop, mascara,
                                                    par['hue_agar'])

        entrada = densidad_optica(limpio, mascara) if args.densidad_optica \
            else (flat_field(limpio, mascara) if par['aplicar_flat'] else limpio)

        try:
            seg, _, _ = segment_cellular_image(
                cv2.cvtColor(entrada, cv2.COLOR_BGR2RGB), model=model,
                normalize=True, postprocess=True,
                bbox_threshold=par['umbral'], device='cpu')
            if seg is None:
                seg = np.zeros(crop.shape[:2], dtype=np.int32)
        except (AttributeError, TypeError, ValueError):
            seg = np.zeros(crop.shape[:2], dtype=np.int32)
        seg = seg.copy()
        seg[mascara == 0] = 0

        hsv = cv2.cvtColor(limpio, cv2.COLOR_BGR2HSV)
        validas = [p for p in regionprops(seg)
                   if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                   and p.solidity >= MIN_SOLIDITY
                   and not es_tinta(hsv, seg, p.label, par['hue_agar'],
                                    par['sat_minima'])]

        # centro y radio del disco dentro del recorte
        ys, xs = np.nonzero(mascara > 0)
        ccx, ccy = float(xs.mean()), float(ys.mean())
        radio = float(np.sqrt((mascara > 0).sum() / np.pi))

        razon, dentro, fuera = radiales(
            [p.centroid for p in validas], ccx, ccy, radio)
        med, disp, peq = tamanos([p.area for p in validas])

        print(f'{ruta.stem:<12} {len(validas):>7} {fuera:>6} {dentro:>7} '
              f'{razon:>7.2f} {med:>9.0f} {disp:>6.2f} {peq:>6.2f}')
        filas.append({'placa': ruta.stem, 'conteo': len(validas),
                      'en_borde': fuera, 'en_centro': dentro,
                      'razon_borde_centro': round(razon, 3),
                      'area_mediana': round(med, 1),
                      'dispersion': round(disp, 3),
                      'fraccion_pequenas': round(peq, 3)})

    df = pd.DataFrame(filas)
    sufijo = ('od' if args.densidad_optica else 'cruda') + \
             ('_sin_limpieza' if args.sin_limpieza else '')
    df.to_csv(salida / f'diagnostico_{sufijo}.csv', index=False)

    print('-' * len(cab))
    print(f'\nRazon borde/centro, mediana: '
          f'{df["razon_borde_centro"].median():.2f}')
    print('  Un valor cercano a 1 indica reparto uniforme, que es lo esperado.')
    print('  Muy por encima de 1 indica acumulacion en el borde, tipica de')
    print('  artefactos del recorte o de la reconstruccion de la rotulacion.')
    print(f'\nDispersion relativa de area, mediana: '
          f'{df["dispersion"].median():.2f}')
    print(f'Fraccion de detecciones anormalmente pequenas: '
          f'{df["fraccion_pequenas"].median():.2f}')
    print(f'\nGuardado en {salida}/diagnostico_{sufijo}.csv')


if __name__ == '__main__':
    main()
