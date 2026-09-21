"""
Mide cuantas colonias hay en la franja del borde que el recorte descarta.

Motivo. El conteo del tercer lote queda un 27 % por debajo del conteo manual
fisico, y el primer sospechoso es que el recorte al 92 % del radio deja sin
examinar el 15,4 % del area de la placa. Suponiendo que las colonias se
repartieran de forma uniforme, ahi caerian dos tercios de las que faltan.

Pero ese supuesto es dudoso, y la inspeccion visual de la franja desenrollada lo
desmiente en parte, porque esta dominada por la rotulacion y por el menisco de
la placa mas que por colonias. Suponer en lugar de medir es justo lo que este
trabajo critica, asi que se mide.

Metodo. Se ejecuta la deteccion sobre el disco completo y se clasifica cada
region segun su distancia al centro, en la zona que el recorte habitual examina
y en la franja que descarta. De las regiones de la franja se informa ademas su
tamano y su oscuridad relativa al agar, porque una colonia es oscura y compacta
mientras que una gota de condensacion es clara y esta pegada a la pared, de modo
que las dos cosas se pueden distinguir sin recurrir al conteo manual.

Lo que este experimento NO hace. No ajusta el radio de recorte para que el total
coincida con el conteo manual. Eso seria ajustar un parametro contra la
respuesta, que invalidaria la prueba a ciegas del lote. Aqui solo se cuenta que
hay en la franja, y la decision sobre el radio se toma despues con la curva de
recuperacion sobre las coordenadas anotadas.

Uso:
    python scripts/exp_franja_borde.py images/mis_fotos_lote3
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
    detect_plate, crop_plate, flat_field, SHRINK,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY, SUPPORTED,
)
from autocalibrar import calibrar, es_tinta
from quitar_rotulacion import limpiar_rotulacion
from cellSAM import get_model, segment_cellular_image

SALIDA = Path('results/colonias/experimentos/22_franja_borde')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('imagenes', nargs='?', default='images/mis_fotos_lote3')
    args = ap.parse_args()

    origen = Path(args.imagenes)
    SALIDA.mkdir(parents=True, exist_ok=True)
    rutas = sorted(p for p in origen.iterdir() if p.suffix.lower() in SUPPORTED)

    print(f'Recorte habitual: {SHRINK:.0%} del radio, '
          f'que descarta el {100 * (1 - SHRINK ** 2):.1f}% del area')
    print('Aqui se examina el disco completo y se reparte lo detectado.\n')
    print('Cargando CellSAM...')
    model = get_model()
    print('Listo.\n')

    cab = (f'{"placa":<10} {"dentro":>7} {"franja":>7} {"total":>6} '
           f'{"area franja":>12} {"oscuridad":>10}')
    print(cab)
    print('-' * len(cab))

    filas, detalle = [], []
    for ruta in rutas:
        img = cv2.imread(str(ruta))
        cx, cy, r = detect_plate(img)

        # disco completo, sin encoger
        crop, mascara = crop_plate(img, cx, cy, r, shrink=1.0)
        par = calibrar(crop, mascara)
        limpio, _, mascara = limpiar_rotulacion(crop, mascara, par['hue_agar'])
        entrada = flat_field(limpio, mascara) if par['aplicar_flat'] else limpio

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
        gris = cv2.cvtColor(limpio, cv2.COLOR_BGR2GRAY).astype(np.float32)
        v_agar = float(np.median(gris[mascara > 0]))

        ys, xs = np.nonzero(mascara > 0)
        ccx, ccy = float(xs.mean()), float(ys.mean())
        radio = float(np.sqrt((mascara > 0).sum() / np.pi))

        dentro = franja = 0
        areas_f, osc_f = [], []
        for p in regionprops(seg):
            if not (MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA):
                continue
            if p.solidity < MIN_SOLIDITY:
                continue
            if es_tinta(hsv, seg, p.label, par['hue_agar'], par['sat_minima']):
                continue
            d = np.hypot(p.centroid[1] - ccx, p.centroid[0] - ccy) / radio
            if d <= SHRINK:
                dentro += 1
            else:
                franja += 1
                areas_f.append(p.area)
                osc_f.append(float(np.median(gris[seg == p.label])) / v_agar)
                detalle.append({'placa': ruta.stem, 'radio_rel': round(d, 3),
                                'area': int(p.area),
                                'oscuridad': round(osc_f[-1], 3)})

        a_med = np.median(areas_f) if areas_f else float('nan')
        o_med = np.median(osc_f) if osc_f else float('nan')
        print(f'{ruta.stem:<10} {dentro:>7} {franja:>7} {dentro + franja:>6} '
              f'{a_med:>12.0f} {o_med:>10.2f}')
        filas.append({'placa': ruta.stem, 'dentro': dentro, 'en_franja': franja,
                      'total': dentro + franja,
                      'area_mediana_franja': None if np.isnan(a_med) else int(a_med),
                      'oscuridad_mediana_franja': None if np.isnan(o_med)
                      else round(o_med, 3)})

        # se guarda despues de cada placa y no al final. Una corrida de estas
        # dura decenas de minutos, y guardar solo al terminar significa perderlo
        # todo si el proceso se corta, cosa que ocurrio.
        pd.DataFrame(filas).to_csv(SALIDA / 'franja_por_placa.csv', index=False)
        pd.DataFrame(detalle).to_csv(SALIDA / 'franja_detalle.csv', index=False)

    df = pd.DataFrame(filas)
    df.to_csv(SALIDA / 'franja_por_placa.csv', index=False)
    pd.DataFrame(detalle).to_csv(SALIDA / 'franja_detalle.csv', index=False)

    print('-' * len(cab))
    print(f'{"TOTAL":<10} {df.dentro.sum():>7} {df.en_franja.sum():>7} '
          f'{df.total.sum():>6}')
    print()
    print(f'El recorte habitual descarta {df.en_franja.sum()} detecciones en '
          f'{len(df)} placas,')
    print(f'es decir {df.en_franja.sum() / len(df):.1f} por placa.')
    print()
    print('Para interpretarlo: una colonia es oscura respecto al agar, con')
    print('oscuridad claramente por debajo de 1, mientras que una gota de')
    print('condensacion es clara o apenas mas oscura que el fondo.')
    print(f'\nGuardado en {SALIDA}/')


if __name__ == '__main__':
    main()
