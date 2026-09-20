"""
Conteo de colonias con parametros derivados de cada fotografia.

A diferencia de scripts/contar_mis_fotos.py, que usa constantes fijadas para un
montaje concreto, aqui el umbral de deteccion, la conveniencia de corregir la
iluminacion y el filtro de color se calculan a partir de propiedades medibles de
la propia imagen (ver scripts/autocalibrar.py).

El motivo es que el sistema esta pensado para laboratorios sin presupuesto, que
fotografiaran sus placas con el telefono que tengan. Se comprobo que tres
parametros calibrados en un montaje fallan en otro, de modo que un pipeline con
constantes no sirve fuera del laboratorio donde se ajusto.

Uso:
    python scripts/contar_auto.py images/mis_fotos_lote3
    python scripts/contar_auto.py images/mis_fotos --comparar

Con --comparar se ejecuta ademas el pipeline de parametros fijos, para ver que
diferencia introduce la auto-calibracion.

Guarda:
    results/colonias/<carpeta>_auto/summary.csv
    results/colonias/<carpeta>_auto/<placa>_count.png
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
    detect_plate, crop_plate, flat_field, draw_overlay, scout, is_ink,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY, SUPPORTED,
)
from autocalibrar import calibrar, es_tinta
from quitar_rotulacion import limpiar_rotulacion
from preproceso_fisico import densidad_optica
from cellSAM import get_model, segment_cellular_image


def segmentar(model, entrada, original, mascara, umbral, hue_agar, sat_minima):
    """Segmenta y filtra, usando el tono del agar medido en esta imagen."""
    try:
        seg, _, _ = segment_cellular_image(
            cv2.cvtColor(entrada, cv2.COLOR_BGR2RGB), model=model,
            normalize=True, postprocess=True,
            bbox_threshold=umbral, device='cpu')
        if seg is None:
            seg = np.zeros(entrada.shape[:2], dtype=np.int32)
    except (AttributeError, TypeError, ValueError):
        seg = np.zeros(entrada.shape[:2], dtype=np.int32)
    seg = seg.copy()
    seg[mascara == 0] = 0

    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    base = [p for p in regionprops(seg)
            if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
            and p.solidity >= MIN_SOLIDITY]
    validas = [p for p in base
               if not es_tinta(hsv, seg, p.label, hue_agar, sat_minima)]
    return seg, validas, len(base) - len(validas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('imagenes', nargs='?', default='images/mis_fotos_lote3')
    ap.add_argument('--comparar', action='store_true',
                    help='ejecutar tambien el pipeline de parametros fijos')
    ap.add_argument('--densidad-optica', action='store_true',
                    help='usar densidad optica como entrada al modelo')
    args = ap.parse_args()

    origen = Path(args.imagenes)
    salida = Path('results/colonias') / f'{origen.name}_auto'
    salida.mkdir(parents=True, exist_ok=True)

    rutas = sorted(p for p in origen.iterdir()
                   if p.suffix.lower() in SUPPORTED
                   and not p.name.startswith('ground'))
    print(f'Imagenes: {len(rutas)}   salida: {salida}\n')
    print('Cargando CellSAM...')
    model = get_model()
    print('Listo.\n')

    cab = (f'{"placa":<14} {"umbral":>7} {"flat":>5} {"hue":>5} {"conteo":>7} '
           f'{"tinta":>6} {"%rotul":>7}')
    if args.comparar:
        cab += f' {"fijo":>6}'
    print(cab)
    print('-' * len(cab))

    filas = []
    for ruta in rutas:
        img = cv2.imread(str(ruta))
        cx, cy, r = detect_plate(img)
        crop, mascara = crop_plate(img, cx, cy, r)

        par = calibrar(crop, mascara)

        # La rotulacion se elimina antes de segmentar, no despues. Filtrarla a
        # posteriori obliga a decidir sobre regiones que montan a medias sobre
        # el trazo, y descarta enteras las colonias que lo tocan. Quitandola
        # primero, el detector no llega a proponer nada sobre ella.
        limpio, mascara_tinta, mascara = limpiar_rotulacion(
            crop, mascara, par['hue_agar'])
        pct_tinta = 100.0 * mascara_tinta.sum() / max(1, (mascara > 0).sum())

        # Entrada al modelo en densidad optica. Por Beer-Lambert la luz que
        # atraviesa la colonia cae de forma exponencial con su biomasa, asi que
        # el logaritmo de la razon entre imagen y fondo es proporcional a esa
        # biomasa. Eso corrige la iluminacion y linealiza en un solo paso, y da
        # mas contraste a las colonias tenues que la imagen cruda.
        #
        # El fondo se estima por morfologia y no por desenfoque, para que no se
        # contamine con las propias colonias.
        entrada = densidad_optica(limpio, mascara) if args.densidad_optica             else (flat_field(limpio, mascara) if par['aplicar_flat'] else limpio)
        # el filtro por region se conserva como red de seguridad, sobre la
        # imagen ya limpia deberia descartar muy poco
        seg, validas, n_tinta = segmentar(
            model, entrada, limpio, mascara,
            par['umbral'], par['hue_agar'], par['sat_minima'])
        n_auto = len(validas)

        fila = {'placa': ruta.stem, 'conteo_auto': n_auto,
                'umbral': round(par['umbral'], 3),
                'flat_field': par['aplicar_flat'],
                'hue_agar': par['hue_agar'],
                'pct_rotulacion': round(pct_tinta, 2),
                'descartadas_tinta': n_tinta}

        linea = (f'{ruta.stem:<14} {par["umbral"]:>7.2f} '
                 f'{"si" if par["aplicar_flat"] else "no":>5} '
                 f'{par["hue_agar"]:>5.0f} {n_auto:>7} {n_tinta:>6} '
                 f'{pct_tinta:>7.1f}')

        if args.comparar:
            plano = flat_field(crop, mascara)
            _, umbral_fijo, _ = scout(plano, mascara)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            seg_f, _, _ = segment_cellular_image(
                cv2.cvtColor(plano, cv2.COLOR_BGR2RGB), model=model,
                normalize=True, postprocess=True,
                bbox_threshold=umbral_fijo, device='cpu')
            if seg_f is None:
                seg_f = np.zeros(crop.shape[:2], dtype=np.int32)
            seg_f = seg_f.copy()
            seg_f[mascara == 0] = 0
            n_fijo = sum(1 for p in regionprops(seg_f)
                         if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                         and p.solidity >= MIN_SOLIDITY
                         and not is_ink(hsv, seg_f, p.label))
            fila['conteo_fijo'] = n_fijo
            linea += f' {n_fijo:>6}'

        print(linea)
        filas.append(fila)
        pd.DataFrame(filas).to_csv(salida / 'summary.csv', index=False)

        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        fig.suptitle(f'{ruta.name}   umbral {par["umbral"]:.2f}, '
                     f'flat-field {"si" if par["aplicar_flat"] else "no"}',
                     fontsize=13, fontweight='bold')
        axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Placa recortada', fontsize=11)
        axes[1].imshow(cv2.cvtColor(entrada, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Entrada al modelo', fontsize=11)
        axes[2].imshow(draw_overlay(crop, seg, validas))
        axes[2].set_title(f'Detección  n = {n_auto}', fontsize=11)
        for a in axes:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(salida / f'{ruta.stem}_count.png', dpi=120,
                    bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(filas)
    df.to_csv(salida / 'summary.csv', index=False)
    print('-' * len(cab))
    print(f'\nTotal detectado: {df["conteo_auto"].sum()} colonias en '
          f'{len(df)} placas')
    print(f'Rango: {df["conteo_auto"].min()} a {df["conteo_auto"].max()}')
    if args.comparar:
        print(f'Con parametros fijos: {df["conteo_fijo"].sum()} colonias')
    print(f'\nResultados en {salida}/')


if __name__ == '__main__':
    main()
