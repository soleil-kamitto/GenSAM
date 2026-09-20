"""
Conteo por consenso entre varias configuraciones.

Motivo. Sin conteo manual disponible no hay forma de elegir un umbral mirando
el resultado, y elegirlo a ojo seria arbitrario. El consenso evita esa eleccion:
se corre el detector con varios umbrales y se conserva cada colonia que aparece
en al menos una fraccion de las corridas. Una deteccion estable entre
configuraciones es probablemente una colonia; una que solo aparece con el umbral
mas permisivo es probablemente ruido.

Es el mismo principio que la votacion entre modelos, aplicado aqui sobre un solo
modelo con distintas sensibilidades, y no requiere conocer la respuesta.

Limite del metodo, que conviene tener presente. El consenso premia lo estable,
y estable no es lo mismo que correcto. Un trazo de rotulador se detecta con los
cuatro umbrales por igual, de modo que reune el consenso completo y la votacion
lo confirma en lugar de descartarlo. Frente a un error sistematico la votacion
no solo no ayuda, sino que lo respalda. Por eso la rotulacion se elimina antes
de votar y no se deja en manos del filtro posterior.

Preprocesamiento. Se usa la densidad optica descrita en
scripts/preproceso_fisico.py, que estima la iluminacion con morfologia en lugar
de desenfoque, para no contaminarla con las propias colonias, y aplica
Beer-Lambert para que el valor resultante sea proporcional a la biomasa
independientemente de cuanta luz llegue a esa zona de la placa.

Uso:
    python scripts/contar_consenso.py images/mis_fotos_lote3
    python scripts/contar_consenso.py images/mis_fotos_lote3 --votos 0.5

Guarda:
    results/colonias/<carpeta>_consenso/summary.csv
    results/colonias/<carpeta>_consenso/<placa>_consenso.png
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
    detect_plate, crop_plate,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY, SUPPORTED,
)
from preproceso_fisico import densidad_optica
from autocalibrar import calibrar, es_tinta
from quitar_rotulacion import limpiar_rotulacion
from cellSAM import get_model, segment_cellular_image

UMBRALES = [0.30, 0.40, 0.50, 0.60]
FRACCION_VOTOS = 0.5     # una colonia debe aparecer en la mitad de las corridas
DIST_MISMA = 0.6         # fraccion del radio para considerar dos detecciones
                         # la misma colonia entre corridas


def detectar(model, entrada, original, mascara, umbral, hue_agar, sat_minima):
    """Detecciones de una corrida, como centroides con su area."""
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
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    return [{'cy': p.centroid[0], 'cx': p.centroid[1], 'area': p.area}
            for p in regionprops(seg)
            if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
            and p.solidity >= MIN_SOLIDITY
            and not es_tinta(hsv, seg, p.label, hue_agar, sat_minima)]


def votar(corridas, fraccion):
    """
    Agrupa las detecciones de todas las corridas y conserva las que reunen
    votos suficientes.

    Dos detecciones de corridas distintas se consideran la misma colonia si sus
    centroides estan mas cerca que DIST_MISMA veces la suma de sus radios
    equivalentes. Se procesa de mayor a menor area para que las colonias grandes
    definan el grupo.
    """
    minimo = max(1, int(round(len(corridas) * fraccion)))
    grupos = []
    for indice, detecciones in enumerate(corridas):
        for d in sorted(detecciones, key=lambda x: -x['area']):
            rd = np.sqrt(d['area'] / np.pi)
            encontrado = None
            for g in grupos:
                rg = np.sqrt(g['area'] / np.pi)
                if np.hypot(d['cx'] - g['cx'], d['cy'] - g['cy']) \
                        < DIST_MISMA * (rd + rg):
                    encontrado = g
                    break
            if encontrado is None:
                grupos.append({'cx': d['cx'], 'cy': d['cy'], 'area': d['area'],
                               'votos': {indice}})
            else:
                encontrado['votos'].add(indice)
    return [g for g in grupos if len(g['votos']) >= minimo], grupos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('imagenes', nargs='?', default='images/mis_fotos_lote3')
    ap.add_argument('--votos', type=float, default=FRACCION_VOTOS)
    args = ap.parse_args()

    origen = Path(args.imagenes)
    salida = Path('results/colonias') / f'{origen.name}_consenso'
    salida.mkdir(parents=True, exist_ok=True)

    rutas = sorted(p for p in origen.iterdir()
                   if p.suffix.lower() in SUPPORTED
                   and not p.name.startswith('ground'))
    print(f'Imagenes: {len(rutas)}')
    print(f'Umbrales: {UMBRALES}')
    print(f'Se conserva lo detectado en al menos el {args.votos*100:.0f}% '
          f'de las corridas\n')
    print('Cargando CellSAM...')
    model = get_model()
    print('Listo.\n')

    cab = f'{"placa":<12}' + ''.join(f'{u:>7.2f}' for u in UMBRALES) + \
          f'{"union":>8}{"consenso":>10}'
    print(cab)
    print('-' * len(cab))

    filas = []
    for ruta in rutas:
        img = cv2.imread(str(ruta))
        cx, cy, r = detect_plate(img)
        crop, mascara = crop_plate(img, cx, cy, r)
        par = calibrar(crop, mascara)

        # La rotulacion se quita antes de votar, no despues. Si no, cada trazo
        # se detecta con los cuatro umbrales por igual, de modo que reune el
        # consenso completo y el mecanismo de votacion lo confirma en lugar de
        # descartarlo: una deteccion estable no es lo mismo que una correcta.
        limpio, _, mascara = limpiar_rotulacion(crop, mascara, par['hue_agar'])
        entrada = densidad_optica(limpio, mascara)

        corridas = []
        for u in UMBRALES:
            corridas.append(detectar(model, entrada, limpio, mascara, u,
                                     par['hue_agar'], par['sat_minima']))

        consenso, todos = votar(corridas, args.votos)

        linea = f'{ruta.stem:<12}' + ''.join(f'{len(c):>7}' for c in corridas) \
                + f'{len(todos):>8}{len(consenso):>10}'
        print(linea)

        fila = {'placa': ruta.stem, 'union': len(todos),
                'consenso': len(consenso)}
        for u, c in zip(UMBRALES, corridas):
            fila[f'thr_{u:.2f}'] = len(c)
        filas.append(fila)
        pd.DataFrame(filas).to_csv(salida / 'summary.csv', index=False)

        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        fig.suptitle(f'{ruta.name}   consenso = {len(consenso)} colonias',
                     fontsize=13, fontweight='bold')
        axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Placa recortada', fontsize=11)
        axes[1].imshow(cv2.cvtColor(entrada, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Densidad óptica, entrada al modelo', fontsize=11)
        axes[2].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        for g in todos:
            seguro = len(g['votos']) >= max(1, int(round(len(UMBRALES)
                                                        * args.votos)))
            rad = np.sqrt(g['area'] / np.pi)
            axes[2].add_patch(plt.Circle(
                (g['cx'], g['cy']), rad * 1.3, fill=False,
                edgecolor='lime' if seguro else 'red',
                linewidth=1.4 if seguro else 0.9,
                linestyle='-' if seguro else ':'))
        axes[2].set_title(f'Verde: {len(consenso)} con consenso.  '
                          f'Rojo punteado: {len(todos)-len(consenso)} dudosas',
                          fontsize=11)
        for a in axes:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(salida / f'{ruta.stem}_consenso.png', dpi=120,
                    bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(filas)
    df.to_csv(salida / 'summary.csv', index=False)
    print('-' * len(cab))
    print(f'\nConsenso total: {df["consenso"].sum()} colonias en {len(df)} placas')
    print(f'Union total:    {df["union"].sum()}')
    print(f'\nResultados en {salida}/')


if __name__ == '__main__':
    main()
