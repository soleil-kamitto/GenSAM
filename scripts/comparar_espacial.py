"""
Compara las detecciones del sistema con el conteo manual, colonia por colonia.

Por que no basta comparar totales. Un total que coincide puede estar compuesto
de una colonia perdida y un falso positivo que se cancelan, de modo que el
acierto agregado esconde dos errores en lugar de indicar que no los hay. Al
cruzar las posiciones se distingue lo que el sistema encontro, lo que se le paso
y lo que se invento, y esos tres numeros dicen que hay que arreglar mientras que
el total no dice nada.

Como se emparejan. Cada marca manual se asigna a lo sumo a una deteccion y cada
deteccion a lo sumo a una marca, resolviendo el reparto completo de forma optima
en lugar de ir tomando el vecino mas cercano. La diferencia importa cuando hay
colonias juntas, porque un emparejamiento avaro puede consumir la deteccion que
le correspondia a la vecina y encadenar errores que no existen.

Una marca y una deteccion se consideran la misma colonia si distan menos que el
radio de la deteccion multiplicado por un margen, con un minimo absoluto para
que las colonias diminutas no queden imposibles de emparejar. El criterio es
generoso a proposito, porque el clic manual se coloca a ojo en el centro y no
tiene por que caer en el centroide exacto de la region segmentada.

Requisitos. El conteo manual sobre las fotografias se genera con
scripts/conteo_manual.py, y las coordenadas de las detecciones las escribe
scripts/contar_auto.py junto al resto de los resultados.

Uso:
    python scripts/comparar_espacial.py images/mis_fotos_lote3
    python scripts/comparar_espacial.py images/mis_fotos --variante _auto
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

MARGEN_RADIO = 1.8      # cuantos radios de la deteccion se admiten de distancia
DISTANCIA_MINIMA = 25   # pixeles de la imagen original, para colonias diminutas
SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def emparejar(manual, detectado, radios):
    """
    Reparte marcas manuales y detecciones de forma optima.

    Devuelve los indices emparejados, los de las marcas sin detectar y los de
    las detecciones sin marca. Se usa el algoritmo hungaro sobre la matriz de
    distancias, con las parejas demasiado lejanas anuladas despues, porque el
    reparto optimo global evita los encadenamientos que produce ir tomando el
    vecino mas cercano cuando hay colonias juntas.
    """
    if len(manual) == 0 or len(detectado) == 0:
        return [], list(range(len(manual))), list(range(len(detectado)))

    d = np.hypot(manual[:, 0][:, None] - detectado[:, 0][None, :],
                 manual[:, 1][:, None] - detectado[:, 1][None, :])
    tolerancia = np.maximum(radios * MARGEN_RADIO, DISTANCIA_MINIMA)
    admisible = d <= tolerancia[None, :]

    # las parejas inadmisibles reciben un costo enorme para que el reparto no
    # las elija salvo que no quede alternativa, y luego se descartan
    costo = np.where(admisible, d, 1e6)
    fila, col = linear_sum_assignment(costo)

    parejas = [(i, j) for i, j in zip(fila, col) if admisible[i, j]]
    usadas_m = {i for i, _ in parejas}
    usadas_d = {j for _, j in parejas}
    perdidas = [i for i in range(len(manual)) if i not in usadas_m]
    sobrantes = [j for j in range(len(detectado)) if j not in usadas_d]
    return parejas, perdidas, sobrantes


def dibujar(ruta_img, manual, detectado, radios, parejas, perdidas, sobrantes,
            destino, titulo):
    """Superpone las tres categorias sobre la fotografia original."""
    img = cv2.imread(str(ruta_img))
    if img is None:
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for i, j in parejas:
        ax.add_patch(plt.Circle((detectado[j, 0], detectado[j, 1]),
                                max(radios[j], 12), fill=False,
                                edgecolor='#27ae60', linewidth=1.8))
    for i in perdidas:
        ax.plot(manual[i, 0], manual[i, 1], 'x', color='#e74c3c',
                markersize=11, markeredgewidth=2.4)
    for j in sobrantes:
        ax.add_patch(plt.Circle((detectado[j, 0], detectado[j, 1]),
                                max(radios[j], 12), fill=False,
                                edgecolor='#f39c12', linewidth=1.8,
                                linestyle='--'))

    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.axis('off')
    leyenda = [
        plt.Line2D([], [], color='#27ae60', marker='o', linestyle='',
                   markerfacecolor='none', markeredgewidth=2,
                   label=f'Acertadas: {len(parejas)}'),
        plt.Line2D([], [], color='#e74c3c', marker='x', linestyle='',
                   markeredgewidth=2, label=f'Perdidas: {len(perdidas)}'),
        plt.Line2D([], [], color='#f39c12', marker='o', linestyle='',
                   markerfacecolor='none', markeredgewidth=2,
                   label=f'Falsos positivos: {len(sobrantes)}'),
    ]
    ax.legend(handles=leyenda, loc='upper right', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(destino, dpi=110, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('imagenes', nargs='?', default='images/mis_fotos_lote3')
    ap.add_argument('--variante', default='_auto',
                    help='sufijo de la carpeta de resultados a comparar')
    args = ap.parse_args()

    origen = Path(args.imagenes)
    gt_dir = Path('results') / f'ground_truth_{origen.name}'
    det_dir = Path('results/colonias') / f'{origen.name}{args.variante}'
    salida = det_dir / 'comparacion_espacial'

    if not gt_dir.exists():
        print(f'No hay conteo manual en {gt_dir}/')
        print(f'Se genera con: python scripts/conteo_manual.py {origen}')
        return
    if not det_dir.exists():
        print(f'No hay resultados en {det_dir}/')
        return
    salida.mkdir(parents=True, exist_ok=True)

    rutas = sorted(p for p in origen.iterdir() if p.suffix.lower() in SUPPORTED)
    cab = (f'{"placa":<12} {"manual":>7} {"sistema":>8} {"acierta":>8} '
           f'{"pierde":>7} {"inventa":>8} {"sensib":>7} {"precis":>7}')
    print(cab)
    print('-' * len(cab))

    filas = []
    for ruta in rutas:
        f_gt = gt_dir / f'{ruta.stem}_puntos.csv'
        f_det = det_dir / f'{ruta.stem}_detecciones.csv'
        if not f_gt.exists():
            print(f'{ruta.stem:<12} sin conteo manual')
            continue
        if not f_det.exists():
            print(f'{ruta.stem:<12} sin coordenadas del sistema, '
                  f'volver a ejecutar contar_auto.py')
            continue

        gt = pd.read_csv(f_gt)
        det = pd.read_csv(f_det)
        manual = gt[['x', 'y']].to_numpy(dtype=float)
        detectado = det[['x', 'y']].to_numpy(dtype=float) if len(det) \
            else np.empty((0, 2))
        radios = det['radio_original'].to_numpy(dtype=float) if len(det) \
            else np.empty(0)

        parejas, perdidas, sobrantes = emparejar(manual, detectado, radios)
        acierta, pierde, inventa = len(parejas), len(perdidas), len(sobrantes)
        sensib = acierta / max(len(manual), 1)
        precis = acierta / max(len(detectado), 1)

        print(f'{ruta.stem:<12} {len(manual):>7} {len(detectado):>8} '
              f'{acierta:>8} {pierde:>7} {inventa:>8} '
              f'{100 * sensib:>6.0f}% {100 * precis:>6.0f}%')
        filas.append({'placa': ruta.stem, 'manual': len(manual),
                      'sistema': len(detectado), 'acertadas': acierta,
                      'perdidas': pierde, 'falsos_positivos': inventa,
                      'sensibilidad': round(sensib, 3),
                      'precision': round(precis, 3)})

        dibujar(ruta, manual, detectado, radios, parejas, perdidas, sobrantes,
                salida / f'{ruta.stem}_espacial.png',
                f'{ruta.name}   manual {len(manual)}, sistema {len(detectado)}')

    if not filas:
        print('\nNo hubo ninguna placa comparable.')
        return

    df = pd.DataFrame(filas)
    df.to_csv(salida / 'comparacion_espacial.csv', index=False)
    print('-' * len(cab))
    tot_m, tot_s = df.manual.sum(), df.sistema.sum()
    tot_a, tot_p, tot_f = (df.acertadas.sum(), df.perdidas.sum(),
                           df.falsos_positivos.sum())
    print(f'{"TOTAL":<12} {tot_m:>7} {tot_s:>8} {tot_a:>8} {tot_p:>7} '
          f'{tot_f:>8} {100 * tot_a / max(tot_m, 1):>6.0f}% '
          f'{100 * tot_a / max(tot_s, 1):>6.0f}%')

    print()
    print('Sensibilidad: de las colonias que marcaste, cuantas encontro.')
    print('Precision: de lo que propuso, cuanto era una colonia de verdad.')
    print()
    if tot_p > tot_f * 1.5:
        print('El fallo dominante es perder colonias, de modo que conviene')
        print('actuar sobre el detector o sobre la imagen de entrada.')
    elif tot_f > tot_p * 1.5:
        print('El fallo dominante son los falsos positivos, de modo que')
        print('conviene actuar sobre los filtros posteriores.')
    else:
        print('Los dos fallos son de magnitud parecida, y en ese caso el total')
        print('los compensa en parte y parece mejor de lo que es.')
    print(f'\nResultados en {salida}/')


if __name__ == '__main__':
    main()
