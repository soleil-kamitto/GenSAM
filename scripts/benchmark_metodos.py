"""
Banco de pruebas: compara metodos de conteo de colonias sobre las 15 placas
propias con conteo manual de referencia.

Motivacion: hasta ahora el proyecto solo ha explorado CellSAM con
preprocesamiento clasico, y los experimentos de umbral y mosaico mostraron que
esa via tiene un techo. Antes de seguir afinandola conviene medir que dan otras
familias de metodos sobre las mismas imagenes.

Los metodos clasicos corren en segundos por placa, frente a los minutos que
tarda CellSAM, asi que se pueden comparar todos de una pasada.

Metodos incluidos
  cellsam      resultados ya calculados, se leen del CSV
  watershed    umbral + transformada de distancia + watershed. Ataca el fallo
               conocido de colonias fusionadas en placas densas
  blob_log     deteccion de blobs por Laplaciano de gaussianas
  blob_doh     deteccion de blobs por determinante del hessiano
  hough        circulos de Hough, aprovechando que las colonias son redondas

Uso:
    python scripts/benchmark_metodos.py

Guarda:
    results/colonias/experimentos/12_benchmark/resultados.csv
    results/colonias/experimentos/12_benchmark/<placa>_metodos.png
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
from scipy import ndimage as ndi
from skimage.feature import blob_log, blob_doh, peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, label

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from contar_mis_fotos import detect_plate, crop_plate, flat_field

IMGS_DIR = Path('images/mis_fotos')
GT_CSV = Path('results/ground_truth_mis_fotos/ground_truth.csv')
CELLSAM_CSV = Path('results/colonias/mis_fotos/summary.csv')
OUT_DIR = Path('results/colonias/experimentos/12_benchmark')

# rango de tamano de colonia, en pixeles sobre el recorte de 1200 px.
# Sale de las areas medidas en las detecciones validadas (439 a ~8000 px2),
# que corresponden a radios de entre 12 y 50 px aproximadamente.
RADIO_MIN, RADIO_MAX = 8, 50
AREA_MIN, AREA_MAX = 300, 50000
TNTC = 250


def preparar(ruta):
    """Recorte de la placa y version corregida, comunes a todos los metodos."""
    img = cv2.imread(str(ruta))
    cx, cy, r = detect_plate(img)
    crop, plate_mask = crop_plate(img, cx, cy, r)
    flat = flat_field(crop, plate_mask)
    gris = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    return crop, plate_mask, gris


def binarizar(gris, plate_mask):
    """Colonias oscuras sobre agar claro, por Otsu dentro de la placa."""
    dentro = plate_mask > 0
    valores = gris[dentro]
    umbral, _ = cv2.threshold(valores, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaria = ((gris < umbral) & dentro).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
    return binaria


def metodo_watershed(crop, plate_mask, gris):
    binaria = binarizar(gris, plate_mask)
    dist = ndi.distance_transform_edt(binaria > 0)
    # los picos de la transformada de distancia son los centros de las colonias,
    # y la separacion minima evita partir una colonia en varias
    coords = peak_local_max(dist, min_distance=RADIO_MIN,
                            labels=binaria > 0, exclude_border=False)
    marcadores = np.zeros(dist.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, 1):
        marcadores[y, x] = i
    etiquetas = watershed(-dist, marcadores, mask=binaria > 0)
    centros = [(p.centroid[1], p.centroid[0])
               for p in regionprops(etiquetas)
               if AREA_MIN <= p.area <= AREA_MAX]
    return centros


def metodo_blob_log(crop, plate_mask, gris):
    inv = (255 - gris).astype(np.float32) / 255.0
    inv[plate_mask == 0] = 0
    blobs = blob_log(inv, min_sigma=RADIO_MIN / np.sqrt(2),
                     max_sigma=RADIO_MAX / np.sqrt(2), num_sigma=8,
                     threshold=0.10, overlap=0.5)
    return [(b[1], b[0]) for b in blobs]


def metodo_blob_doh(crop, plate_mask, gris):
    inv = (255 - gris).astype(np.float32) / 255.0
    inv[plate_mask == 0] = 0
    blobs = blob_doh(inv, min_sigma=RADIO_MIN, max_sigma=RADIO_MAX,
                     num_sigma=8, threshold=0.003, overlap=0.5)
    return [(b[1], b[0]) for b in blobs]


def metodo_hough(crop, plate_mask, gris):
    suave = cv2.medianBlur(gris, 5)
    circulos = cv2.HoughCircles(
        suave, cv2.HOUGH_GRADIENT, dp=1.0, minDist=RADIO_MIN * 2,
        param1=80, param2=18, minRadius=RADIO_MIN, maxRadius=RADIO_MAX)
    if circulos is None:
        return []
    salida = []
    for x, y, r in np.round(circulos[0]).astype(int):
        if 0 <= y < plate_mask.shape[0] and 0 <= x < plate_mask.shape[1] \
                and plate_mask[y, x] > 0:
            salida.append((float(x), float(y)))
    return salida


METODOS = {
    'watershed': metodo_watershed,
    'blob_log': metodo_blob_log,
    'blob_doh': metodo_blob_doh,
    'hough': metodo_hough,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(GT_CSV).set_index('image')['count'].to_dict()
    cellsam = pd.read_csv(CELLSAM_CSV).set_index('image')['count'].to_dict()

    imagenes = sorted(p for p in IMGS_DIR.glob('*.jpg'))
    filas = []

    cab = f'{"placa":<14} {"manual":>7} {"cellsam":>8}'
    for m in METODOS:
        cab += f' {m:>10}'
    print(cab)
    print('-' * len(cab))

    for ruta in imagenes:
        stem = ruta.stem
        crop, plate_mask, gris = preparar(ruta)

        resultados = {}
        centros_por_metodo = {}
        for nombre, fn in METODOS.items():
            try:
                centros = fn(crop, plate_mask, gris)
            except Exception as e:
                print(f'  {stem} {nombre}: fallo ({type(e).__name__})')
                centros = []
            resultados[nombre] = len(centros)
            centros_por_metodo[nombre] = centros

        linea = (f'{stem:<14} {gt.get(stem, 0):>7} '
                 f'{cellsam.get(stem, 0):>8}')
        for m in METODOS:
            linea += f' {resultados[m]:>10}'
        print(linea)

        fila = {'placa': stem, 'manual': gt.get(stem, 0),
                'cellsam': cellsam.get(stem, 0)}
        fila.update(resultados)
        filas.append(fila)

        # figura comparativa por placa
        n = len(METODOS) + 1
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
        fig.suptitle(f'{stem}   manual = {gt.get(stem, 0)}',
                     fontsize=12, fontweight='bold')
        axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Placa', fontsize=10)
        axes[0].axis('off')
        for k, (nombre, centros) in enumerate(centros_por_metodo.items(), 1):
            axes[k].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if centros:
                xs, ys = zip(*centros)
                axes[k].scatter(xs, ys, s=26, facecolors='none',
                                edgecolors='red', linewidths=1.1)
            axes[k].set_title(f'{nombre}  n={len(centros)}', fontsize=10)
            axes[k].axis('off')
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{stem}_metodos.png', dpi=110,
                    bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / 'resultados.csv', index=False)

    # ── resumen de error ─────────────────────────────────────────────────────
    contables = df[df['manual'] <= TNTC]
    print('\n' + '=' * 58)
    print(f'{"metodo":<12} {"MAE contable":>13} {"MAE global":>11} {"acierto":>9}')
    print('-' * 58)
    for m in ['cellsam'] + list(METODOS):
        err_c = (contables[m] - contables['manual']).abs()
        err_g = (df[m] - df['manual']).abs()
        acierto = 1 - err_g.sum() / df['manual'].sum()
        print(f'{m:<12} {err_c.mean():>13.2f} {err_g.mean():>11.2f} '
              f'{acierto*100:>8.1f}%')
    print('=' * 58)
    print(f'\nResultados en {OUT_DIR}/')


if __name__ == '__main__':
    main()
