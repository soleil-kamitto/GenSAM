"""
Compara la morfologia de las colonias de actinomicetos con las de AGAR.

Esta medicion sostiene la premisa central del trabajo. Se afirma que los
datasets publicos de colonias contienen formas compactas de borde neto, mientras
que los actinomicetos son filamentosos y de borde difuso, y que por eso los
detectores entrenados con los primeros no sirven para los segundos. Hasta ahora
esa afirmacion era cualitativa; aqui se cuantifica.

Se miden cuatro descriptores por colonia:

  solidez        area dividida por el area de su envolvente convexa. Una forma
                 compacta se acerca a 1, una irregular o con proyecciones baja
  circularidad   4*pi*area / perimetro^2. Un circulo perfecto vale 1, y baja
                 cuanto mas recortado sea el contorno
  nitidez        magnitud media del gradiente en el borde de la colonia. Un
                 borde neto da valores altos, uno difuso valores bajos
  contraste      diferencia de intensidad entre la colonia y el agar de
                 alrededor, normalizada

AGAR aporta cajas, asi que la colonia se segmenta dentro de cada caja. Las fotos
propias aportan el centro de cada colonia, del conteo manual, asi que se
segmenta dentro de una ventana centrada en ese punto. En ambos casos se
segmenta igual, por Otsu local, para que la comparacion sea justa.

Uso:
    python scripts/exp_morfologia_comparada.py

Guarda:
    results/colonias/experimentos/16_morfologia/descriptores.csv
    results/colonias/experimentos/16_morfologia/comparacion.png
"""
import csv
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from skimage.measure import regionprops, label

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from contar_mis_fotos import detect_plate, crop_plate, SHRINK

AGAR_DIR = Path('images/agar_muestra')
FOTOS = Path('images/mis_fotos')
PUNTOS = Path('results/ground_truth_mis_fotos')
OUT_DIR = Path('results/colonias/experimentos/16_morfologia')

VENTANA = 2.2      # la ventana de analisis, en multiplos del radio de la colonia
RADIO_PROPIO = 26  # radio tipico en el recorte de 1200 px
MAX_POR_PLACA = 60 # se limita para que ninguna placa domine la muestra


def medir(parche):
    """
    Segmenta la colonia central del parche y devuelve sus descriptores.

    Devuelve None si no se encuentra una region central razonable, lo que ocurre
    cuando la colonia es demasiado tenue o la ventana cayo sobre agar vacio.
    """
    if parche.size == 0 or min(parche.shape[:2]) < 12:
        return None
    gris = cv2.cvtColor(parche, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    # las colonias son mas oscuras que el agar en estas imagenes
    umbral, binaria = cv2.threshold(gris, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    etiquetas = label(binaria > 0)
    if etiquetas.max() == 0:
        return None

    # se toma la region que contiene el centro del parche
    h, w = gris.shape
    cy, cx = h // 2, w // 2
    id_centro = etiquetas[cy, cx]
    if id_centro == 0:
        # el centro quedo fuera; se usa la region mas cercana al centro
        props = regionprops(etiquetas)
        if not props:
            return None
        p = min(props, key=lambda q: (q.centroid[0] - cy) ** 2
                + (q.centroid[1] - cx) ** 2)
    else:
        p = next((q for q in regionprops(etiquetas) if q.label == id_centro), None)
    if p is None or p.area < 30:
        return None

    mascara = (etiquetas == p.label).astype(np.uint8)

    # nitidez del borde: gradiente medio sobre el contorno dilatado
    borde = cv2.morphologyEx(mascara, cv2.MORPH_GRADIENT,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    magnitud = np.sqrt(gx ** 2 + gy ** 2)
    nitidez = float(magnitud[borde > 0].mean()) if (borde > 0).any() else 0.0

    dentro = gris[mascara > 0].astype(np.float32)
    fuera = gris[mascara == 0].astype(np.float32)
    if dentro.size == 0 or fuera.size == 0:
        return None
    contraste = float((fuera.mean() - dentro.mean()) / (fuera.mean() + 1e-6))

    perimetro = p.perimeter if p.perimeter > 0 else 1.0
    circularidad = float(4 * np.pi * p.area / (perimetro ** 2))

    return {'solidez': float(p.solidity),
            'circularidad': min(circularidad, 1.5),
            'nitidez': nitidez,
            'contraste': contraste,
            'area': float(p.area)}


def medir_agar():
    filas = []
    for j in sorted(AGAR_DIR.rglob('*.json')):
        img_p = j.with_suffix('.jpg')
        if not img_p.exists():
            continue
        d = json.loads(j.read_text(encoding='utf-8'))
        if d.get('colonies_number', -1) < 0:
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        etiquetas = d.get('labels', [])[:MAX_POR_PLACA]
        for lab in etiquetas:
            r = max(lab['width'], lab['height']) / 2
            cx = lab['x'] + lab['width'] / 2
            cy = lab['y'] + lab['height'] / 2
            v = int(r * VENTANA)
            y0, y1 = int(cy - v), int(cy + v)
            x0, x1 = int(cx - v), int(cx + v)
            if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
                continue
            m = medir(img[y0:y1, x0:x1])
            if m:
                m.update({'origen': 'AGAR', 'placa': img_p.stem,
                          'especie': lab.get('class', '?')})
                filas.append(m)
    return filas


def medir_propias():
    filas = []
    for ruta in sorted(FOTOS.glob('*.jpg')):
        stem = ruta.stem
        csv_p = PUNTOS / f'{stem}_puntos.csv'
        if not csv_p.exists():
            continue
        img = cv2.imread(str(ruta))
        cx0, cy0, r0 = detect_plate(img)
        crop, mask = crop_plate(img, cx0, cy0, r0)

        r_uso = int(r0 * SHRINK)
        x1o, y1o = max(0, cx0 - r_uso), max(0, cy0 - r_uso)
        escala = crop.shape[1] / (2 * r_uso)

        with open(csv_p, newline='') as f:
            pts = [(float(q['x']), float(q['y'])) for q in csv.DictReader(f)]
        for x, y in pts[:MAX_POR_PLACA]:
            xr, yr = (x - x1o) * escala, (y - y1o) * escala
            v = int(RADIO_PROPIO * VENTANA)
            y0, y1 = int(yr - v), int(yr + v)
            x0, x1 = int(xr - v), int(xr + v)
            if y0 < 0 or x0 < 0 or y1 > crop.shape[0] or x1 > crop.shape[1]:
                continue
            if mask[min(max(int(yr), 0), mask.shape[0] - 1),
                    min(max(int(xr), 0), mask.shape[1] - 1)] == 0:
                continue
            m = medir(crop[y0:y1, x0:x1])
            if m:
                m.update({'origen': 'Actinomicetos', 'placa': stem,
                          'especie': 'actinomiceto'})
                filas.append(m)
    return filas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Midiendo colonias de AGAR...')
    a = medir_agar()
    print(f'  {len(a)} colonias medidas')
    print('Midiendo colonias propias de actinomicetos...')
    b = medir_propias()
    print(f'  {len(b)} colonias medidas\n')

    if not a or not b:
        print('Faltan datos en alguno de los dos conjuntos.')
        return

    df = pd.DataFrame(a + b)
    df.to_csv(OUT_DIR / 'descriptores.csv', index=False)

    descriptores = ['solidez', 'circularidad', 'nitidez', 'contraste']
    print(f'{"descriptor":<14} {"AGAR":>16} {"Actinomicetos":>16} {"p":>10}')
    print('-' * 60)
    resumen = []
    for d in descriptores:
        x = df[df['origen'] == 'AGAR'][d].dropna()
        y = df[df['origen'] == 'Actinomicetos'][d].dropna()
        # Mann-Whitney, que no asume normalidad
        u, p = stats.mannwhitneyu(x, y, alternative='two-sided')
        print(f'{d:<14} {x.mean():>8.3f} ±{x.std():<6.3f} '
              f'{y.mean():>8.3f} ±{y.std():<6.3f} {p:>10.2e}')
        resumen.append({'descriptor': d, 'agar_media': x.mean(),
                        'agar_sd': x.std(), 'actino_media': y.mean(),
                        'actino_sd': y.std(), 'p_valor': p})
    pd.DataFrame(resumen).to_csv(OUT_DIR / 'resumen_estadistico.csv', index=False)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    etiquetas = {
        'solidez': 'Solidez\n(1 = forma compacta)',
        'circularidad': 'Circularidad\n(1 = círculo perfecto)',
        'nitidez': 'Nitidez del borde\n(gradiente medio)',
        'contraste': 'Contraste con el agar',
    }
    for ax, d in zip(axes, descriptores):
        datos = [df[df['origen'] == o][d].dropna()
                 for o in ['AGAR', 'Actinomicetos']]
        partes = ax.violinplot(datos, showmedians=True)
        for cuerpo, color in zip(partes['bodies'], ['#4c72b0', '#dd8452']):
            cuerpo.set_facecolor(color)
            cuerpo.set_alpha(0.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['AGAR\n(compactas)', 'Actinomicetos\n(propias)'],
                           fontsize=9)
        ax.set_title(etiquetas[d], fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Morfología de las colonias: dataset público frente a '
                 'actinomicetos propios', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'comparacion.png', dpi=130, bbox_inches='tight')
    plt.close()

    print(f'\nResultados en {OUT_DIR}/')


if __name__ == '__main__':
    main()
