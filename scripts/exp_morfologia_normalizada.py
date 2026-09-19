"""
Comparacion morfologica entre AGAR y actinomicetos, con la escala controlada.

La primera version de esta medida (scripts/exp_morfologia_comparada.py) encontro
que las colonias propias son mas compactas, mas circulares, de borde mas nitido
y con mas contraste que las de AGAR, con significancia alta en los cuatro
descriptores. Pero esa comparacion tenia un fallo: las imagenes de AGAR rondan
los 4000 px y los recortes propios estan reescalados a 1200 px, y tanto la
nitidez del borde como la circularidad dependen de cuantos pixeles ocupa la
colonia. Con escalas distintas la comparacion no es valida.

Aqui se corrige llevando cada colonia, venga de donde venga, a un tamano fijo
antes de medirla. Asi los descriptores describen la forma y no la resolucion de
la camara.

Se anade ademas un descriptor pensado para el caracter filamentoso, la relacion
entre el perimetro real y el de un circulo de la misma area. Un borde con
proyecciones tiene mas perimetro del que le corresponde por su area.

Uso:
    python scripts/exp_morfologia_normalizada.py

Guarda:
    results/colonias/experimentos/17_morfologia_normalizada/descriptores.csv
    results/colonias/experimentos/17_morfologia_normalizada/comparacion.png
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
OUT_DIR = Path('results/colonias/experimentos/17_morfologia_normalizada')

LADO_NORMALIZADO = 96   # todo parche se lleva a este tamano antes de medir
VENTANA = 2.2
RADIO_PROPIO = 26
MAX_POR_PLACA = 60


def medir(parche):
    """Mide la colonia central de un parche ya normalizado de tamano."""
    if parche.size == 0:
        return None
    parche = cv2.resize(parche, (LADO_NORMALIZADO, LADO_NORMALIZADO),
                        interpolation=cv2.INTER_AREA)
    gris = cv2.cvtColor(parche, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    _, binaria = cv2.threshold(gris, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    etiquetas = label(binaria > 0)
    if etiquetas.max() == 0:
        return None

    c = LADO_NORMALIZADO // 2
    id_centro = etiquetas[c, c]
    props = regionprops(etiquetas)
    if id_centro == 0:
        if not props:
            return None
        p = min(props, key=lambda q: (q.centroid[0] - c) ** 2
                + (q.centroid[1] - c) ** 2)
    else:
        p = next((q for q in props if q.label == id_centro), None)
    if p is None or p.area < 40:
        return None

    mascara = (etiquetas == p.label).astype(np.uint8)
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
    circularidad = min(float(4 * np.pi * p.area / (perimetro ** 2)), 1.5)
    # exceso de perimetro respecto a un circulo de la misma area: mide cuanto
    # se aparta el contorno de lo liso, que es lo que se espera de una colonia
    # filamentosa
    perim_circulo = 2 * np.sqrt(np.pi * p.area)
    exceso_perimetro = float(perimetro / perim_circulo)

    return {'solidez': float(p.solidity),
            'circularidad': circularidad,
            'exceso_perimetro': min(exceso_perimetro, 4.0),
            'nitidez': nitidez,
            'contraste': contraste}


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
        for lab in d.get('labels', [])[:MAX_POR_PLACA]:
            r = max(lab['width'], lab['height']) / 2
            cx = lab['x'] + lab['width'] / 2
            cy = lab['y'] + lab['height'] / 2
            v = int(r * VENTANA)
            y0, y1, x0, x1 = int(cy - v), int(cy + v), int(cx - v), int(cx + v)
            if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
                continue
            m = medir(img[y0:y1, x0:x1])
            if m:
                m.update({'origen': 'AGAR', 'placa': img_p.stem})
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
        # se trabaja sobre la imagen original, sin el reescalado del pipeline,
        # para no perder resolucion antes de normalizar
        cx0, cy0, r0 = detect_plate(img)
        r_uso = int(r0 * SHRINK)
        # radio de colonia en pixeles de la imagen original
        radio_orig = RADIO_PROPIO * (2 * r_uso) / 1200

        with open(csv_p, newline='') as f:
            pts = [(float(q['x']), float(q['y'])) for q in csv.DictReader(f)]
        for x, y in pts[:MAX_POR_PLACA]:
            v = int(radio_orig * VENTANA)
            y0, y1, x0, x1 = int(y - v), int(y + v), int(x - v), int(x + v)
            if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
                continue
            m = medir(img[y0:y1, x0:x1])
            if m:
                m.update({'origen': 'Actinomicetos', 'placa': stem})
                filas.append(m)
    return filas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Escala normalizada: cada colonia se lleva a {LADO_NORMALIZADO} px\n')

    print('Midiendo AGAR...')
    a = medir_agar()
    print(f'  {len(a)} colonias')
    print('Midiendo actinomicetos propios...')
    b = medir_propias()
    print(f'  {len(b)} colonias\n')
    if not a or not b:
        print('Faltan datos.')
        return

    df = pd.DataFrame(a + b)
    df.to_csv(OUT_DIR / 'descriptores.csv', index=False)

    desc = ['solidez', 'circularidad', 'exceso_perimetro', 'nitidez', 'contraste']
    print(f'{"descriptor":<18} {"AGAR":>16} {"Actinomicetos":>16} {"p":>10}')
    print('-' * 64)
    resumen = []
    for d in desc:
        x = df[df['origen'] == 'AGAR'][d].dropna()
        y = df[df['origen'] == 'Actinomicetos'][d].dropna()
        _, p = stats.mannwhitneyu(x, y, alternative='two-sided')
        print(f'{d:<18} {x.mean():>8.3f} ±{x.std():<6.3f} '
              f'{y.mean():>8.3f} ±{y.std():<6.3f} {p:>10.2e}')
        resumen.append({'descriptor': d, 'agar_media': x.mean(),
                        'actino_media': y.mean(), 'p_valor': p})
    pd.DataFrame(resumen).to_csv(OUT_DIR / 'resumen_estadistico.csv', index=False)

    etiquetas = {
        'solidez': 'Solidez\n(1 = compacta)',
        'circularidad': 'Circularidad\n(1 = círculo)',
        'exceso_perimetro': 'Exceso de perímetro\n(1 = contorno liso)',
        'nitidez': 'Nitidez del borde',
        'contraste': 'Contraste con el agar',
    }
    fig, axes = plt.subplots(1, len(desc), figsize=(4.1 * len(desc), 4.6))
    for ax, d in zip(axes, desc):
        datos = [df[df['origen'] == o][d].dropna()
                 for o in ['AGAR', 'Actinomicetos']]
        partes = ax.violinplot(datos, showmedians=True)
        for cuerpo, color in zip(partes['bodies'], ['#4c72b0', '#dd8452']):
            cuerpo.set_facecolor(color)
            cuerpo.set_alpha(0.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['AGAR', 'Actinomicetos'], fontsize=9)
        ax.set_title(etiquetas[d], fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Morfología con la escala controlada: cada colonia medida al '
                 'mismo tamaño en píxeles', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'comparacion.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'\nResultados en {OUT_DIR}/')


if __name__ == '__main__':
    main()
