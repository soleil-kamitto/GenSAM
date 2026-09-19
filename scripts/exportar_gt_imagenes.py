"""
Exporta una imagen por placa con las cruces del conteo manual superpuestas
sobre la fotografia original, a partir de las coordenadas guardadas por
scripts/conteo_manual.py.

Uso:
    python scripts/exportar_gt_imagenes.py

Guarda:
    results/ground_truth_mis_fotos/imagenes/<placa>_gt.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMGS_DIR = Path('images/mis_fotos')
PUNTOS_DIR = Path('results/ground_truth_mis_fotos')
OUT_DIR = PUNTOS_DIR / 'imagenes'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archivos = sorted(PUNTOS_DIR.glob('*_puntos.csv'))

    for f in archivos:
        stem = f.name.replace('_puntos.csv', '')
        ruta_img = IMGS_DIR / f'{stem}.jpg'
        if not ruta_img.exists():
            print(f'{stem}: imagen original no encontrada, se omite')
            continue

        with open(f, newline='') as fh:
            pts = [(float(r['x']), float(r['y'])) for r in csv.DictReader(fh)]

        img = np.asarray(Image.open(ruta_img))
        alto, ancho = img.shape[:2]
        # sin titulo ni margenes incrustados, para que la imagen sirva tal cual
        # como panel dentro de las figuras del capitulo
        fig = plt.figure(figsize=(ancho / 200, alto / 200), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '+', color='red', markersize=10,
                    markeredgewidth=1.8)
        ax.axis('off')
        fig.savefig(OUT_DIR / f'{stem}_gt.png', dpi=200)
        plt.close(fig)
        print(f'{stem:<14} {len(pts):>4} marcas -> {OUT_DIR / f"{stem}_gt.png"}')

    print(f'\n{len(archivos)} imagenes guardadas en {OUT_DIR}/')


if __name__ == '__main__':
    main()
