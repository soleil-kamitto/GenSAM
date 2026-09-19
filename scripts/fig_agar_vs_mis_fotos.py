"""
Genera material visual para inspeccionar el dataset AGAR y compararlo con las
fotografias propias.

AGAR es un dataset publico de 18.000 fotografias de placas con 336.442 colonias
anotadas por microbiologos. La muestra libre trae 40 imagenes en cuatro
condiciones de captura. Interesa saber cuanto se parecen a las fotografias del
proyecto, porque de eso depende que un modelo entrenado con AGAR sirva aqui.

Uso:
    python scripts/fig_agar_vs_mis_fotos.py

Guarda:
    docs/figuras/agar_vs_mis_fotos.png     comparacion lado a lado
    docs/figuras/agar_mosaico_<subset>.png mosaico de cada condicion
"""
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

AGAR_DIR = Path('images/agar_muestra')
MIS_FOTOS = Path('images/mis_fotos')
LOTE2 = Path('images/mis_fotos_lote2')
OUT_DIR = Path('docs/figuras')

SUBSETS = ['bright', 'dark', 'vague', 'lower-resolution']


def cargar(ruta, ancho=900):
    """Carga una imagen reducida, para que las figuras no pesen de mas."""
    img = Image.open(ruta).convert('RGB')
    if img.width > ancho:
        alto = int(round(img.height * ancho / img.width))
        img = img.resize((ancho, alto), Image.LANCZOS)
    return np.asarray(img), img.width / Image.open(ruta).width


def mosaico_subset(subset):
    """Mosaico de las 10 imagenes de una condicion, con su numero de colonias."""
    carpeta = AGAR_DIR / subset
    jpgs = sorted(carpeta.glob('*.jpg'))
    if not jpgs:
        return
    n = len(jpgs)
    cols = 5
    filas = (n + cols - 1) // cols
    fig, axes = plt.subplots(filas, cols, figsize=(4 * cols, 4.2 * filas))
    axes = np.atleast_1d(axes).ravel()
    for ax, ruta in zip(axes, jpgs):
        img, _ = cargar(ruta, ancho=500)
        ax.imshow(img)
        meta = json.loads((ruta.with_suffix('.json')).read_text())
        especie = ', '.join(meta.get('classes', []))
        ax.set_title(f'{ruta.stem}   {meta["colonies_number"]} colonias\n{especie}',
                     fontsize=9)
        ax.axis('off')
    for ax in axes[len(jpgs):]:
        ax.axis('off')
    fig.suptitle(f'AGAR, condicion "{subset}"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    salida = OUT_DIR / f'agar_mosaico_{subset}.png'
    plt.savefig(salida, dpi=95, bbox_inches='tight')
    plt.close()
    print(f'  {salida}')


def comparacion():
    """
    Una imagen de AGAR con sus cajas anotadas, junto a dos fotografias propias,
    para juzgar el parecido entre ambos conjuntos.
    """
    # se elige una placa densa y con anotaciones completas. Ojo que AGAR marca
    # con colonies_number = -1 las placas que sus microbiologos consideraron
    # incontables, y esas vienen sin cajas
    ruta_agar = AGAR_DIR / 'bright' / '518.jpg'
    meta = json.loads((ruta_agar.with_suffix('.json')).read_text())
    img_agar, escala = cargar(ruta_agar, ancho=900)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8))

    axes[0].imshow(img_agar)
    for lab in meta['labels']:
        axes[0].add_patch(Rectangle(
            (lab['x'] * escala, lab['y'] * escala),
            lab['width'] * escala, lab['height'] * escala,
            fill=False, edgecolor='red', linewidth=0.7))
    axes[0].set_title(
        f'AGAR (dataset publico)\n{meta["colonies_number"]} colonias anotadas, '
        f'{", ".join(meta["classes"])}', fontsize=11)

    img_mia, _ = cargar(MIS_FOTOS / 'RC73-A-2.5.jpg', ancho=900)
    axes[1].imshow(img_mia)
    axes[1].set_title('Fotografía propia, primer lote\nRC73-A-2.5, 352 colonias '
                      'contadas a mano', fontsize=11)

    img_nueva, _ = cargar(LOTE2 / 'NRC73-A-2.5-.jpg', ancho=900)
    axes[2].imshow(img_nueva)
    axes[2].set_title('Fotografía propia, segundo lote\nNRC73-A-2.5, sin conteo '
                      'de referencia', fontsize=11)

    for a in axes:
        a.axis('off')
    fig.suptitle('El dataset AGAR frente a las fotografías del proyecto',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    salida = OUT_DIR / 'agar_vs_mis_fotos.png'
    plt.savefig(salida, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  {salida}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Generando figuras...')
    comparacion()
    for s in SUBSETS:
        mosaico_subset(s)
    print('\nListo.')


if __name__ == '__main__':
    main()
