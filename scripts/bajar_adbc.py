"""
Descarga el conjunto ADBC desde Figshare.

Que es. 369 fotografias de placas de 24 especies bacterianas, con 56.865
colonias anotadas, tomadas con tres telefonos distintos y sin iluminacion
normalizada ni protocolo estricto de posicionamiento. Publicado en Scientific
Data en 2023 con licencia CC BY 4.0.

Para que sirve aqui. Es la validacion externa que los trabajos recientes
declaran como pendiente, y que es justamente lo que este proyecto sostiene: que
unos parametros ajustados en un laboratorio no transfieren a otro con otro
telefono y otra luz. Medir el pipeline sobre capturas ajenas convierte esa
afirmacion en un resultado comprobable por terceros.

El archivo images.xls trae el recuento de unidades formadoras de colonia por
placa, de modo que permite evaluar conteo y no solo deteccion, que es la medida
que le importa a un laboratorio.

Uso:
    python scripts/bajar_adbc.py --solo-anotaciones
    python scripts/bajar_adbc.py
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ARTICULO = 22022540
API = f'https://api.figshare.com/v2/articles/{ARTICULO}'
DESTINO = Path('datasets/adbc')
ANOTACIONES = {'annot_COCO.json', 'annot_tab.csv', 'annot_tab.tsv',
               'annot_VOC_XML.zip', 'annot_YOLO.zip', 'images.xls',
               'TSV_to_COCO.py', 'bbox_placement_test.py'}


def listar():
    with urllib.request.urlopen(API, timeout=60) as r:
        return json.load(r)['files']


def bajar(f, carpeta, reintentos=3):
    """
    Descarga un archivo, saltandolo si ya esta completo.

    Se comprueba el tamano y no solo la existencia, porque una descarga cortada
    deja un archivo a medias que pasaria desapercibido y corromperia el analisis
    mas adelante.
    """
    destino = carpeta / f['name']
    if destino.exists() and destino.stat().st_size == f['size']:
        return 'ya estaba'
    for intento in range(reintentos):
        try:
            urllib.request.urlretrieve(f['download_url'], destino)
            if destino.stat().st_size == f['size']:
                return 'bajado'
            destino.unlink(missing_ok=True)
        except Exception as e:      # red inestable, se reintenta
            if intento == reintentos - 1:
                return f'FALLO: {e}'
            time.sleep(2 * (intento + 1))
    return 'FALLO: tamano incorrecto'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solo-anotaciones', action='store_true',
                    help='no bajar las 369 fotografias')
    args = ap.parse_args()

    imagenes = DESTINO / 'imagenes'
    imagenes.mkdir(parents=True, exist_ok=True)

    print('Consultando Figshare...')
    archivos = listar()
    anot = [f for f in archivos if f['name'] in ANOTACIONES]
    fotos = [f for f in archivos if f['name'].lower().endswith('.jpg')]
    print(f'{len(fotos)} fotografias, {len(anot)} archivos de anotacion')
    print(f'Total {sum(f["size"] for f in archivos) / 1e6:.0f} MB\n')

    print('--- Anotaciones ---')
    for f in anot:
        print(f'{f["name"]:<26} {bajar(f, DESTINO)}')

    if args.solo_anotaciones:
        print('\nAnotaciones listas. Las fotografias se bajan sin la opcion.')
        return

    print(f'\n--- Fotografias ({len(fotos)}) ---')
    fallos = 0
    for i, f in enumerate(fotos, 1):
        estado = bajar(f, imagenes)
        if estado.startswith('FALLO'):
            fallos += 1
            print(f'  {f["name"]}: {estado}')
        if i % 25 == 0 or i == len(fotos):
            print(f'  {i}/{len(fotos)}   fallos: {fallos}')

    bajadas = len(list(imagenes.glob('*.jpg')))
    print(f'\n{bajadas} fotografias en {imagenes}/')
    if fallos:
        print(f'{fallos} fallaron; volver a ejecutar salta las ya completas.')
        sys.exit(1)


if __name__ == '__main__':
    main()
