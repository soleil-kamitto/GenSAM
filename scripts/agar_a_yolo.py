"""
Convierte el dataset AGAR al formato que espera YOLO (ultralytics).

AGAR trae, junto a cada imagen, un archivo .json con las cajas en pixeles
absolutos y la esquina superior izquierda como origen. YOLO espera un .txt por
imagen, con una linea por objeto y las coordenadas normalizadas entre 0 y 1
tomando el centro de la caja:

    <clase> <x_centro> <y_centro> <ancho> <alto>

Ademas separa las imagenes en entrenamiento, validacion y prueba, y escribe el
archivo data.yaml que ultralytics necesita.

Dos decisiones importantes
  1. Las placas que AGAR marca con colonies_number = -1 son las que sus
     microbiologos consideraron incontables, y vienen sin cajas. Se excluyen,
     porque entrenar con ellas enseñaria al modelo que una placa llena de
     colonias no tiene ninguna.
  2. Por defecto se entrena con una sola clase, "colonia", en vez de las cinco
     especies. Para contar no hace falta distinguir especie, y juntar todo en
     una clase da mas ejemplos por clase y un modelo mas robusto. Con
     --por-especie se puede conservar la especie.

Uso:
    python scripts/agar_a_yolo.py images/agar_muestra
    python scripts/agar_a_yolo.py <carpeta_agar> --salida datasets/agar_yolo
    python scripts/agar_a_yolo.py <carpeta_agar> --por-especie

Genera:
    <salida>/images/{train,val,test}/*.jpg
    <salida>/labels/{train,val,test}/*.txt
    <salida>/data.yaml
"""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

SPLITS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEMILLA = 42


def leer_anotacion(ruta_json):
    """Devuelve (lista_de_cajas, especies, numero_declarado) o None si se excluye."""
    d = json.loads(ruta_json.read_text(encoding='utf-8'))
    n = d.get('colonies_number', -1)
    if n is None or n < 0:
        return None                      # placa incontable, sin anotar
    return d.get('labels', []), d.get('classes', []), n


def convertir(ruta_img, cajas, clases_idx, por_especie):
    """Pasa las cajas de AGAR a lineas en formato YOLO."""
    with Image.open(ruta_img) as im:
        ancho, alto = im.size

    lineas = []
    for c in cajas:
        x, y, w, h = c['x'], c['y'], c['width'], c['height']
        # AGAR da la esquina superior izquierda; YOLO quiere el centro
        xc, yc = (x + w / 2) / ancho, (y + h / 2) / alto
        wn, hn = w / ancho, h / alto
        # se recorta a [0,1] porque alguna caja puede salirse del borde
        xc, yc = min(max(xc, 0), 1), min(max(yc, 0), 1)
        wn, hn = min(max(wn, 0), 1), min(max(hn, 0), 1)
        if wn <= 0 or hn <= 0:
            continue
        idx = clases_idx[c['class']] if por_especie else 0
        lineas.append(f'{idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}')
    return lineas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('origen', help='carpeta con las imagenes y .json de AGAR')
    ap.add_argument('--salida', default='datasets/agar_yolo')
    ap.add_argument('--por-especie', action='store_true',
                    help='conservar la especie como clase, en vez de una sola')
    args = ap.parse_args()

    origen = Path(args.origen)
    salida = Path(args.salida)
    jsons = sorted(origen.rglob('*.json'))
    if not jsons:
        print(f'No se encontraron anotaciones en {origen}')
        return

    # primera pasada: inventario, para conocer especies y excluir incontables
    validos, excluidos = [], 0
    especies = Counter()
    for j in jsons:
        img = j.with_suffix('.jpg')
        if not img.exists():
            continue
        info = leer_anotacion(j)
        if info is None:
            excluidos += 1
            continue
        cajas, clases, n = info
        if not cajas:
            excluidos += 1
            continue
        for c in cajas:
            especies[c['class']] += 1
        validos.append((img, j, cajas, n))

    if not validos:
        print('No quedaron imagenes utilizables')
        return

    nombres = sorted(especies) if args.por_especie else ['colonia']
    clases_idx = {e: i for i, e in enumerate(sorted(especies))}

    print(f'imagenes utilizables : {len(validos)}')
    print(f'excluidas (incontables o sin cajas): {excluidos}')
    print(f'colonias anotadas    : {sum(especies.values())}')
    print(f'especies             : {dict(especies)}')
    print(f'clases para YOLO     : {nombres}\n')

    # reparto reproducible
    random.Random(SEMILLA).shuffle(validos)
    n_total = len(validos)
    n_train = int(n_total * SPLITS['train'])
    n_val = int(n_total * SPLITS['val'])
    reparto = {
        'train': validos[:n_train],
        'val': validos[n_train:n_train + n_val],
        'test': validos[n_train + n_val:],
    }

    for sub in SPLITS:
        (salida / 'images' / sub).mkdir(parents=True, exist_ok=True)
        (salida / 'labels' / sub).mkdir(parents=True, exist_ok=True)

    for sub, items in reparto.items():
        total_cajas = 0
        for img, _, cajas, _ in items:
            # el nombre incluye la carpeta de origen para no chocar entre subsets
            nombre = f'{img.parent.name}_{img.stem}'
            shutil.copy2(img, salida / 'images' / sub / f'{nombre}.jpg')
            lineas = convertir(img, cajas, clases_idx, args.por_especie)
            (salida / 'labels' / sub / f'{nombre}.txt').write_text(
                '\n'.join(lineas), encoding='utf-8')
            total_cajas += len(lineas)
        print(f'{sub:6}: {len(items):4} imagenes, {total_cajas:6} colonias')

    data_yaml = salida / 'data.yaml'
    data_yaml.write_text(
        f'# Dataset AGAR convertido a formato YOLO\n'
        f'# generado por scripts/agar_a_yolo.py\n'
        f'path: {salida.resolve().as_posix()}\n'
        f'train: images/train\n'
        f'val: images/val\n'
        f'test: images/test\n\n'
        f'nc: {len(nombres)}\n'
        f'names: {nombres}\n',
        encoding='utf-8')
    print(f'\nListo. Configuracion en {data_yaml}')


if __name__ == '__main__':
    main()
