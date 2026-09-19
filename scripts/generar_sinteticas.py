"""
Generador de placas sinteticas de actinomicetos.

Motivacion: el fine-tuning de CellSAM ya se intento con las 16 placas del
conjunto de referencia y empeoro los resultados (MAE 12.75 ajustando el
decodificador de mascaras y 51.20 ajustando AnchorDETR, frente a 5.94 del modelo
sin tocar). La causa mas probable es la falta de datos. Ademas, los actinomicetos
no aparecen en ningun dataset publico: AGAR cubre S. aureus, E. coli,
P. aeruginosa, B. subtilis y C. albicans, todas de colonia compacta, mientras que
los actinomicetos son filamentosos y de borde difuso.

Este generador construye placas nuevas pegando colonias reales, recortadas de
las fotografias propias usando las coordenadas del conteo manual, sobre fondos
de placa reales tomados de las placas casi esteriles. La anotacion resultante es
exacta y gratuita, porque se conoce la posicion donde se pega cada colonia.

El enfoque sigue la linea de la aumentacion por copia y pegado validada en la
literatura de deteccion de colonias, con dos cuidados que importan para el
realismo:

  1. Las colonias se pegan con una mascara circular suavizada y mezcla por
     transparencia, para que no queden bordes recortados que el modelo aprenda
     a reconocer como artefactos.
  2. Se ajusta la intensidad de la colonia al fondo donde cae, porque las
     fotografias tienen un gradiente de iluminacion y una colonia pegada con el
     brillo de otra zona se veria falsa.

Uso:
    python scripts/generar_sinteticas.py --n 200
    python scripts/generar_sinteticas.py --n 50 --salida datasets/sinteticas_prueba

Guarda, en formato YOLO listo para entrenar:
    <salida>/images/{train,val}/*.jpg
    <salida>/labels/{train,val}/*.txt
    <salida>/data.yaml
    <salida>/muestra_visual.png
"""
import argparse
import csv
import json
import random
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from contar_mis_fotos import detect_plate, crop_plate

FOTOS = Path('images/mis_fotos')
PUNTOS = Path('results/ground_truth_mis_fotos')
LADO = 1200          # lado del recorte de placa, igual que en el pipeline
RADIO_DEFECTO = 26   # radio tipico de colonia en ese recorte
SEMILLA = 42

# placas casi esteriles: sirven de fondo limpio. Se excluyen sus pocas colonias
# recortando alrededor de las coordenadas conocidas.
FONDOS = ['M1-B', 'M1-C', 'M2-B', 'M1-D']


def cargar_recorte(stem):
    """Recorte circular de la placa, del mismo modo que el pipeline."""
    ruta = FOTOS / f'{stem}.jpg'
    if not ruta.exists():
        return None, None, None
    img = cv2.imread(str(ruta))
    cx, cy, r = detect_plate(img)
    crop, mask = crop_plate(img, cx, cy, r)
    return crop, mask, (cx, cy, r)


def puntos_en_recorte(stem, geom, forma):
    """Traslada las marcas del conteo manual al sistema del recorte."""
    ruta = PUNTOS / f'{stem}_puntos.csv'
    if not ruta.exists():
        return []
    cx, cy, r = geom
    from contar_mis_fotos import SHRINK
    r_uso = int(r * SHRINK)
    x1, y1 = max(0, cx - r_uso), max(0, cy - r_uso)
    escala = forma[1] / (2 * r_uso)
    with open(ruta, newline='') as f:
        pts = [(float(q['x']), float(q['y'])) for q in csv.DictReader(f)]
    return [((x - x1) * escala, (y - y1) * escala) for x, y in pts]


def extraer_colonias():
    """
    Recorta parches cuadrados centrados en cada colonia anotada.

    Devuelve una lista de parches BGR. Solo se toman colonias bien dentro de la
    placa, para que el parche no incluya el borde ni zonas enmascaradas.
    """
    parches = []
    for ruta in sorted(FOTOS.glob('*.jpg')):
        stem = ruta.stem
        crop, mask, geom = cargar_recorte(stem)
        if crop is None:
            continue
        pts = puntos_en_recorte(stem, geom, crop.shape)
        h, w = crop.shape[:2]
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            r = RADIO_DEFECTO
            if xi - r < 0 or yi - r < 0 or xi + r >= w or yi + r >= h:
                continue
            if mask[yi, xi] == 0:
                continue
            parche = crop[yi - r:yi + r, xi - r:xi + r].copy()
            if parche.shape[0] != 2 * r or parche.shape[1] != 2 * r:
                continue
            if not parche_utilizable(parche, r):
                continue
            parches.append(parche)
    return parches


def parche_utilizable(parche, r):
    """
    Descarta parches que producirian artefactos visibles al pegarlos.

    Los recortes tomados del punto quemado del centro de la placa, o de zonas
    con reflejos, arrastran medialunas muy brillantes. Al pegarlos aparecen
    halos que no existen en una colonia real, y un modelo entrenado con ellos
    aprenderia a reconocer el artefacto en lugar de la colonia.
    """
    gris = cv2.cvtColor(parche, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # el centro debe ser mas oscuro que el borde: asi es una colonia sobre agar
    centro = np.zeros(gris.shape, np.uint8)
    cv2.circle(centro, (r, r), int(r * 0.55), 1, -1)
    dentro = gris[centro == 1]
    fuera = gris[centro == 0]
    if dentro.size == 0 or fuera.size == 0:
        return False
    if dentro.mean() >= fuera.mean():
        return False

    # sin zonas quemadas ni reflejos fuertes
    if gris.max() > 245 or gris.std() > 45:
        return False
    return True


def cargar_fondos():
    """Fondos de placa reales, con sus escasas colonias borradas por difuminado."""
    fondos = []
    for stem in FONDOS:
        crop, mask, geom = cargar_recorte(stem)
        if crop is None:
            continue
        pts = puntos_en_recorte(stem, geom, crop.shape)
        limpio = crop.copy()
        # se tapan las pocas colonias reales con el agar de alrededor
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            r = RADIO_DEFECTO + 8
            y0, y1 = max(0, yi - r), min(limpio.shape[0], yi + r)
            x0, x1 = max(0, xi - r), min(limpio.shape[1], xi + r)
            if y1 - y0 < 3 or x1 - x0 < 3:
                continue
            parche = limpio[y0:y1, x0:x1]
            limpio[y0:y1, x0:x1] = cv2.medianBlur(parche, 2 * (r // 2) + 1)
        fondos.append((limpio, mask))
    return fondos


def pegar(lienzo, parche, cx, cy):
    """
    Pega una colonia con borde suavizado y ajuste de intensidad al fondo.

    Sin el suavizado quedaria un cuadrado visible, y sin el ajuste de intensidad
    la colonia conservaria el brillo de la zona de donde se recorto, lo que en
    una imagen con gradiente de iluminacion se nota de inmediato.
    """
    r = parche.shape[0] // 2
    h, w = lienzo.shape[:2]
    y0, y1 = cy - r, cy + r
    x0, x1 = cx - r, cx + r
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return False

    destino = lienzo[y0:y1, x0:x1].astype(np.float32)
    fuente = parche.astype(np.float32)

    # ajuste de intensidad: se iguala el brillo del anillo exterior del parche
    # (que es agar) con el del fondo donde va a caer
    mascara = np.zeros((2 * r, 2 * r), np.float32)
    cv2.circle(mascara, (r, r), int(r * 0.78), 1.0, -1)
    mascara = cv2.GaussianBlur(mascara, (0, 0), sigmaX=r * 0.18)
    anillo = mascara < 0.05
    if anillo.sum() > 10:
        desfase = destino[anillo].mean(axis=0) - fuente[anillo].mean(axis=0)
        fuente = np.clip(fuente + desfase, 0, 255)

    m3 = mascara[..., None]
    lienzo[y0:y1, x0:x1] = (destino * (1 - m3) + fuente * m3).astype(np.uint8)
    return True


def generar_una(fondos, parches, rng, densidad):
    """Crea una placa sintetica con el numero de colonias pedido."""
    fondo, mask = fondos[rng.randrange(len(fondos))]
    lienzo = fondo.copy()
    h, w = lienzo.shape[:2]
    centro = (w // 2, h // 2)
    radio_util = int(min(h, w) / 2 * 0.93)

    colocadas = []
    intentos = 0
    while len(colocadas) < densidad and intentos < densidad * 40:
        intentos += 1
        ang = rng.uniform(0, 2 * np.pi)
        # raiz cuadrada para repartir de forma uniforme por area, no por radio
        rad = radio_util * np.sqrt(rng.random())
        cx = int(centro[0] + rad * np.cos(ang))
        cy = int(centro[1] + rad * np.sin(ang))

        parche = parches[rng.randrange(len(parches))]
        escala = rng.uniform(0.7, 1.35)
        lado = int(parche.shape[0] * escala) // 2 * 2
        if lado < 12:
            continue
        p = cv2.resize(parche, (lado, lado), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            p = cv2.flip(p, rng.randrange(-1, 2))
        r = lado // 2

        # se permite algo de solape, porque en placas densas las colonias se
        # tocan, pero no superposicion total
        if any(np.hypot(cx - qx, cy - qy) < (r + qr) * 0.55
               for qx, qy, qr in colocadas):
            continue
        if mask[min(max(cy, 0), h - 1), min(max(cx, 0), w - 1)] == 0:
            continue
        if pegar(lienzo, p, cx, cy):
            colocadas.append((cx, cy, r))

    return lienzo, colocadas


def variar_captura(img, mask, rng):
    """
    Simula que la fotografia se tomo con otro telefono, otra luz y otro encuadre.

    El objetivo es que un modelo entrenado con estas placas no dependa de las
    condiciones de un laboratorio concreto. Se midio que entre los montajes
    disponibles el tono del agar va de 45 a 110, el gradiente de iluminacion de
    99 a 122 y el ruido de 18 a 30, de modo que la variacion aplicada cubre ese
    rango y algo mas.

    Cuatro efectos, todos presentes en fotografias reales de celular:
      balance de blancos, que desplaza el tono
      gradiente de iluminacion, el punto caliente del transiluminador o la
        sombra de una lampara lateral
      viñeteo, oscurecimiento hacia las esquinas propio de lentes pequenas
      ruido y compresion, peores cuanto mas barata la camara
    """
    out = img.astype(np.float32)
    h, w = out.shape[:2]

    # balance de blancos: se escalan los canales por separado
    for c in range(3):
        out[..., c] *= rng.uniform(0.82, 1.18)

    # gradiente de iluminacion desde un punto cualquiera de la placa
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx, fy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    dist = np.sqrt((gx - fx) ** 2 + (gy - fy) ** 2)
    dist /= (dist.max() + 1e-6)
    fuerza = rng.uniform(0.0, 0.45)
    out *= (1 + fuerza * (0.5 - dist))[..., None]

    # viñeteo centrado
    cy2, cx2 = h / 2, w / 2
    rad = np.sqrt((gx - cx2) ** 2 + (gy - cy2) ** 2)
    rad /= (rad.max() + 1e-6)
    out *= (1 - rng.uniform(0.0, 0.35) * rad ** 2)[..., None]

    # exposicion global
    out *= rng.uniform(0.85, 1.15)

    # ruido de sensor. Se usa un generador de numpy sembrado desde rng, para
    # que el resultado siga siendo reproducible
    ruido_rng = np.random.default_rng(rng.randrange(2 ** 32))
    out += ruido_rng.normal(0, rng.uniform(1.5, 9.0), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # desenfoque leve, por pulso de mano o enfoque imperfecto
    if rng.random() < 0.35:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 1.1))

    # compresion JPEG agresiva, como la de un telefono de gama baja
    if rng.random() < 0.5:
        calidad = int(rng.uniform(55, 92))
        _, buf = cv2.imencode('.jpg', out,
                              [cv2.IMWRITE_JPEG_QUALITY, calidad])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    out[mask == 0] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200, help='placas a generar')
    ap.add_argument('--salida', default='datasets/sinteticas_yolo')
    ap.add_argument('--min-densidad', type=int, default=5)
    ap.add_argument('--max-densidad', type=int, default=320)
    ap.add_argument('--variar-captura', action='store_true',
                    help='simular distintos telefonos, luces y encuadres')
    args = ap.parse_args()

    salida = Path(args.salida)
    rng = random.Random(SEMILLA)

    print('Extrayendo colonias reales de las fotografias anotadas...')
    parches = extraer_colonias()
    print(f'  {len(parches)} colonias recortadas')
    if not parches:
        print('No hay colonias disponibles. Revisa el conteo manual.')
        return

    print('Preparando fondos de placa...')
    fondos = cargar_fondos()
    print(f'  {len(fondos)} fondos')
    if not fondos:
        print('No hay fondos disponibles.')
        return

    for sub in ['train', 'val']:
        (salida / 'images' / sub).mkdir(parents=True, exist_ok=True)
        (salida / 'labels' / sub).mkdir(parents=True, exist_ok=True)

    print(f'\nGenerando {args.n} placas sinteticas...')
    total_colonias = 0
    muestras = []
    for i in range(args.n):
        # la densidad se sortea en escala logaritmica, para tener tanto placas
        # escasas como densas, igual que en el conjunto real
        densidad = int(np.exp(rng.uniform(np.log(args.min_densidad),
                                          np.log(args.max_densidad))))
        img, colocadas = generar_una(fondos, parches, rng, densidad)
        if args.variar_captura:
            # la mascara de la placa es la del fondo usado; se recupera del
            # propio lienzo, donde lo exterior quedo en negro
            mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
            img = variar_captura(img, mask, rng)
        sub = 'train' if i < args.n * 0.85 else 'val'
        nombre = f'sintetica_{i:05d}'

        cv2.imwrite(str(salida / 'images' / sub / f'{nombre}.jpg'), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        h, w = img.shape[:2]
        lineas = [f'0 {cx/w:.6f} {cy/h:.6f} {2*r/w:.6f} {2*r/h:.6f}'
                  for cx, cy, r in colocadas]
        (salida / 'labels' / sub / f'{nombre}.txt').write_text('\n'.join(lineas))
        total_colonias += len(colocadas)
        if len(muestras) < 6:
            muestras.append((img.copy(), colocadas))
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{args.n}  ({total_colonias} colonias hasta ahora)')

    (salida / 'data.yaml').write_text(
        f'# Placas sinteticas de actinomicetos\n'
        f'# generadas por scripts/generar_sinteticas.py\n'
        f'path: {salida.resolve().as_posix()}\n'
        f'train: images/train\n'
        f'val: images/val\n\n'
        f'nc: 1\n'
        f"names: ['colonia']\n")

    # muestra visual, para poder juzgar el realismo de un vistazo
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (img, colocadas) in zip(axes.ravel(), muestras):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for cx, cy, r in colocadas:
            ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                    edgecolor='red', linewidth=0.8))
        ax.set_title(f'{len(colocadas)} colonias', fontsize=10)
        ax.axis('off')
    fig.suptitle('Placas sinteticas generadas, con su anotacion',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(salida / 'muestra_visual.png', dpi=110, bbox_inches='tight')
    plt.close()

    print(f'\n{args.n} placas, {total_colonias} colonias anotadas')
    print(f'media de {total_colonias/args.n:.0f} colonias por placa')
    print(f'\nDataset en {salida}/')
    print(f'Revisa el realismo en {salida}/muestra_visual.png')


if __name__ == '__main__':
    main()
