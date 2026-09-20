"""
Separa las colonias que el detector unio en una sola region.

Motivo. Al medir el alargamiento de los componentes oscuros se vio que en las
placas densas las colonias se tocan y forman cadenas. Cuando eso pasa el
detector propone una sola region para varias colonias, de modo que el conteo se
queda corto justo donde mas importa acertar, porque son las placas con mas
unidades formadoras de colonia.

Metodo. Dentro de la region se calcula la transformada de distancia, que en cada
pixel vale lo que dista del borde de la region. Una colonia redonda tiene un
unico maximo en su centro, con valor igual a su radio. Dos colonias pegadas
tienen dos maximos separados por un cuello donde la distancia baja. Se toman
esos maximos como marcadores y se reparte la region entre ellos por cuencas
hidrograficas, que es el procedimiento habitual para este caso desde los
trabajos clasicos de morfologia matematica y el que usan herramientas de conteo
de colonias como OpenCFU.

Salvaguarda. La separacion por cuencas siempre puede partir cualquier cosa, asi
que aplicada sin condiciones fragmentaria tambien las colonias unicas de contorno
irregular. Aqui un corte solo se acepta si mejora lo que se pretende mejorar:

  cada trozo debe alcanzar el area minima de una colonia
  cada trozo debe ser mas compacto que la region de la que sale

La segunda condicion es la que evita el abuso. Partir una colonia redonda da dos
medias lunas, que son menos compactas que el disco original, de modo que el corte
se rechaza solo. Partir dos colonias pegadas da dos discos, que son mas compactos
que la figura de ocho original, de modo que el corte se acepta.

Control negativo. La opcion --control aplica el procedimiento a un conjunto donde
las colonias estan bien separadas. Si la regla mide lo que dice medir, alli no
debe partir casi nada. Es la misma comprobacion que descarto el descriptor de
forma para la rotulacion, y conviene hacerla antes de adoptar cualquier regla.

Uso como modulo:
    from separar_pegadas import separar
    nuevas = separar(seg, props_validas)

Uso directo, para medir cuanto parte en cada conjunto:
    python scripts/separar_pegadas.py images/mis_fotos_lote3
    python scripts/separar_pegadas.py images/placas --control
"""
import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

SEPARACION_MINIMA = 0.55   # fraccion del radio equivalente entre dos maximos
MEJORA_MINIMA = 1.02       # cuanto debe subir la compacidad para aceptar el corte


def compacidad(mascara):
    """
    Que tan cerca esta una figura de ser un disco, entre 0 y 1.

    Se usa el cociente entre el area y la del circulo de igual perimetro, que
    vale 1 para el disco y baja para cualquier otra forma. Sirve aqui porque lo
    que se quiere comprobar es justamente si los trozos se parecen mas a discos
    que la figura de la que salen.
    """
    m = mascara.astype(np.uint8)
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return 0.0
    c = max(contornos, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    if per <= 0:
        return 0.0
    return float(4.0 * np.pi * cv2.contourArea(c) / (per ** 2))


def separar_region(region_mask, area_minima):
    """
    Devuelve una lista de mascaras, una por colonia, o la original sin tocar.

    Si no hay motivo para partir, o si el corte no mejora la compacidad, se
    devuelve la region entera en una lista de un elemento.
    """
    m = region_mask.astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    radio = np.sqrt(m.sum() / np.pi)

    # dos maximos mas cercanos que esto pertenecen a la misma colonia, y solo
    # reflejan que su contorno no es perfectamente liso
    aparte = max(3, int(round(radio * SEPARACION_MINIMA)))
    picos = peak_local_max(dist, min_distance=aparte, labels=m,
                           exclude_border=False)
    if len(picos) < 2:
        return [region_mask]

    marcadores = np.zeros(m.shape, dtype=np.int32)
    for i, (y, x) in enumerate(picos, start=1):
        marcadores[y, x] = i
    etiquetas = watershed(-dist, marcadores, mask=m)

    trozos = []
    for i in range(1, len(picos) + 1):
        t = etiquetas == i
        if t.sum() < area_minima:
            return [region_mask]        # un trozo demasiado pequeno invalida
        trozos.append(t)

    base = compacidad(region_mask)
    if any(compacidad(t) < base * MEJORA_MINIMA for t in trozos):
        return [region_mask]            # el corte no mejora nada, se rechaza
    return trozos


def separar(label_mask, props, area_minima):
    """
    Aplica la separacion a cada region valida.

    Devuelve la lista de mascaras finales y cuantos cortes se aceptaron, que es
    lo que hay que vigilar en el control negativo.
    """
    finales, cortes = [], 0
    for p in props:
        trozos = separar_region(label_mask == p.label, area_minima)
        if len(trozos) > 1:
            cortes += len(trozos) - 1
        finales.extend(trozos)
    return finales, cortes


def _medir(carpeta, control):
    """Mide cuanto partiria el procedimiento sobre un conjunto dado."""
    from contar_mis_fotos import (
        detect_plate, crop_plate, flat_field,
        MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY, SUPPORTED,
    )
    from autocalibrar import calibrar, es_tinta
    from quitar_rotulacion import limpiar_rotulacion
    from cellSAM import get_model, segment_cellular_image

    rutas = sorted(p for p in Path(carpeta).iterdir()
                   if p.suffix.lower() in SUPPORTED
                   and not p.name.startswith('ground'))
    print(f'Conjunto: {carpeta}   {len(rutas)} placas')
    if control:
        print('Modo control: aqui la regla NO deberia partir casi nada.\n')
    print('Cargando CellSAM...')
    model = get_model()
    print('Listo.\n')

    cab = f'{"placa":<14} {"antes":>6} {"cortes":>7} {"despues":>8} {"subida":>7}'
    print(cab)
    print('-' * len(cab))

    filas = []
    for ruta in rutas:
        img = cv2.imread(str(ruta))
        cx, cy, r = detect_plate(img)
        crop, mascara = crop_plate(img, cx, cy, r)
        par = calibrar(crop, mascara)
        limpio, _, mascara = limpiar_rotulacion(crop, mascara, par['hue_agar'])
        entrada = flat_field(limpio, mascara) if par['aplicar_flat'] else limpio

        try:
            seg, _, _ = segment_cellular_image(
                cv2.cvtColor(entrada, cv2.COLOR_BGR2RGB), model=model,
                normalize=True, postprocess=True,
                bbox_threshold=par['umbral'], device='cpu')
            if seg is None:
                seg = np.zeros(crop.shape[:2], dtype=np.int32)
        except (AttributeError, TypeError, ValueError):
            seg = np.zeros(crop.shape[:2], dtype=np.int32)
        seg = seg.copy()
        seg[mascara == 0] = 0

        hsv = cv2.cvtColor(limpio, cv2.COLOR_BGR2HSV)
        validas = [p for p in regionprops(seg)
                   if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                   and p.solidity >= MIN_SOLIDITY
                   and not es_tinta(hsv, seg, p.label, par['hue_agar'],
                                    par['sat_minima'])]

        antes = len(validas)
        _, cortes = separar(seg, validas, MIN_COLONY_AREA)
        despues = antes + cortes
        subida = 100.0 * cortes / max(antes, 1)
        print(f'{ruta.stem:<14} {antes:>6} {cortes:>7} {despues:>8} '
              f'{subida:>6.1f}%')
        filas.append({'placa': ruta.stem, 'antes': antes, 'cortes': cortes,
                      'despues': despues, 'subida_pct': round(subida, 1)})

    df = pd.DataFrame(filas)
    salida = Path('results/colonias/experimentos/21_separar_pegadas')
    salida.mkdir(parents=True, exist_ok=True)
    nombre = Path(carpeta).name + ('_control' if control else '')
    df.to_csv(salida / f'{nombre}.csv', index=False)

    print('-' * len(cab))
    total_antes = df['antes'].sum()
    total_cortes = df['cortes'].sum()
    print(f'\nAntes {total_antes}, cortes {total_cortes}, '
          f'despues {total_antes + total_cortes}')
    print(f'Subida global: {100.0 * total_cortes / max(total_antes, 1):.1f}%')
    print(f'Placas donde no parte nada: {(df["cortes"] == 0).sum()} de {len(df)}')
    print(f'\nGuardado en {salida}/{nombre}.csv')


def _prueba():
    """
    Comprueba la regla sobre figuras construidas, donde la respuesta se sabe.

    Interesa sobre todo la primera mitad: una colonia unica alargada no debe
    partirse, porque ese es el modo de fallo que haria subir el conteo sin
    motivo. Se llega hasta una elipse de 2,5 a 1, mas alargada de lo que suele
    ser una colonia de actinomiceto.
    """
    def disco(circulos, lado=340):
        m = np.zeros((lado, lado), np.uint8)
        for (x, y, r) in circulos:
            cv2.circle(m, (x, y), r, 1, -1)
        return m.astype(bool)

    def elipse(ejes, angulo=0, lado=340):
        m = np.zeros((lado, lado), np.uint8)
        cv2.ellipse(m, (170, 170), ejes, angulo, 0, 360, 1, -1)
        return m.astype(bool)

    casos = [
        ('colonia redonda', disco([(170, 170, 45)]), 1),
        ('colonia elipse 1,5 a 1', elipse((66, 44)), 1),
        ('colonia elipse 2 a 1', elipse((80, 40)), 1),
        ('colonia elipse 2,5 a 1 inclinada', elipse((90, 36), 30), 1),
        ('dos iguales, solape leve', disco([(132, 170, 45), (208, 170, 45)]), 2),
        ('dos iguales, apenas tocandose',
         disco([(126, 170, 45), (214, 170, 45)]), 2),
        ('dos distintas, tocandose',
         disco([(140, 170, 50), (212, 170, 32)]), 2),
        ('tres en cadena',
         disco([(95, 170, 38), (170, 170, 38), (245, 170, 38)]), 3),
        ('cuatro en racimo',
         disco([(140, 140, 35), (200, 140, 35),
                (140, 200, 35), (200, 200, 35)]), 4),
    ]

    print(f'{"caso":<34} {"esperado":>9} {"obtenido":>9} {"compac":>7}')
    print('-' * 64)
    fallos = 0
    for nombre, m, esperado in casos:
        obtenido = len(separar_region(m, area_minima=300))
        if obtenido != esperado:
            fallos += 1
        print(f'{nombre:<34} {esperado:>9} {obtenido:>9} '
              f'{compacidad(m):>7.2f} {"ok" if obtenido == esperado else "FALLA"}')
    print('-' * 64)
    print(f'{len(casos) - fallos} de {len(casos)} correctos')
    return fallos == 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('imagenes', nargs='?', default='images/mis_fotos_lote3')
    ap.add_argument('--control', action='store_true',
                    help='conjunto donde la regla no deberia partir casi nada')
    ap.add_argument('--prueba', action='store_true',
                    help='comprobar la regla sobre figuras construidas')
    a = ap.parse_args()
    if a.prueba:
        sys.exit(0 if _prueba() else 1)
    _medir(a.imagenes, a.control)
