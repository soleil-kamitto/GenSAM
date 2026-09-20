"""
Evalua el pipeline sobre ADBC, que son placas de otro laboratorio.

Por que importa. Todo lo ajustado hasta aqui se apoya en fotografias tomadas en
un mismo laboratorio, y la afirmacion central del trabajo es que unos parametros
ajustados en un montaje no transfieren a otro. Esa afirmacion no se puede
sostener midiendo solo sobre el montaje propio. ADBC son 369 placas de 24
especies tomadas con tres telefonos distintos y sin iluminacion normalizada, con
recuento anotado por placa, de modo que sirve como prueba ajena.

Que se mide. El error de conteo por placa, que es lo que le importa a un
laboratorio, y no la precision de deteccion por caja, que es lo que suelen
informar los trabajos de deteccion. Se separa por tramos de densidad porque el
efecto que se investiga depende de ella.

Submuestra. Sin GPU las 369 placas no son viables, asi que se toma un numero
igual de placas por tramo, con semilla fija. Estratificar importa aqui, porque
muestrear al azar daria sobre todo placas de densidad media y dejaria sin medir
justamente los extremos donde se espera ver el efecto.

Uso:
    python scripts/evaluar_adbc.py --por-tramo 10
    python scripts/evaluar_adbc.py --por-tramo 10 --mosaico
    python scripts/evaluar_adbc.py --solo-muestra
"""
import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from contar_mis_fotos import (
    detect_plate, crop_plate, flat_field,
    MIN_COLONY_AREA, MAX_COLONY_AREA, MIN_SOLIDITY,
)
from autocalibrar import calibrar

DATOS = Path('datasets/adbc')
TRAMOS = [(0, 10), (10, 30), (30, 60), (60, 150), (150, 250), (250, 10 ** 9)]
SEMILLA = 20260920


def nombre_tramo(lo, hi):
    return f'{lo + 1}-{hi}' if hi < 10 ** 8 else f'>{lo}'


def muestra(por_tramo):
    """Submuestra estratificada por densidad, reproducible."""
    gt = pd.read_csv(DATOS / 'conteo_por_placa.csv')
    gt['tramo'] = [next(nombre_tramo(lo, hi) for lo, hi in TRAMOS
                        if lo < c <= hi) for c in gt.colonias]
    rng = np.random.default_rng(SEMILLA)
    partes = []
    for lo, hi in TRAMOS:
        s = gt[gt.tramo == nombre_tramo(lo, hi)]
        if not len(s):
            continue
        n = min(por_tramo, len(s))
        partes.append(s.iloc[rng.choice(len(s), n, replace=False)])
    return pd.concat(partes).sort_values('colonias').reset_index(drop=True)


def contar(model, ruta, usar_mosaico):
    """
    Cuenta una placa con el pipeline auto-calibrado.

    Devuelve el conteo, o None si no se pudo localizar la placa, caso que se
    informa aparte en lugar de contarse como cero: un fallo de deteccion de
    placa es un fallo distinto de un fallo de conteo y mezclarlos ocultaria cual
    de los dos esta ocurriendo.
    """
    from cellSAM import segment_cellular_image
    img = cv2.imread(str(ruta))
    if img is None:
        return None, 'no se pudo leer'
    try:
        cx, cy, r = detect_plate(img)
    except Exception as e:
        return None, f'placa no localizada: {e}'
    if r is None or r <= 0:
        return None, 'placa no localizada'

    if usar_mosaico:
        from contar_mosaico import (recortar, cortes, detectar_baldosa,
                                    fusionar, SOLAPE)
        dim = 2400
        crop, mascara = recortar(img, cx, cy, r, dim)
        par = calibrar(crop, mascara)
        base = flat_field(crop, mascara) if par['aplicar_flat'] else crop
        escala = dim / 1200.0
        lim = (MIN_COLONY_AREA * escala ** 2, MAX_COLONY_AREA * escala ** 2)
        alto, ancho = base.shape[:2]
        fy, fx = cortes(alto, 2, SOLAPE), cortes(ancho, 2, SOLAPE)
        brutas = []
        for i, (y0, y1) in enumerate(fy):
            for j, (x0, x1) in enumerate(fx):
                sm = mascara[y0:y1, x0:x1]
                if (sm > 0).sum() < 500:
                    continue
                borde = (i == 0, i == len(fy) - 1, j == 0, j == len(fx) - 1)
                for d in detectar_baldosa(model, base[y0:y1, x0:x1],
                                          crop[y0:y1, x0:x1], sm,
                                          par['umbral'], par['hue_agar'],
                                          par['sat_minima'], lim, borde):
                    brutas.append({'cy': d['cy'] + y0, 'cx': d['cx'] + x0,
                                   'area': d['area']})
        return len(fusionar(brutas)), 'ok'

    crop, mascara = crop_plate(img, cx, cy, r)
    par = calibrar(crop, mascara)
    entrada = flat_field(crop, mascara) if par['aplicar_flat'] else crop
    try:
        seg, _, _ = segment_cellular_image(
            cv2.cvtColor(entrada, cv2.COLOR_BGR2RGB), model=model,
            normalize=True, postprocess=True,
            bbox_threshold=par['umbral'], device='cpu')
        if seg is None:
            return 0, 'ok'
    except (AttributeError, TypeError, ValueError):
        return 0, 'ok'
    seg = seg.copy()
    seg[mascara == 0] = 0
    n = sum(1 for p in regionprops(seg)
            if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
            and p.solidity >= MIN_SOLIDITY)
    return n, 'ok'


def resumir(df):
    """Error de conteo, global y por tramo de densidad."""
    ok = df[df.estado == 'ok'].copy()
    if not len(ok):
        print('Ninguna placa se pudo medir.')
        return
    ok['error'] = ok.contado - ok.colonias
    print()
    print(f'{"tramo":<10} {"placas":>7} {"real":>7} {"contado":>8} '
          f'{"MAE":>7} {"sesgo":>7} {"acierto":>8}')
    print('-' * 60)
    for lo, hi in TRAMOS:
        s = ok[ok.tramo == nombre_tramo(lo, hi)]
        if not len(s):
            continue
        mae = s.error.abs().mean()
        acierto = 1 - s.error.abs().sum() / max(s.colonias.sum(), 1)
        print(f'{nombre_tramo(lo, hi):<10} {len(s):>7} {s.colonias.sum():>7} '
              f'{s.contado.sum():>8} {mae:>7.1f} {s.error.mean():>7.1f} '
              f'{100 * acierto:>7.1f}%')
    print('-' * 60)
    mae = ok.error.abs().mean()
    acierto = 1 - ok.error.abs().sum() / max(ok.colonias.sum(), 1)
    print(f'{"TOTAL":<10} {len(ok):>7} {ok.colonias.sum():>7} '
          f'{ok.contado.sum():>8} {mae:>7.1f} {ok.error.mean():>7.1f} '
          f'{100 * acierto:>7.1f}%')
    fallos = df[df.estado != 'ok']
    if len(fallos):
        print(f'\n{len(fallos)} placas sin medir:')
        for _, f in fallos.iterrows():
            print(f'  {f.image_name}: {f.estado}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--por-tramo', type=int, default=10)
    ap.add_argument('--mosaico', action='store_true')
    ap.add_argument('--solo-muestra', action='store_true',
                    help='mostrar la submuestra sin ejecutar el conteo')
    args = ap.parse_args()

    sel = muestra(args.por_tramo)
    print(f'Submuestra estratificada: {len(sel)} placas, '
          f'{sel.colonias.sum()} colonias anotadas')
    print(f'Semilla {SEMILLA}, {args.por_tramo} placas por tramo')
    print(sel.groupby('tramo').colonias.agg(['count', 'sum', 'median'])
          .to_string())
    if args.solo_muestra:
        return

    from cellSAM import get_model
    print('\nCargando CellSAM...')
    model = get_model()
    print('Listo.\n')

    salida = Path('results/colonias/adbc')
    salida.mkdir(parents=True, exist_ok=True)
    nombre = f'adbc_{"mosaico" if args.mosaico else "placa_entera"}'

    print(f'{"placa":<18} {"tramo":<9} {"real":>6} {"contado":>8} {"error":>7}')
    print('-' * 52)
    filas = []
    for _, f in sel.iterrows():
        ruta = DATOS / 'imagenes' / f.image_name
        if not ruta.exists():
            filas.append({**f.to_dict(), 'contado': None,
                          'estado': 'imagen no descargada'})
            continue
        n, estado = contar(model, ruta, args.mosaico)
        filas.append({**f.to_dict(), 'contado': n, 'estado': estado})
        if estado == 'ok':
            print(f'{f.image_name:<18} {f.tramo:<9} {f.colonias:>6} '
                  f'{n:>8} {n - f.colonias:>+7}')
        else:
            print(f'{f.image_name:<18} {f.tramo:<9} {f.colonias:>6} '
                  f'{"-":>8}  {estado}')
        pd.DataFrame(filas).to_csv(salida / f'{nombre}.csv', index=False)

    df = pd.DataFrame(filas)
    df.to_csv(salida / f'{nombre}.csv', index=False)
    resumir(df)
    print(f'\nGuardado en {salida}/{nombre}.csv')


if __name__ == '__main__':
    main()
