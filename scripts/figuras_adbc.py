"""
Analiza y dibuja el resultado de la validacion externa sobre ADBC.

Produce tres cosas.

Primero, la comparacion entre lo contado y lo anotado, en ejes logaritmicos
porque el recuento abarca de 2 a 747 colonias por placa y en ejes lineales las
placas ralas quedarian todas amontonadas contra el origen.

Segundo, el error relativo por tramo de densidad. Se usa el error relativo y no
el absoluto porque equivocarse en cinco colonias sobre 8 y sobre 450 no es el
mismo fallo, y un promedio de errores absolutos sobre un conjunto tan desigual
lo dominan siempre las placas mas pobladas.

Tercero, la comparacion entre la placa entera y el mosaico, que es la prediccion
que se dejo registrada antes de ejecutar nada: el mosaico deberia mejorar en las
placas densas y resultar indiferente en las ralas.

Uso:
    python scripts/figuras_adbc.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

# Estos valores se repiten aqui en lugar de importarse de evaluar_adbc porque
# ese modulo arrastra CellSAM y, con el modelo ocupando el procesador en otras
# corridas, solo cargarlo tarda minutos. Un script de dibujo no debe depender
# del modelo.
TRAMOS = [(0, 10), (10, 30), (30, 60), (60, 150), (150, 250), (250, 10 ** 9)]
DIAMETRO_PLACA_MM = 90.0


def nombre_tramo(lo, hi):
    return f'{lo + 1}-{hi}' if hi < 10 ** 8 else f'>{lo}'


def area_minima_fisica(dim_recorte=1200, shrink=0.92, diam_minimo_mm=0.8):
    """Area minima en pixeles a partir del diametro declarado en milimetros."""
    px_por_mm = dim_recorte / (shrink * DIAMETRO_PLACA_MM)
    return float(np.pi * (diam_minimo_mm * px_por_mm / 2.0) ** 2)


ENTRADA = Path('results/colonias/adbc')
SALIDA = Path('results/colonias/adbc')
COLORES = ['#2c7bb6', '#5ab4d6', '#a6d96a', '#fdae61', '#e8703a', '#d7191c']


def cargar(nombre):
    f = ENTRADA / f'{nombre}.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[df.estado == 'ok'].copy()
    if not len(df):
        return None
    df['error'] = df.contado - df.colonias
    df['rel'] = df.error / df.colonias
    return df


def tabla(df, titulo):
    print(f'\n--- {titulo}  ({len(df)} placas) ---')
    print(f'{"tramo":<10} {"placas":>7} {"anotado":>8} {"contado":>8} '
          f'{"MAE":>7} {"err rel":>8} {"acierto":>8}')
    print('-' * 62)
    for lo, hi in TRAMOS:
        s = df[df.tramo == nombre_tramo(lo, hi)]
        if not len(s):
            continue
        acierto = 1 - s.error.abs().sum() / max(s.colonias.sum(), 1)
        print(f'{nombre_tramo(lo, hi):<10} {len(s):>7} {s.colonias.sum():>8} '
              f'{s.contado.sum():>8} {s.error.abs().mean():>7.1f} '
              f'{100 * s.rel.abs().median():>7.0f}% {100 * acierto:>7.1f}%')
    print('-' * 62)
    acierto = 1 - df.error.abs().sum() / max(df.colonias.sum(), 1)
    print(f'{"TOTAL":<10} {len(df):>7} {df.colonias.sum():>8} '
          f'{df.contado.sum():>8} {df.error.abs().mean():>7.1f} '
          f'{100 * df.rel.abs().median():>7.0f}% {100 * acierto:>7.1f}%')
    return acierto


def dibujar(entera, mosaico):
    n = 3 if mosaico is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.6))
    fig.suptitle('Validación externa sobre ADBC, placas de otro laboratorio',
                 fontsize=14, fontweight='bold')

    # 1. contado frente a anotado
    ax = axes[0]
    for color, (lo, hi) in zip(COLORES, TRAMOS):
        s = entera[entera.tramo == nombre_tramo(lo, hi)]
        if len(s):
            ax.scatter(s.colonias, s.contado.clip(lower=0.7), s=46,
                       color=color, edgecolor='white', linewidth=0.7,
                       label=nombre_tramo(lo, hi), zorder=3)
    lim = [1, max(entera.colonias.max(), entera.contado.max()) * 1.4]
    ax.plot(lim, lim, '--', color='#555', linewidth=1, zorder=1)
    ax.fill_between(lim, [x * 0.8 for x in lim], [x * 1.2 for x in lim],
                    color='#999', alpha=0.15, zorder=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('Colonias anotadas')
    ax.set_ylabel('Colonias contadas')
    ax.set_title('Conteo automático frente a la anotación\n'
                 'la banda gris es el margen del 20 %', fontsize=11)
    ax.legend(title='colonias por placa', fontsize=8, title_fontsize=8,
              loc='upper left')
    ax.grid(alpha=0.25, which='both')

    # 2. error relativo por tramo
    ax = axes[1]
    datos, etiquetas, colores = [], [], []
    for color, (lo, hi) in zip(COLORES, TRAMOS):
        s = entera[entera.tramo == nombre_tramo(lo, hi)]
        if len(s):
            datos.append(100 * s.rel.values)
            etiquetas.append(nombre_tramo(lo, hi))
            colores.append(color)
    # las etiquetas se ponen aparte porque el nombre del parametro cambio entre
    # versiones de matplotlib y asi el script sirve en las dos
    caja = ax.boxplot(datos, patch_artist=True, widths=0.6)
    ax.set_xticks(range(1, len(etiquetas) + 1))
    ax.set_xticklabels(etiquetas)
    for parche, color in zip(caja['boxes'], colores):
        parche.set_facecolor(color)
        parche.set_alpha(0.75)
    for mediana in caja['medians']:
        mediana.set_color('#222')
    ax.axhline(0, color='#555', linestyle='--', linewidth=1)
    ax.set_xlabel('Colonias por placa')
    ax.set_ylabel('Error relativo, %')
    ax.set_title('Error según la densidad de la placa\n'
                 'por encima de cero sobrecuenta', fontsize=11)
    ax.grid(alpha=0.25, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    # 3. mosaico frente a placa entera
    if mosaico is not None:
        ax = axes[2]
        junto = entera.merge(mosaico, on='image_name', suffixes=('_ent', '_mos'))
        for color, (lo, hi) in zip(COLORES, TRAMOS):
            s = junto[junto.tramo_ent == nombre_tramo(lo, hi)]
            if len(s):
                ax.scatter(100 * s.rel_ent.abs(), 100 * s.rel_mos.abs(),
                           s=46, color=color, edgecolor='white',
                           linewidth=0.7, label=nombre_tramo(lo, hi), zorder=3)
        tope = max(1, 100 * max(junto.rel_ent.abs().max(),
                                junto.rel_mos.abs().max())) * 1.1
        ax.plot([0, tope], [0, tope], '--', color='#555', linewidth=1)
        ax.set_xlim(0, tope)
        ax.set_ylim(0, tope)
        ax.set_xlabel('Error con la placa entera, %')
        ax.set_ylabel('Error con el mosaico, %')
        ax.set_title('Mosaico frente a placa entera\n'
                     'por debajo de la diagonal el mosaico gana', fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.25)

    plt.tight_layout()
    destino = SALIDA / 'validacion_adbc.png'
    plt.savefig(destino, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'\nFigura en {destino}')


def sensibilidad(nombre):
    """Cuanto depende el resultado del unico parametro discutible que queda."""
    f = ENTRADA / f'{nombre}_areas.csv'
    res = ENTRADA / f'{nombre}.csv'
    if not (f.exists() and res.exists()):
        return
    ar = pd.read_csv(f)
    df = pd.read_csv(res)
    anotado = df[df.estado == 'ok'].colonias.sum()
    print(f'\n--- Sensibilidad al área mínima, {nombre} ---')
    print(f'{"diámetro mm":>12} {"px2":>7} {"contado":>9} {"vs anotado":>12}')
    for mm in (0.4, 0.6, 0.8, 1.0, 1.35):
        piso = area_minima_fisica(diam_minimo_mm=mm)
        n = int((ar.area >= piso).sum())
        print(f'{mm:>12.2f} {piso:>7.0f} {n:>9} '
              f'{100 * n / max(anotado, 1) - 100:>+11.0f}%')
    print(f'{"anotado":>12} {"":>7} {anotado:>9}')


def main():
    entera = cargar('adbc_placa_entera')
    if entera is None:
        print('Todavía no hay resultados de la placa entera.')
        return
    mosaico = cargar('adbc_mosaico')

    tabla(entera, 'Placa entera')
    if mosaico is not None:
        tabla(mosaico, 'Mosaico')
        junto = entera.merge(mosaico, on='image_name', suffixes=('_ent', '_mos'))
        mejora = (junto.rel_mos.abs() < junto.rel_ent.abs())
        print(f'\n--- La predicción registrada ---')
        print('Se predijo: el mosaico mejora en placas densas y es indiferente')
        print('en las ralas.')
        print(f'{"tramo":<10} {"placas":>7} {"mejora mosaico":>16}')
        for lo, hi in TRAMOS:
            s = junto[junto.tramo_ent == nombre_tramo(lo, hi)]
            if len(s):
                m = mejora[s.index]
                print(f'{nombre_tramo(lo, hi):<10} {len(s):>7} '
                      f'{100 * m.mean():>15.0f}%')

    sensibilidad('adbc_placa_entera')
    dibujar(entera, mosaico)


if __name__ == '__main__':
    main()
