"""
Compara los conteos del sistema (results/colonias/mis_fotos/summary.csv) contra
el ground truth manual de la investigadora
(results/ground_truth_mis_fotos/ground_truth.csv).

Criterios registrados con la investigadora
- M2-A-3 tiene GT=0 deliberado. El crecimiento en cadena de esa placa fue
  juzgado como no valido (no son colonias contables), de modo que las 29
  detecciones del sistema ahi cuentan como falsos positivos.
- Las placas con GT > 250 exceden el rango contable estandar en microbiologia
  (25-250, TNTC). Se reportan el MAE global y el MAE restringido a ese rango.

Uso:
    python scripts/comparar_gt_mis_fotos.py

Guarda:
    results/ground_truth_mis_fotos/comparacion.csv
    results/ground_truth_mis_fotos/comparacion.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GT_CSV   = Path('results/ground_truth_mis_fotos/ground_truth.csv')
SYS_CSV  = Path('results/colonias/mis_fotos/summary.csv')
OUT_DIR  = Path('results/ground_truth_mis_fotos')
TNTC     = 250   # limite superior del rango contable estandar

COLOR_PUNTO  = '#4c72b0'
COLOR_LINEA  = '#999999'
COLOR_TEXTO  = '#333333'


def main():
    gt  = pd.read_csv(GT_CSV).rename(columns={'count': 'manual'})
    sys_ = pd.read_csv(SYS_CSV)[['image', 'count']].rename(
        columns={'count': 'sistema'})
    df = gt.merge(sys_, on='image')

    df['error']     = df['sistema'] - df['manual']
    df['abs_error'] = df['error'].abs()
    df['accuracy']  = np.where(
        df[['manual', 'sistema']].max(axis=1) > 0,
        df[['manual', 'sistema']].min(axis=1)
        / df[['manual', 'sistema']].max(axis=1),
        1.0)
    df['rango_contable'] = df['manual'] <= TNTC

    df.to_csv(OUT_DIR / 'comparacion.csv', index=False)

    # ── metricas ─────────────────────────────────────────────────────────────
    mae_todo = df['abs_error'].mean()
    contables = df[df['rango_contable']]
    mae_cont = contables['abs_error'].mean()
    acc_cont = contables['accuracy'].mean()

    ancho = max(len(s) for s in df['image'])
    print(f'{"placa":<{ancho}} {"manual":>7} {"sistema":>8} {"err":>6} {"acc":>6}')
    print('-' * (ancho + 31))
    for _, r in df.iterrows():
        marca = '' if r['rango_contable'] else '  (TNTC)'
        print(f'{r["image"]:<{ancho}} {r["manual"]:>7} {r["sistema"]:>8} '
              f'{r["error"]:>+6} {r["accuracy"]*100:>5.0f}%{marca}')
    print('-' * (ancho + 31))
    print(f'MAE global (15 placas)            = {mae_todo:.2f} colonias/placa')
    print(f'MAE rango contable (GT<={TNTC}, n={len(contables)}) = '
          f'{mae_cont:.2f} colonias/placa')
    print(f'Accuracy media rango contable     = {acc_cont*100:.1f}%')
    print(f'(referencia dataset anterior: MAE = 5.94, 16 placas)')

    # ── figura: sistema vs manual con linea de identidad, dos paneles ────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    escasas = df[df['manual'] <= 35]
    paneles = [
        (axes[0], escasas, 35, 'Placas escasas y moderadas (GT hasta 35)'),
        (axes[1], df, 370, 'Todas las placas'),
    ]
    # puntos que merecen etiqueta directa (los desvios que cuentan la historia)
    etiquetar = {'M2-A-3', 'M1-D', 'RC73-A-2.5', 'RC73-B-2.5', 'M2-A'}

    for ax, datos, lim, titulo in paneles:
        ax.plot([0, lim], [0, lim], '--', color=COLOR_LINEA, linewidth=1.5,
                label='identidad (conteo perfecto)', zorder=1)
        ax.scatter(datos['manual'], datos['sistema'], s=55,
                   color=COLOR_PUNTO, zorder=2)
        for _, r in datos.iterrows():
            if r['image'] in etiquetar:
                ax.annotate(r['image'], (r['manual'], r['sistema']),
                            textcoords='offset points', xytext=(6, 5),
                            fontsize=8.5, color=COLOR_TEXTO)
        ax.set_xlim(-lim * 0.03, lim)
        ax.set_ylim(-lim * 0.03, lim)
        ax.set_xlabel('Conteo manual (ground truth)')
        ax.set_ylabel('Conteo del sistema')
        ax.set_title(titulo, fontsize=11)
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_aspect('equal')

    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle('Sistema vs conteo manual, 15 placas propias',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'comparacion.png', dpi=130, bbox_inches='tight')
    print(f'\nCSV:    {OUT_DIR / "comparacion.csv"}')
    print(f'Figura: {OUT_DIR / "comparacion.png"}')


if __name__ == '__main__':
    main()
