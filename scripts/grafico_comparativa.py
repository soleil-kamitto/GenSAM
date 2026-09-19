"""
Genera un grafico comparativo de todos los experimentos realizados.

Uso:
    python scripts/grafico_comparativa.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


GT = {
    'actinomicetos_1': (63, 64),
    'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60),
    'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25),
    'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67),
    'actinomicetos_8': (24, 13),
}

EXPERIMENTOS = {
    'normalize=True\n(sin postproceso)':   'results/colonias/experimentos/03_postprocess_off/summary.csv',
    'normalize=True\n(con postproceso)':   'results/colonias/experimentos/04_postprocess_on/summary.csv',
    'normalize=False\n(fine-tuning DETR)': 'results/colonias/finetuning/anchordeter_finetuned_summary.csv',
}

COLORES = {
    'Ground Truth':                          '#2ca02c',
    'normalize=True\n(sin postproceso)':     '#4C72B0',
    'normalize=True\n(con postproceso)':     '#DD8452',
    'normalize=False\n(fine-tuning DETR)':   '#9467bd',
}

OUT = Path('results/colonias')


def load_csv(path):
    df = pd.read_csv(path)
    return {row['image']: (int(row['plate_A']), int(row['plate_B']))
            for _, row in df.iterrows()}


def mae(counts, gt, plate_idx):
    errs = [abs(counts[stem][plate_idx] - gt[stem][plate_idx])
            for stem in gt if stem in counts]
    return np.mean(errs) if errs else float('nan')


def make_chart():
    data = {name: load_csv(p) for name, p in EXPERIMENTOS.items() if Path(p).exists()}
    if not data:
        print('No se encontraron CSVs de experimentos.')
        return

    stems    = sorted(GT.keys())
    x        = np.arange(len(stems))
    n_exp    = len(data)
    width    = 0.18
    labels_x = [s.replace('actinomicetos_', 'actin_') for s in stems]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Comparativa de experimentos CellSAM\n(conteo de colonias vs ground truth)',
                 fontsize=14, fontweight='bold')

    for ax_idx, (ax, plate_label, plate_i) in enumerate(
            [(axes[0], 'Placa A', 0), (axes[1], 'Placa B', 1)]):

        # GT
        gt_vals = [GT[s][plate_i] for s in stems]
        offset_gt = -(n_exp / 2) * width
        bars_gt = ax.bar(x + offset_gt, gt_vals, width,
                         label='Ground Truth', color=COLORES['Ground Truth'], alpha=0.9)
        ax.bar_label(bars_gt, padding=2, fontsize=7.5, fontweight='bold')

        # Experimentos
        for j, (exp_name, counts) in enumerate(data.items()):
            vals   = [counts.get(s, (0, 0))[plate_i] for s in stems]
            offset = (-(n_exp / 2) + j + 1) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=exp_name, color=COLORES.get(exp_name, f'C{j+2}'), alpha=0.85)
            ax.bar_label(bars, padding=2, fontsize=7.5)

        # MAE en el titulo del subplot
        mae_strs = []
        for exp_name, counts in data.items():
            m = mae(counts, GT, plate_i)
            short = exp_name.split('\n')[0]
            mae_strs.append(f'{short}: MAE={m:.1f}')
        ax.set_title(f'{plate_label}   |   ' + '   '.join(mae_strs), fontsize=11)

        all_vals = gt_vals + [counts.get(s, (0, 0))[plate_i]
                              for counts in data.values() for s in stems]
        ax.set_ylim(0, max(all_vals, default=1) * 1.3)
        ax.set_ylabel('Colonias contadas', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_x, rotation=30, ha='right', fontsize=9)

    plt.tight_layout()
    out = OUT / 'comparativa_experimentos.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Grafico guardado: {out}')

    # Tabla de errores
    print('\n=== ERROR POR IMAGEN (pred - GT) ===')
    header = f'{"imagen":<20}' + ''.join(
        f'{"A":>6}{"B":>6}' for _ in ['GT'] + list(data.keys()))
    print(header)
    for stem in stems:
        row = f'{stem:<20}'
        gt_a, gt_b = GT[stem]
        row += f'  {gt_a:>3}/{gt_b:<3}'
        for counts in data.values():
            ca, cb = counts.get(stem, (0, 0))
            row += f'  {ca-gt_a:>+3}/{cb-gt_b:<+3}'
        print(row)

    print('\n=== MAE POR EXPERIMENTO ===')
    for exp_name, counts in data.items():
        errs = []
        for stem in stems:
            if stem in counts:
                ga, gb = GT[stem]
                ca, cb = counts[stem]
                errs += [abs(ca - ga), abs(cb - gb)]
        short = exp_name.replace('\n', ' ')
        print(f'  {short:<40} MAE = {np.mean(errs):.2f} colonias/placa')


if __name__ == '__main__':
    make_chart()
