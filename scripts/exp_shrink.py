"""
Experimento de margen de recorte: corre el pipeline completo con un valor de
SHRINK dado y escribe en su propia carpeta, sin tocar los resultados de
results/colonias/mis_fotos/ (que son los documentados en el capitulo).

Ademas compara contra el conteo manual de referencia y reporta MAE y acierto
agregado, para poder decidir el margen con datos.

Motivacion: la curva de recuperacion medida sobre el ground truth manual
(856 colonias) muestra que el recorte al 86 % deja fuera el 21 % de las
colonias reales, pero ampliarlo introduce falsos positivos por la condensacion
pegada a la pared de la placa. Este script permite medir ese compromiso.

Uso:
    python scripts/exp_shrink.py 0.92
    python scripts/exp_shrink.py 0.90 --out results/colonias/experimentos/09_shrink_090

Guarda (por defecto en results/colonias/experimentos/shrink_<valor>/):
    summary.csv       conteos del sistema por placa
    comparacion.csv   contraste contra el conteo manual, con MAE y acierto
    <placa>_count.png paneles por placa
"""
import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from contar_mis_fotos import (
    detect_plate, crop_plate, flat_field, scout, segment, draw_overlay,
    SUPPORTED,
)
from cellSAM import get_model

GT_CSV = Path('results/ground_truth_mis_fotos/ground_truth.csv')
TNTC = 250


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shrink', type=float, help='radio de recorte, p.ej. 0.92')
    ap.add_argument('--imagenes', default='images/mis_fotos')
    ap.add_argument('--out', default=None)
    ap.add_argument('--solo', nargs='*', default=None,
                    help='procesar solo estas placas, p.ej. --solo RC73-A-2.5')
    ap.add_argument('--reanudar', action='store_true',
                    help='saltar las placas que ya tienen resultado guardado')
    args = ap.parse_args()

    shrink = args.shrink
    out_dir = Path(args.out) if args.out else Path(
        f'results/colonias/experimentos/shrink_{shrink:.2f}'.replace('.', ''))
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in Path(args.imagenes).iterdir()
                    if p.suffix.lower() in SUPPORTED
                    and not p.name.startswith('ground'))

    if args.solo:
        images = [p for p in images if p.stem in args.solo]

    # resultados previos, para poder reanudar sin repetir lo ya calculado
    previos = {}
    csv_previo = out_dir / 'summary.csv'
    if csv_previo.exists():
        for _, r in pd.read_csv(csv_previo).iterrows():
            previos[r['image']] = r.to_dict()

    if args.reanudar:
        pendientes = [p for p in images if p.stem not in previos]
        print(f'Reanudando: {len(previos)} placas ya hechas, '
              f'{len(pendientes)} pendientes')
        images = pendientes

    print(f'Experimento de margen: shrink={shrink}')
    print(f'Salida: {out_dir}')
    print(f'Imagenes a procesar: {len(images)}\n')
    if not images:
        print('Nada pendiente.')
        return
    print('Cargando modelo CellSAM...')
    model = get_model()
    print('Listo.\n')

    header = f'{"imagen":<20} {"thr":>5} {"conteo":>6} {"tinta":>6} {"borde":>6}'
    print(header)
    print('-' * len(header))

    rows = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        cx, cy, r = detect_plate(img)
        # shrink explicito, para no depender del valor por defecto del modulo
        crop, plate_mask = crop_plate(img, cx, cy, r, shrink=shrink)
        flat = flat_field(crop, plate_mask)
        _, thr, _ = scout(flat, plate_mask)
        seg_mask, valid, n_ink, n_rim = segment(model, flat, crop,
                                                plate_mask, thr)
        n = len(valid)
        print(f'{img_path.stem:<20} {thr:>5.2f} {n:>6} {n_ink:>6} {n_rim:>6}')
        rows.append({'image': img_path.stem, 'count': n,
                     'bbox_threshold': thr, 'descartadas_tinta': n_ink,
                     'descartadas_borde': n_rim})

        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        fig.suptitle(f'{img_path.name}   (shrink={shrink})',
                     fontsize=13, fontweight='bold')
        axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Placa original')
        axes[1].imshow(cv2.cvtColor(flat, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Corregida (entrada al modelo)')
        axes[2].imshow(draw_overlay(crop, seg_mask, valid))
        axes[2].set_title(f'n={n}  thr={thr:.2f}')
        for a in axes:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f'{img_path.stem}_count.png', dpi=120,
                    bbox_inches='tight')
        plt.close()

    # combina lo recien calculado con lo que ya hubiera guardado
    for r in rows:
        previos[r['image']] = r
    df = pd.DataFrame(list(previos.values())).sort_values('image')
    df.to_csv(out_dir / 'summary.csv', index=False)

    # ── contraste contra el conteo manual ────────────────────────────────────
    if GT_CSV.exists():
        gt = pd.read_csv(GT_CSV).rename(columns={'count': 'manual'})
        comp = gt.merge(df[['image', 'count']].rename(
            columns={'count': 'sistema'}), on='image')
        comp['error'] = comp['sistema'] - comp['manual']
        comp['abs_error'] = comp['error'].abs()
        comp['rango_contable'] = comp['manual'] <= TNTC
        comp.to_csv(out_dir / 'comparacion.csv', index=False)

        contables = comp[comp['rango_contable']]
        mae_cont = contables['abs_error'].mean()
        mae_glob = comp['abs_error'].mean()
        acierto = 1 - comp['abs_error'].sum() / comp['manual'].sum()

        print('-' * len(header))
        print(f'\n{"placa":<14} {"manual":>7} {"sistema":>8} {"error":>6}')
        for _, r in comp.iterrows():
            print(f'{r["image"]:<14} {r["manual"]:>7} {r["sistema"]:>8} '
                  f'{r["error"]:>+6}')
        print()
        print(f'MAE rango contable (n={len(contables)}) = {mae_cont:.2f}')
        print(f'MAE global (n={len(comp)})             = {mae_glob:.2f}')
        print(f'Acierto agregado                      = {acierto*100:.1f}%')
        print(f'\n(referencia shrink=0.86: MAE contable 5.46, global 15.40, '
              f'acierto 73.0%)')

    print(f'\nResultados en {out_dir}/')


if __name__ == '__main__':
    main()
