"""
Experimento: efecto del postprocesamiento en CellSAM.

Corre dos variantes sobre todas las placas (normalize=True fijo):
  1. CellSAM  postprocess=False  (sin postprocesamiento)
  2. CellSAM  postprocess=True   (con postprocesamiento)

Uso:
    python scripts/exp_postprocess.py images/placas/

Resultados en results/colonias/:
    cellsam_post_off/              imagenes y CSV sin postprocesamiento
    cellsam_post_on/               imagenes y CSV con postprocesamiento
    comparacion_postprocesamiento.png
    resumen_postprocesamiento.csv
"""

import sys
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from skimage.measure import regionprops
from cellSAM import get_model, segment_cellular_image


# ── PARAMETROS ────────────────────────────────────────────────────────────────
MIN_COLONY_AREA = 300
MAX_COLONY_AREA = 50000
MIN_SOLIDITY    = 0.50
NORMALIZE       = True
BBOX_THRESHOLD  = 0.4

SUPPORTED = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
BASE_OUT  = Path('results/colonias')

EXPERIMENTS = [
    {'name': 'cellsam_post_off', 'postprocess': False},
    {'name': 'cellsam_post_on',  'postprocess': True},
]

# ── MODELO ────────────────────────────────────────────────────────────────────
_model = None

def get_cached_model():
    global _model
    if _model is None:
        print('Cargando modelo CellSAM...')
        _model = get_model()
        print('Modelo listo.\n')
    return _model


# ── DETECCION Y RECORTE ───────────────────────────────────────────────────────

def detect_plates(img_bgr):
    h, w     = img_bgr.shape[:2]
    portrait = h > w
    plates   = []
    for i in range(2):
        if portrait:
            y0, y1 = i * h // 2, (i + 1) * h // 2
            half   = img_bgr[y0:y1, :]
            ox, oy = 0, y0
        else:
            x0, x1 = i * w // 2, (i + 1) * w // 2
            half   = img_bgr[:, x0:x1]
            ox, oy = x0, 0
        hh, hw  = half.shape[:2]
        gray    = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(hh, hw), param1=60, param2=25,
            minRadius=int(min(hh, hw) * 0.30),
            maxRadius=int(min(hh, hw) * 0.52),
        )
        if circles is not None:
            best = np.round(circles[0][0]).astype(int)
            cx, cy, r = int(best[0]) + ox, int(best[1]) + oy, int(best[2])
        else:
            cx, cy, r = hw // 2 + ox, hh // 2 + oy, int(min(hh, hw) * 0.43)
        plates.append((cx, cy, r))
    return plates


def crop_plate(img_bgr, cx, cy, r, shrink=0.86):
    r_use = int(r * shrink)
    x1 = max(0, cx - r_use);  y1 = max(0, cy - r_use)
    x2 = min(img_bgr.shape[1], cx + r_use)
    y2 = min(img_bgr.shape[0], cy + r_use)
    crop   = img_bgr[y1:y2, x1:x2].copy()
    hc, wc = crop.shape[:2]
    mask   = np.zeros((hc, wc), dtype=np.uint8)
    cv2.circle(mask, (cx - x1, cy - y1), r_use, 255, -1)
    crop[mask == 0] = 0
    return crop, mask


# ── SEGMENTACION ──────────────────────────────────────────────────────────────

def segment(crop_bgr, plate_mask, postprocess):
    model    = get_cached_model()
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    try:
        mask, _, _ = segment_cellular_image(
            crop_rgb, model=model,
            normalize=NORMALIZE, postprocess=postprocess,
            bbox_threshold=BBOX_THRESHOLD, device='cpu',
        )
        if mask is None:
            mask = np.zeros(crop_bgr.shape[:2], dtype=np.int32)
    except (AttributeError, TypeError) as e:
        print(f'    [aviso] {e}')
        mask = np.zeros(crop_bgr.shape[:2], dtype=np.int32)

    mask = mask.copy()
    mask[plate_mask == 0] = 0
    props = regionprops(mask)
    valid = [p for p in props if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY]
    return mask, valid


def draw_overlay(crop_bgr, valid_props, label_mask):
    overlay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for prop in valid_props:
        region = label_mask == prop.label
        overlay[region] = overlay[region] * 0.45 + np.array([0.15, 0.85, 0.35]) * 0.55
    return overlay


# ── PROCESAR UNA IMAGEN ───────────────────────────────────────────────────────

def process_image(img_path, exp, output_dir):
    img    = cv2.imread(str(img_path))
    plates = detect_plates(img)
    counts = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    label = 'postprocess=True' if exp['postprocess'] else 'postprocess=False'
    fig.suptitle(f"{img_path.name}  [{label}]", fontsize=12, fontweight='bold')

    for i, (cx, cy, r) in enumerate(plates):
        name       = ['A', 'B'][i]
        crop, mask = crop_plate(img, cx, cy, r)
        lbl, valid = segment(crop, mask, exp['postprocess'])
        n          = len(valid)
        counts.append(n)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(crop_rgb)
        axes[i, 0].set_title(f'Placa {name} original', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(draw_overlay(crop, valid, lbl))
        axes[i, 1].set_title(f'Placa {name}: {n} colonias', fontsize=10)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f'{img_path.stem}.png', dpi=120, bbox_inches='tight')
    plt.close()
    return counts


# ── GRAFICO COMPARATIVO ───────────────────────────────────────────────────────

def make_chart(all_results, gt, output_dir):
    img_names   = [r['name'] for r in next(iter(all_results.values()))]
    short_names = [n.replace('actinomicetos_', 'actin_') for n in img_names]
    x     = np.arange(len(img_names))
    width = 0.20

    colors = {'cellsam_post_off': '#4C72B0', 'cellsam_post_on': '#DD8452'}
    labels = {
        'cellsam_post_off': 'CellSAM postprocess=False',
        'cellsam_post_on':  'CellSAM postprocess=True',
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    fig.suptitle('Efecto del postprocesamiento en CellSAM (normalize=True)',
                 fontsize=14, fontweight='bold')

    for ax, plate_idx, plate_label in [(ax1, 0, 'Placa A'), (ax2, 1, 'Placa B')]:
        gt_vals = [gt.get(r['name'], (0, 0))[plate_idx]
                   for r in next(iter(all_results.values()))]

        b_gt = ax.bar(x - width * 1.5, gt_vals, width,
                      label='Ground truth', color='#2ca02c', alpha=0.85)
        ax.bar_label(b_gt, padding=2, fontsize=8)

        for j, (exp_name, records) in enumerate(all_results.items()):
            vals = [r['counts'][plate_idx] for r in records]
            offset = width * (j - 0.5)
            bars = ax.bar(x + offset, vals, width,
                          label=labels[exp_name], color=colors[exp_name], alpha=0.85)
            ax.bar_label(bars, padding=2, fontsize=8)

        all_vals = gt_vals + [r['counts'][plate_idx]
                              for exp in all_results.values() for r in exp]
        ax.set_ylabel('Colonias contadas')
        ax.set_title(plate_label, fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(all_vals, default=1) * 1.28)
        ax.grid(axis='y', alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    out = output_dir / 'comparacion_postprocesamiento.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Grafico guardado: {out}')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('images/placas')
    gt_path    = images_dir / 'ground_truth.csv'

    gt = {}
    if gt_path.exists():
        with open(gt_path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                stem = Path(row['image']).stem
                gt[stem] = (int(row['plate_A']), int(row['plate_B']))
        print(f'Ground truth: {len(gt)} imagenes\n')

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for exp in EXPERIMENTS:
        exp_dir = BASE_OUT / exp['name']
        exp_dir.mkdir(parents=True, exist_ok=True)
        records = []

        # Saltar si ya esta completo
        summary_path = exp_dir / 'summary.csv'
        if summary_path.exists():
            with open(summary_path, newline='') as f:
                done = sum(1 for _ in csv.reader(f)) - 1
            if done >= len(images):
                print(f'  {exp["name"]}: ya completo, cargando CSV.')
                with open(summary_path, newline='') as f:
                    for row in csv.DictReader(f):
                        records.append({
                            'name':   row['image'],
                            'counts': [int(row['plate_A']), int(row['plate_B'])],
                        })
                all_results[exp['name']] = records
                continue

        label = 'postprocess=True' if exp['postprocess'] else 'postprocess=False'
        print(f"\n{'='*55}")
        print(f"  Metodo: {label}")
        print(f"{'='*55}")

        for idx, img_path in enumerate(images, 1):
            print(f'  [{idx}/{len(images)}] {img_path.name}', end='  ')
            counts = process_image(img_path, exp, exp_dir)
            ga, gb = gt.get(img_path.stem, (None, None))
            if ga is not None:
                print(f'A={counts[0]}(err={counts[0]-ga:+d})  B={counts[1]}(err={counts[1]-gb:+d})')
            else:
                print(f'A={counts[0]}  B={counts[1]}')
            records.append({'name': img_path.stem, 'counts': counts})

            with open(summary_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['image', 'plate_A', 'plate_B'])
                for rec in records:
                    w.writerow([rec['name'], rec['counts'][0], rec['counts'][1]])

        all_results[exp['name']] = records

    # Grafico y CSV comparativo
    print('\nGenerando comparativa...')
    make_chart(all_results, gt, BASE_OUT)

    rows = []
    for img_stem in [r['name'] for r in next(iter(all_results.values()))]:
        ga, gb = gt.get(img_stem, (None, None))
        row    = {'imagen': img_stem, 'gt_A': ga, 'gt_B': gb}
        for exp_name, records in all_results.items():
            rec    = next(r for r in records if r['name'] == img_stem)
            ca, cb = rec['counts']
            row[f'{exp_name}_A'] = ca
            row[f'{exp_name}_B'] = cb
            if ga is not None:
                row[f'err_{exp_name}_A'] = ca - ga
                row[f'err_{exp_name}_B'] = cb - gb
        rows.append(row)

    csv_out = BASE_OUT / 'resumen_postprocesamiento.csv'
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f'CSV guardado: {csv_out}')

    print('\n' + '='*55)
    print('  ERROR ABSOLUTO MEDIO vs GROUND TRUTH')
    print('='*55)
    for exp in EXPERIMENTS:
        errs = []
        for rec in all_results[exp['name']]:
            if rec['name'] in gt:
                ga, gb = gt[rec['name']]
                errs.append(abs(rec['counts'][0] - ga))
                errs.append(abs(rec['counts'][1] - gb))
        label = 'postprocess=True ' if exp['postprocess'] else 'postprocess=False'
        if errs:
            print(f'  {label}  {np.mean(errs):.1f} colonias/placa')
    print('='*55)


if __name__ == '__main__':
    main()
