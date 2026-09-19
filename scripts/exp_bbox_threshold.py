"""
Experimento: sweep de bbox_threshold en CellSAM (normalize=True).

Prueba distintos valores de bbox_threshold y muestra el error absoluto
medio vs ground truth para elegir el valor optimo.

Uso:
    python scripts/exp_bbox_threshold.py images/placas/
"""

import sys
import csv
import warnings
import numpy as np
import cv2
from pathlib import Path
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from cellSAM import get_model, segment_cellular_image


MIN_COLONY_AREA = 300
MAX_COLONY_AREA = 50000
MIN_SOLIDITY    = 0.50
SUPPORTED       = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
OUT_DIR         = Path('results/colonias/experimentos/05_bbox_sweep')

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


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


def count_with_threshold(model, images_dir, threshold):
    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in SUPPORTED
                    and not p.name.startswith('ground'))
    results = {}
    for img_path in images:
        img    = cv2.imread(str(img_path))
        counts = []
        for cx, cy, r in detect_plates(img):
            crop, plate_mask = crop_plate(img, cx, cy, r)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            try:
                mask, _, _ = segment_cellular_image(
                    crop_rgb, model=model,
                    normalize=True, postprocess=False,
                    bbox_threshold=threshold, device='cpu',
                )
                if mask is None:
                    mask = np.zeros(crop.shape[:2], dtype=np.int32)
            except (AttributeError, TypeError):
                mask = np.zeros(crop.shape[:2], dtype=np.int32)
            mask = mask.copy()
            mask[plate_mask == 0] = 0
            props = regionprops(mask)
            n = sum(1 for p in props
                    if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                    and p.solidity >= MIN_SOLIDITY)
            counts.append(n)
        results[img_path.stem] = counts
    return results


def main():
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('images/placas')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_path = images_dir / 'ground_truth.csv'
    gt = {}
    if gt_path.exists():
        with open(gt_path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                stem = Path(row['image']).stem
                gt[stem] = (int(row['plate_A']), int(row['plate_B']))
        print(f'Ground truth: {len(gt)} imagenes\n')

    print('Cargando modelo CellSAM...')
    model = get_model()
    print('Listo.\n')

    sweep_results = []

    for thr in THRESHOLDS:
        print(f'bbox_threshold={thr:.2f} ...', end='  ', flush=True)
        counts = count_with_threshold(model, images_dir, thr)

        errs = []
        for stem, (ca, cb) in counts.items():
            ga, gb = gt.get(stem, (None, None))
            if ga is not None:
                errs += [abs(ca - ga), abs(cb - gb)]

        mae  = np.mean(errs) if errs else float('nan')
        mean_a = np.mean([v[0] for v in counts.values()])
        mean_b = np.mean([v[1] for v in counts.values()])
        print(f'MAE={mae:.1f}  media_A={mean_a:.1f}  media_B={mean_b:.1f}')

        sweep_results.append({
            'threshold': thr,
            'mae': mae,
            'media_A': mean_a,
            'media_B': mean_b,
            **{f'{stem}_A': v[0] for stem, v in counts.items()},
            **{f'{stem}_B': v[1] for stem, v in counts.items()},
        })

    # Guardar CSV
    df = pd.DataFrame(sweep_results)
    csv_out = OUT_DIR / 'sweep_resultados.csv'
    df.to_csv(csv_out, index=False)
    print(f'\nCSV guardado: {csv_out}')

    # Grafico MAE vs threshold
    thrs = [r['threshold'] for r in sweep_results]
    maes = [r['mae'] for r in sweep_results]
    best_idx = int(np.argmin(maes))
    best_thr = thrs[best_idx]
    best_mae = maes[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thrs, maes, 'o-', color='#4C72B0', linewidth=2, markersize=8)
    ax.axvline(best_thr, color='red', linestyle='--', alpha=0.7,
               label=f'Optimo: thr={best_thr}  MAE={best_mae:.1f}')
    ax.axhline(best_mae, color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('bbox_threshold', fontsize=12)
    ax.set_ylabel('Error absoluto medio (colonias/placa)', fontsize=12)
    ax.set_title('Efecto del bbox_threshold en el conteo de colonias', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    img_out = OUT_DIR / 'sweep_mae.png'
    plt.savefig(img_out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Grafico guardado: {img_out}')

    print(f'\n{"="*50}')
    print(f'  MEJOR threshold: {best_thr}  (MAE = {best_mae:.1f})')
    print(f'  Threshold actual: 0.40  (MAE = {maes[THRESHOLDS.index(0.40)]:.1f})')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
