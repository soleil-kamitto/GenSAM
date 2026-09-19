"""
Conteo de colonias con la configuracion optima encontrada:
  normalize=True, postprocess=True, bbox_threshold=0.80

Uso:
    python scripts/contar_optimo.py [images/placas/]

Guarda:
    results/colonias/experimentos/06_thr_optimo/summary.csv
    results/colonias/experimentos/06_thr_optimo/actinomicetos_X_count.png
"""
import sys
import csv
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from skimage.measure import regionprops

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from cellSAM import get_model, segment_cellular_image

# ── parametros optimos ────────────────────────────────────────────────────────
BBOX_THRESHOLD  = 0.80
NORMALIZE       = True
POSTPROCESS     = True
MIN_COLONY_AREA = 300
MAX_COLONY_AREA = 50000
MIN_SOLIDITY    = 0.50
SUPPORTED       = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
OUT_DIR         = Path('results/colonias/experimentos/06_thr_optimo')

GT = {
    'actinomicetos_1': (63, 64), 'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60), 'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25), 'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67), 'actinomicetos_8': (24, 13),
}


def detect_plates(img_bgr):
    h, w = img_bgr.shape[:2]
    portrait = h > w
    plates = []
    for i in range(2):
        if portrait:
            y0, y1 = i * h // 2, (i + 1) * h // 2
            half = img_bgr[y0:y1, :]
            ox, oy = 0, y0
        else:
            x0, x1 = i * w // 2, (i + 1) * w // 2
            half = img_bgr[:, x0:x1]
            ox, oy = x0, 0
        hh, hw = half.shape[:2]
        gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
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
    crop = img_bgr[y1:y2, x1:x2].copy()
    hc, wc = crop.shape[:2]
    mask = np.zeros((hc, wc), dtype=np.uint8)
    cv2.circle(mask, (cx - x1, cy - y1), r_use, 255, -1)
    crop[mask == 0] = 0
    return crop, mask


def segment(model, crop_bgr, plate_mask):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    try:
        seg_mask, _, _ = segment_cellular_image(
            crop_rgb, model=model,
            normalize=NORMALIZE, postprocess=POSTPROCESS,
            bbox_threshold=BBOX_THRESHOLD, device='cpu',
        )
        if seg_mask is None:
            seg_mask = np.zeros(crop_rgb.shape[:2], dtype=np.int32)
    except (AttributeError, TypeError):
        seg_mask = np.zeros(crop_bgr.shape[:2], dtype=np.int32)
    seg_mask = seg_mask.copy()
    seg_mask[plate_mask == 0] = 0
    props = regionprops(seg_mask)
    valid = [p for p in props
             if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY]
    return seg_mask, valid


def draw_overlay(crop_bgr, label_mask, valid_props):
    rng = np.random.default_rng(seed=42)
    overlay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for p in valid_props:
        color = rng.uniform(0.2, 1.0, size=3)
        region = label_mask == p.label
        overlay[region] = overlay[region] * 0.3 + color * 0.7
    return overlay


def main():
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('images/placas')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in SUPPORTED
                    and not p.name.startswith('ground'))

    print(f'Configuracion: normalize={NORMALIZE}, postprocess={POSTPROCESS}, '
          f'bbox_threshold={BBOX_THRESHOLD}')
    print(f'Imagenes encontradas: {len(images)}\n')
    print('Cargando modelo CellSAM...')
    model = get_model()
    print('Listo.\n')

    rows = []
    all_errs = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        plates = detect_plates(img)
        gt_a, gt_b = GT.get(img_path.stem, (None, None))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'{img_path.name}   |   GT: A={gt_a}  B={gt_b}',
            fontsize=13, fontweight='bold'
        )

        counts = []
        for i, (cx, cy, r) in enumerate(plates):
            plate_name = 'AB'[i]
            gt_val = (gt_a, gt_b)[i]
            crop, plate_mask = crop_plate(img, cx, cy, r)
            seg_mask, valid = segment(model, crop, plate_mask)
            n = len(valid)
            counts.append(n)

            err_str = f'  (err={n - gt_val:+d})' if gt_val is not None else ''
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            axes[i, 0].imshow(crop_rgb)
            axes[i, 0].set_title(f'Placa {plate_name} original  (GT={gt_val})',
                                  fontsize=11)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(draw_overlay(crop, seg_mask, valid))
            axes[i, 1].set_title(
                f'Placa {plate_name}  detectadas={n}{err_str}', fontsize=11)
            axes[i, 1].axis('off')

            if gt_val is not None:
                all_errs.append(abs(n - gt_val))

        plt.tight_layout()
        out_img = OUT_DIR / f'{img_path.stem}_count.png'
        plt.savefig(out_img, dpi=120, bbox_inches='tight')
        plt.close()

        ca, cb = counts[0], counts[1]
        rows.append({'image': img_path.stem, 'plate_A': ca, 'plate_B': cb})

        err_a = f'{ca - gt_a:+d}' if gt_a is not None else '?'
        err_b = f'{cb - gt_b:+d}' if gt_b is not None else '?'
        print(f'{img_path.stem:<22}  A={ca:>3} (err={err_a:>3})   '
              f'B={cb:>3} (err={err_b:>3})')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'summary.csv', index=False)

    mae = np.mean(all_errs) if all_errs else float('nan')
    print(f'\nMAE = {mae:.2f} colonias/placa')
    print(f'CSV guardado: {OUT_DIR / "summary.csv"}')
    print(f'Imagenes guardadas en: {OUT_DIR}/')


if __name__ == '__main__':
    main()
