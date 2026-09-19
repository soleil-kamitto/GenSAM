"""
Conteo adaptivo con estimacion clasica de densidad (CLAHE + umbral adaptativo).

Pipeline por placa:
  1. CLAHE + umbral adaptativo (~10 ms) -> estima n_colonias clasico
  2. Segun esa estimacion elige bbox_threshold para CellSAM:
       n_adapt < 28             -> 0.65  (placa escasa o dificil de ver)
       28 <= n_adapt < 55       -> 0.75  (densidad moderada)
       n_adapt >= 55            -> 0.80  (placa densa con colonias bien visibles)
  3. CellSAM corre con ese threshold

Cambio principal vs version anterior: shrink=1.0 (incluye toda la placa detectada).

Uso:
    python scripts/contar_scout.py [images/placas/]

Guarda:
    results/colonias/experimentos/07_scout_adaptivo/summary.csv
    results/colonias/experimentos/07_scout_adaptivo/actinomicetos_X_count.png
"""
import sys
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

# ── parametros ────────────────────────────────────────────────────────────────
NORMALIZE       = True
POSTPROCESS     = True
MIN_COLONY_AREA = 300
MAX_COLONY_AREA = 50000
MIN_SOLIDITY    = 0.50
SHRINK          = 1.0        # incluye toda la placa detectada (vs 0.86 anterior)
SUPPORTED       = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
OUT_DIR         = Path('results/colonias/experimentos/07_scout_adaptivo')

# regla de seleccion de bbox_threshold segun densidad estimada
DENSE_THR    = 55   # n_adapt >= DENSE_THR  -> bbox 0.80
MODERATE_THR = 40   # n_adapt >= MOD_THR    -> bbox 0.75
# n_adapt < MODERATE_THR                    -> bbox 0.65

GT = {
    'actinomicetos_1': (63, 64), 'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60), 'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25), 'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67), 'actinomicetos_8': (24, 13),
}


# ── deteccion y recorte ───────────────────────────────────────────────────────
def detect_plates(img_bgr):
    h, w = img_bgr.shape[:2]
    portrait = h > w
    plates = []
    for i in range(2):
        if portrait:
            y0, y1 = i * h // 2, (i + 1) * h // 2
            half = img_bgr[y0:y1, :]; ox, oy = 0, y0
        else:
            x0, x1 = i * w // 2, (i + 1) * w // 2
            half = img_bgr[:, x0:x1]; ox, oy = x0, 0
        hh, hw = half.shape[:2]
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


def crop_plate(img_bgr, cx, cy, r, shrink=SHRINK):
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


# ── SCOUT: estimacion clasica de colonias con CLAHE + umbral adaptativo ───────
def scout(crop_bgr, plate_mask):
    """
    Estima el numero de colonias con vision clasica (~10 ms).
    CLAHE mejora el contraste local y el umbral adaptativo detecta estructura
    independientemente del nivel de brillo absoluto de cada placa.
    Devuelve (n_adapt, bbox_threshold elegido, razon).
    """
    gray  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.bitwise_and(gray, gray, mask=plate_mask)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh   = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(enh, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 71, -8)
    thresh = cv2.bitwise_and(thresh, thresh, mask=plate_mask)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel2, iterations=2)
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2, iterations=2)

    _, labels = cv2.connectedComponents(thresh)
    props = regionprops(labels)
    valid = [p for p in props
             if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY]
    n_adapt = len(valid)

    if n_adapt >= DENSE_THR:
        thr   = 0.80
        razon = f'densa (adapt={n_adapt})'
    elif n_adapt >= MODERATE_THR:
        thr   = 0.75
        razon = f'moderada (adapt={n_adapt})'
    else:
        thr   = 0.65
        razon = f'escasa/dificil (adapt={n_adapt})'

    return n_adapt, thr, razon


# ── CellSAM ───────────────────────────────────────────────────────────────────
def segment(model, crop_bgr, plate_mask, bbox_threshold):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    try:
        seg_mask, _, _ = segment_cellular_image(
            crop_rgb, model=model,
            normalize=NORMALIZE, postprocess=POSTPROCESS,
            bbox_threshold=bbox_threshold, device='cpu',
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
        color  = rng.uniform(0.2, 1.0, size=3)
        region = label_mask == p.label
        overlay[region] = overlay[region] * 0.3 + color * 0.7
    return overlay


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('images/placas')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in SUPPORTED
                    and not p.name.startswith('ground'))

    print(f'Imagenes: {len(images)}  shrink={SHRINK}\n')
    print('Cargando modelo CellSAM...')
    model = get_model()
    print('Listo.\n')

    header = (f'{"imagen":<22} {"Pl":>2} {"adapt":>5} '
              f'{"thr":>5} {"pred":>5} {"GT":>5} {"err":>5}  {"acc":>5}')
    print(header)
    print('-' * len(header))

    rows, all_errs, per_plate_acc = [], [], []

    for img_path in images:
        img    = cv2.imread(str(img_path))
        plates = detect_plates(img)
        gt_a, gt_b = GT.get(img_path.stem, (None, None))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{img_path.name}   |   GT: A={gt_a}  B={gt_b}',
                     fontsize=13, fontweight='bold')

        counts = []
        for i, (cx, cy, r) in enumerate(plates):
            gt_val = (gt_a, gt_b)[i]
            crop, plate_mask = crop_plate(img, cx, cy, r)

            n_adapt, thr, razon = scout(crop, plate_mask)
            seg_mask, valid     = segment(model, crop, plate_mask, thr)
            n = len(valid)
            counts.append(n)

            err_str = f'{n - gt_val:+d}' if gt_val is not None else '?'
            acc_str = ''
            if gt_val is not None:
                err_abs = abs(n - gt_val)
                all_errs.append(err_abs)
                acc = min(n, gt_val) / max(n, gt_val) if max(n, gt_val) > 0 else 1.0
                per_plate_acc.append(acc)
                acc_str = f'{acc * 100:.0f}%'
            print(f'{img_path.stem:<22} {"AB"[i]:>2} {n_adapt:>5} '
                  f'{thr:>5.2f} {n:>5} {str(gt_val):>5} {err_str:>5}  '
                  f'{acc_str:>5}   {razon}')

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            axes[i, 0].imshow(crop_rgb)
            axes[i, 0].set_title(
                f'Placa {"AB"[i]} original  (GT={gt_val})', fontsize=11)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(draw_overlay(crop, seg_mask, valid))
            axes[i, 1].set_title(
                f'Placa {"AB"[i]}  n={n}  err={err_str}  thr={thr:.2f}  adapt={n_adapt}',
                fontsize=11)
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{img_path.stem}_count.png',
                    dpi=120, bbox_inches='tight')
        plt.close()

        rows.append({'image': img_path.stem,
                     'plate_A': counts[0], 'plate_B': counts[1]})

    print('-' * len(header))
    mae      = float(np.mean(all_errs))      if all_errs      else float('nan')
    mean_acc = float(np.mean(per_plate_acc)) if per_plate_acc else float('nan')
    print(f'\nMAE             = {mae:.2f} colonias/placa')
    print(f'Accuracy media  = {mean_acc * 100:.1f}%')
    print(f'(linea base: MAE=6.0, accuracy=88.0%)')

    pd.DataFrame(rows).to_csv(OUT_DIR / 'summary.csv', index=False)
    print(f'\nCSV guardado: {OUT_DIR / "summary.csv"}')
    print(f'PNGs:         {OUT_DIR}/')


if __name__ == '__main__':
    main()
