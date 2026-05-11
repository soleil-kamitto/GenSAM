"""
Experimento: efecto del preprocesamiento en CellSAM.

Corre tres metodos sobre todas las placas y compara contra ground truth:
  1. Classical CV  (top-hat + watershed, sin deep learning)
  2. CellSAM  normalize=True   (con preprocesamiento: percentil + CLAHE)
  3. CellSAM  normalize=False  (sin preprocesamiento)

postprocess=False y bbox_threshold=0.4 se mantienen fijos en ambas variantes.

Uso:
    python scripts/exp_normalize.py images/placas/

Resultados en results/colonias/:
    classical_cv/                      imagenes y CSV del metodo clasico
    cellsam_con_prepro/                imagenes y CSV con normalize=True
    cellsam_sin_prepro/                imagenes y CSV con normalize=False
    comparacion_preprocesamiento.png   grafico comparativo de los tres
    resumen_preprocesamiento.csv       tabla con conteos y errores
"""

import sys
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, label as sk_label
from cellSAM import get_model, segment_cellular_image


# ── PARAMETROS ────────────────────────────────────────────────────────────────
MIN_COLONY_AREA    = 300
MAX_COLONY_AREA    = 50000
MIN_SOLIDITY       = 0.50
TOPHAT_KERNEL      = 81
TOPHAT_THRESHOLD   = 10
SAT_THRESHOLD      = 60
MAX_HOLE_AREA      = 400
OPEN_ITERATIONS    = 1
WATERSHED_MIN_DIST = 20
BBOX_THRESHOLD     = 0.4
POSTPROCESS        = False

SUPPORTED = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

BASE_OUT = Path('results/colonias')

EXPERIMENTS = [
    {'name': 'cellsam_con_prepro', 'method': 'cellsam', 'normalize': True},
    {'name': 'cellsam_sin_prepro', 'method': 'cellsam', 'normalize': False},
    {'name': 'classical_cv',       'method': 'classical'},
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


# ── SEGMENTACION CLASICA ──────────────────────────────────────────────────────

def segment_classical(crop_bgr, plate_mask):
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray    = cv2.bitwise_and(gray, gray, mask=plate_mask)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    tophat  = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    _, bin_lum = cv2.threshold(tophat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    bin_lum    = cv2.bitwise_and(bin_lum, bin_lum, mask=plate_mask)
    hsv        = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat        = cv2.bitwise_and(hsv[:, :, 1], hsv[:, :, 1], mask=plate_mask)
    _, bin_col = cv2.threshold(sat, SAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    binary     = cv2.bitwise_or(bin_lum, bin_col)
    binary     = cv2.bitwise_and(binary, binary, mask=plate_mask)
    inv_labeled   = sk_label(~binary.astype(bool))
    binary_filled = binary.copy()
    for rid in range(1, inv_labeled.max() + 1):
        hole = inv_labeled == rid
        if hole.sum() <= MAX_HOLE_AREA:
            binary_filled[hole] = 255
    k3            = np.ones((3, 3), np.uint8)
    binary_filled = cv2.morphologyEx(binary_filled, cv2.MORPH_OPEN, k3, iterations=OPEN_ITERATIONS)
    binary_filled = cv2.bitwise_and(binary_filled, binary_filled, mask=plate_mask)
    dist   = ndimage.distance_transform_edt(binary_filled)
    coords = peak_local_max(dist, min_distance=WATERSHED_MIN_DIST,
                            threshold_rel=0.15, labels=binary_filled.astype(bool))
    if len(coords) == 0:
        _, labels_out = cv2.connectedComponents(binary_filled)
        props = regionprops(labels_out)
        valid = [p for p in props if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                 and p.solidity >= MIN_SOLIDITY]
        return labels_out, valid
    local_max              = np.zeros_like(dist, dtype=bool)
    local_max[tuple(coords.T)] = True
    markers   = sk_label(local_max)
    labels_ws = watershed(-dist, markers, mask=binary_filled.astype(bool))
    props = regionprops(labels_ws)
    valid = [p for p in props if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY]
    return labels_ws, valid


# ── SEGMENTACION CELLSAM ──────────────────────────────────────────────────────

def segment_cellsam(crop_bgr, plate_mask, normalize):
    model    = get_cached_model()
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mask, _, _ = segment_cellular_image(
        crop_rgb, model=model,
        normalize=normalize, postprocess=POSTPROCESS,
        bbox_threshold=BBOX_THRESHOLD, device='cpu',
    )
    mask = mask.copy()
    mask[plate_mask == 0] = 0
    props = regionprops(mask)
    valid = [p for p in props if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY]
    return mask, valid


# ── OVERLAY ───────────────────────────────────────────────────────────────────

def draw_overlay(crop_bgr, valid_props, label_mask):
    overlay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for prop in valid_props:
        region = label_mask == prop.label
        overlay[region] = overlay[region] * 0.45 + np.array([0.15, 0.85, 0.35]) * 0.55
    return overlay


# ── PROCESAR UNA IMAGEN CON UN METODO ────────────────────────────────────────

def process_image(img_path, exp, output_dir):
    img    = cv2.imread(str(img_path))
    plates = detect_plates(img)
    counts = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    label = exp['name'].replace('_', ' ')
    fig.suptitle(f"{img_path.name}  [{label}]", fontsize=12, fontweight='bold')

    for i, (cx, cy, r) in enumerate(plates):
        name       = ['A', 'B'][i]
        crop, mask = crop_plate(img, cx, cy, r)

        if exp['method'] == 'classical':
            lbl, valid = segment_classical(crop, mask)
        else:
            lbl, valid = segment_cellsam(crop, mask, normalize=exp['normalize'])

        n = len(valid)
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
    exp_names  = list(all_results.keys())
    img_names  = [r['name'] for r in next(iter(all_results.values()))]
    short_names = [n.replace('actinomicetos_', 'actin_') for n in img_names]

    colors = {
        'classical_cv':       '#4C72B0',
        'cellsam_con_prepro': '#DD8452',
        'cellsam_sin_prepro': '#E377C2',
    }
    labels = {
        'classical_cv':       'Classical CV',
        'cellsam_con_prepro': 'CellSAM con prepro (normalize=True)',
        'cellsam_sin_prepro': 'CellSAM sin prepro (normalize=False)',
    }

    x     = np.arange(len(img_names))
    n_exp = len(exp_names)
    width = 0.18
    offsets = np.linspace(-(n_exp) * width / 2, (n_exp) * width / 2, n_exp + 1)[:-1] + width / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    fig.suptitle('Efecto del preprocesamiento en CellSAM', fontsize=14, fontweight='bold')

    for ax, plate_idx, plate_label in [(ax1, 0, 'Placa A'), (ax2, 1, 'Placa B')]:
        gt_vals = [gt.get(r['name'], (0, 0))[plate_idx]
                   for r in next(iter(all_results.values()))]
        b_gt = ax.bar(x - (n_exp) * width / 2 - width / 2, gt_vals, width,
                      label='Ground truth', color='#2ca02c', alpha=0.85)
        ax.bar_label(b_gt, padding=2, fontsize=7)

        for j, exp_name in enumerate(exp_names):
            vals = [r['counts'][plate_idx] for r in all_results[exp_name]]
            bars = ax.bar(x + offsets[j], vals, width,
                          label=labels[exp_name], color=colors[exp_name], alpha=0.85)
            ax.bar_label(bars, padding=2, fontsize=7)

        all_vals = gt_vals + [r['counts'][plate_idx]
                              for exp in all_results.values() for r in exp]
        ax.set_ylabel('Colonias contadas')
        ax.set_title(plate_label, fontsize=12)
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(all_vals, default=1) * 1.28)
        ax.grid(axis='y', alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    out = output_dir / 'comparacion_preprocesamiento.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Grafico guardado: {out}')


# ── TABLA CSV ─────────────────────────────────────────────────────────────────

def make_csv(all_results, gt, output_dir):
    rows = []
    img_names = [r['name'] for r in next(iter(all_results.values()))]
    for img_name in img_names:
        ga, gb = gt.get(img_name, (None, None))
        row = {'imagen': img_name, 'gt_A': ga, 'gt_B': gb}
        for exp_name, records in all_results.items():
            rec = next(r for r in records if r['name'] == img_name)
            ca, cb = rec['counts']
            row[f'{exp_name}_A'] = ca
            row[f'{exp_name}_B'] = cb
            if ga is not None:
                row[f'err_{exp_name}_A'] = ca - ga
                row[f'err_{exp_name}_B'] = cb - gb
        rows.append(row)

    out = output_dir / 'resumen_preprocesamiento.csv'
    df  = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f'CSV guardado: {out}')
    return df


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
    if not images:
        print(f'No hay imagenes en {images_dir}')
        sys.exit(1)

    BASE_OUT.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for exp in EXPERIMENTS:
        exp_dir = BASE_OUT / exp['name']
        exp_dir.mkdir(parents=True, exist_ok=True)
        records = []
        print(f"\n{'='*55}")
        print(f"  Metodo: {exp['name']}")
        print(f"{'='*55}")

        for idx, img_path in enumerate(images, 1):
            print(f'  [{idx}/{len(images)}] {img_path.name}', end='  ')
            counts = process_image(img_path, exp, exp_dir)
            ga, gb = gt.get(img_path.stem, (None, None))
            if ga is not None:
                err_a = counts[0] - ga
                err_b = counts[1] - gb
                print(f'A={counts[0]}(err={err_a:+d})  B={counts[1]}(err={err_b:+d})')
            else:
                print(f'A={counts[0]}  B={counts[1]}')
            records.append({'name': img_path.stem, 'counts': counts})

            # CSV por experimento
            with open(exp_dir / 'summary.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['image', 'plate_A', 'plate_B'])
                for rec in records:
                    w.writerow([rec['name'], rec['counts'][0], rec['counts'][1]])

        all_results[exp['name']] = records

    # Grafico y tabla comparativa
    print('\nGenerando grafico comparativo...')
    make_chart(all_results, gt, BASE_OUT)
    df = make_csv(all_results, gt, BASE_OUT)

    # Resumen de error por metodo
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
        if errs:
            print(f"  {exp['name']:<28} {np.mean(errs):.1f} colonias/placa")
    print('='*55)


if __name__ == '__main__':
    main()
