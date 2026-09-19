"""
Corre CellSAM (normalize=True, postprocess=True) sobre cada placa,
genera paneles 3-columna con segmentacion por instancia (colores aleatorios),
y embebe los outputs en cellsam_actinomicetos_clean.ipynb.
"""
import json, base64, io, csv
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops
from cellSAM import get_model, segment_cellular_image

IMAGES_DIR = Path(r'c:\Users\Sol\cellsam_project\cellsam\images\placas')
GT_CSV     = IMAGES_DIR / 'ground_truth.csv'
NB_PATH    = Path(r'c:\Users\Sol\cellsam_project\cellsam\notebooks\cellsam_actinomicetos_clean.ipynb')

MIN_COLONY_AREA = 300
MAX_COLONY_AREA = 50000
MIN_SOLIDITY    = 0.50

GT = {}
with open(GT_CSV, newline='', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        stem = Path(row['image']).stem
        GT[stem] = (int(row['plate_A']), int(row['plate_B']))

# ---- helpers ----

def detect_plates(img_bgr):
    h, w = img_bgr.shape[:2]
    portrait = h > w
    plates = []
    for i in range(2):
        if portrait:
            y0, y1 = i*h//2, (i+1)*h//2
            half = img_bgr[y0:y1, :]
            ox, oy = 0, y0
        else:
            x0, x1 = i*w//2, (i+1)*w//2
            half = img_bgr[:, x0:x1]
            ox, oy = x0, 0
        hh, hw = half.shape[:2]
        gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(hh, hw), param1=60, param2=25,
            minRadius=int(min(hh, hw)*0.30), maxRadius=int(min(hh, hw)*0.52))
        if circles is not None:
            best = np.round(circles[0][0]).astype(int)
            cx, cy, r = int(best[0])+ox, int(best[1])+oy, int(best[2])
        else:
            cx, cy, r = hw//2+ox, hh//2+oy, int(min(hh,hw)*0.43)
        plates.append((cx, cy, r))
    return plates


def crop_plate(img_bgr, cx, cy, r):
    r_use = r  # shrink=1.0
    x1 = max(0, cx-r_use); y1 = max(0, cy-r_use)
    x2 = min(img_bgr.shape[1], cx+r_use)
    y2 = min(img_bgr.shape[0], cy+r_use)
    crop = img_bgr[y1:y2, x1:x2].copy()
    hc, wc = crop.shape[:2]
    mask = np.zeros((hc, wc), dtype=np.uint8)
    cv2.circle(mask, (cx-x1, cy-y1), r_use, 255, -1)
    crop[mask == 0] = 0
    return crop, mask


def apply_clahe(crop_bgr, plate_mask):
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced[plate_mask == 0] = 0
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)


def draw_instance_overlay(crop_bgr, valid_props, label_mask, plate_mask):
    """Cada colonia con un color aleatorio distinto."""
    rng = np.random.default_rng(42)
    overlay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay[plate_mask == 0] = 0
    colors = rng.uniform(0.3, 1.0, size=(len(valid_props), 3))
    for prop, color in zip(valid_props, colors):
        region = label_mask == prop.label
        overlay[region] = overlay[region] * 0.35 + color * 0.65
    return np.clip(overlay, 0, 1)


def fig_to_b64(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def png_output(b64):
    return {"output_type": "display_data", "metadata": {},
            "data": {"image/png": b64, "text/plain": ["<Figure>"]}}


# ---- cargar modelo ----
print('Cargando modelo CellSAM...')
model = get_model()
print('Modelo listo.\n')

# ---- procesar imagenes ----
panel_outputs = {}
all_results   = []

for img_path in sorted(IMAGES_DIR.glob('actinomicetos_*.jpeg')):
    stem = img_path.stem
    print(f'Procesando {stem}...')
    img    = cv2.imread(str(img_path))
    plates = detect_plates(img)
    gt_vals = GT.get(stem, (None, None))

    panels = []
    counts = []

    for idx, (cx, cy, r) in enumerate(plates):
        placa_id = 'AB'[idx]
        gt_val   = gt_vals[idx] if gt_vals else None

        crop, pmask = crop_plate(img, cx, cy, r)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = segment_cellular_image(
            crop_rgb, model=model,
            normalize=True, postprocess=True, device='cpu',
        )
        lmask = result[0].copy() if result[0] is not None else np.zeros(crop_rgb.shape[:2], dtype=np.int32)
        lmask[pmask == 0] = 0

        props = regionprops(lmask)
        valid = [p for p in props
                 if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                 and p.solidity >= MIN_SOLIDITY]
        n_col = len(valid)
        counts.append(n_col)

        original_rgb = crop_rgb.copy()
        original_rgb[pmask == 0] = 0
        clahe_rgb    = apply_clahe(crop, pmask)
        overlay_rgb  = draw_instance_overlay(crop, valid, lmask, pmask)

        panels.append((original_rgb, clahe_rgb, overlay_rgb, placa_id, n_col, gt_val))
        print(f'  Placa {placa_id}: {n_col} colonias  (GT={gt_val})')

    # figura 2x3
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(stem.replace('actinomicetos_', 'Actinomicetos '),
                 fontsize=15, fontweight='bold', y=1.01)

    for col, title in enumerate(['Original', 'Contraste (CLAHE)', 'Segmentacion CellSAM']):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=10)

    for row, (orig, clahe, overlay, placa_id, n_col, gt_val) in enumerate(panels):
        axes[row, 0].imshow(orig)
        axes[row, 1].imshow(clahe)
        axes[row, 2].imshow(overlay)
        axes[row, 0].set_ylabel(f'Placa {placa_id}', fontsize=12, fontweight='bold', labelpad=10)
        gt_str = f'GT = {gt_val}' if gt_val is not None else ''
        axes[row, 2].set_xlabel(f'CellSAM = {n_col}    {gt_str}', fontsize=11, labelpad=6)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    panel_outputs[stem] = [png_output(b64)]

    all_results.append({
        'image': stem,
        'cellsam_A': counts[0], 'cellsam_B': counts[1],
        'gt_A': gt_vals[0],     'gt_B': gt_vals[1],
    })
    print(f'  figura OK ({len(b64)//1024} KB)')

# ---- tabla y graficos ----
import pandas as pd

rows = []
for r in all_results:
    for placa, idx in [('A', 0), ('B', 1)]:
        gt_v = r[f'gt_{placa}']; cs_v = r[f'cellsam_{placa}']
        err  = cs_v - gt_v if gt_v is not None else None
        rows.append({'Imagen': r['image'].replace('actinomicetos_',''),
                     'Placa': placa, 'Ground Truth': gt_v,
                     'CellSAM': cs_v, 'Error': err,
                     'Err abs': abs(err) if err is not None else None})
df = pd.DataFrame(rows)
mae  = df['Err abs'].dropna().mean()
bias = df['Error'].dropna().mean()
table_txt = df.to_string(index=False) + \
    f'\n\nError absoluto medio (MAE) : {mae:.1f} colonias/placa\nSesgo medio (Bias)         : {bias:+.1f} colonias/placa\n'

x = np.arange(len(all_results)); width = 0.30
names = [r['image'].replace('actinomicetos_','') for r in all_results]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
fig.suptitle('CellSAM vs Ground Truth  (normalize=True, postprocess=True)', fontsize=14, fontweight='bold')
for ax, placa in [(ax1,'A'),(ax2,'B')]:
    gt_v = [r[f'gt_{placa}'] for r in all_results]
    cs_v = [r[f'cellsam_{placa}'] for r in all_results]
    b0 = ax.bar(x-width/2, gt_v, width, label='Ground truth', color='#2ca02c', alpha=0.85)
    b1 = ax.bar(x+width/2, cs_v, width, label='CellSAM',      color='#DD8452', alpha=0.85)
    ax.bar_label(b0, padding=3, fontsize=9); ax.bar_label(b1, padding=3, fontsize=9)
    ax.set_ylabel('Colonias'); ax.set_title(f'Placa {placa}', fontsize=11)
    ax.legend(fontsize=9); ax.set_ylim(0, max(gt_v+cs_v)*1.25); ax.grid(axis='y', alpha=0.3)
ax2.set_xticks(x); ax2.set_xticklabels([f'Img {n}' for n in names], fontsize=9)
plt.tight_layout()
b64_bars = fig_to_b64(fig); plt.close(fig)

labels, errors, colors_b = [], [], []
for r in all_results:
    for placa, idx in [('A',0),('B',1)]:
        err = r[f'cellsam_{placa}'] - r[f'gt_{placa}']
        labels.append(r['image'].replace('actinomicetos_','')+placa)
        errors.append(err); colors_b.append('#DD8452' if err>0 else '#4C72B0')
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(labels, errors, color=colors_b, alpha=0.85)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Error (CellSAM - GT)'); ax.set_xlabel('Imagen + Placa')
ax.set_title('Error por placa  (naranja = sobreconteo | azul = subconteo)')
ax.grid(axis='y', alpha=0.3); plt.tight_layout()
b64_err = fig_to_b64(fig); plt.close(fig)

# ---- embeberlo todo en el notebook ----
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

for cell in nb['cells']:
    cid = cell['id']
    if cid == 'code-pipeline':
        cell['outputs'] = []
        cell['execution_count'] = 1
        for stem in sorted(panel_outputs):
            cell['outputs'].extend(panel_outputs[stem])
    elif cid == 'code-table':
        cell['outputs'] = [{"output_type":"stream","name":"stdout","text":table_txt}]
        cell['execution_count'] = 1
    elif cid == 'code-chart':
        cell['outputs'] = [png_output(b64_bars)]
        cell['execution_count'] = 1
    elif cid == 'code-errplot':
        mae_txt = f'MAE  : {mae:.1f} colonias/placa\nSesgo: {bias:+.1f} colonias/placa\n'
        cell['outputs'] = [png_output(b64_err),
                           {"output_type":"stream","name":"stdout","text":mae_txt}]
        cell['execution_count'] = 1

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'\nNotebook guardado.  MAE={mae:.1f}  Bias={bias:+.1f}')
