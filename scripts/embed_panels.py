"""
Genera paneles 3-columna por placa (Original | CLAHE | Segmentacion)
y los embebe como outputs en cellsam_actinomicetos_clean.ipynb.
La columna de segmentacion usa los PNGs ya generados en results/colonies/.
"""
import json
import base64
import io
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

IMAGES_DIR = Path(r'c:\Users\Sol\cellsam_project\cellsam\images\placas')
RESULTS_DIR = Path(r'c:\Users\Sol\cellsam_project\cellsam\results\colonies')
NB_PATH = Path(r'c:\Users\Sol\cellsam_project\cellsam\notebooks\cellsam_actinomicetos_clean.ipynb')

GT = {
    'actinomicetos_1': (63, 64), 'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60), 'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25), 'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67), 'actinomicetos_8': (24, 13),
}

CELLSAM_COUNTS = {
    'actinomicetos_1': (66, 68), 'actinomicetos_2': (66, 68),
    'actinomicetos_3': (64, 62), 'actinomicetos_4': (43, 66),
    'actinomicetos_5': (17, 35), 'actinomicetos_6': (49, 37),
    'actinomicetos_7': (54, 72), 'actinomicetos_8': (25, 12),
}
SEG_DIR = Path(r'c:\Users\Sol\cellsam_project\cellsam\results\colonias\experimentos\07_scout_adaptivo')


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


def crop_plate(img_bgr, cx, cy, r, shrink=1.0):
    r_use = int(r * shrink)
    x1 = max(0, cx - r_use)
    y1 = max(0, cy - r_use)
    x2 = min(img_bgr.shape[1], cx + r_use)
    y2 = min(img_bgr.shape[0], cy + r_use)
    crop = img_bgr[y1:y2, x1:x2].copy()
    hc, wc = crop.shape[:2]
    mask = np.zeros((hc, wc), dtype=np.uint8)
    cv2.circle(mask, (cx - x1, cy - y1), r_use, 255, -1)
    crop[mask == 0] = 0
    return crop, mask


def apply_clahe(crop_bgr, plate_mask):
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced[plate_mask == 0] = 0
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)


def extract_seg_from_postprocess(png_path, plate_idx):
    """
    04_postprocess_on: subplots(2,2), col derecha = segmentacion.
    Detecta el contenido oscuro (fondo negro de placa) para recortar
    exactamente cada placa sin incluir titulos ni separadores.
    """
    img = cv2.imread(str(png_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    right_gray = cv2.cvtColor(img[:, w // 2:], cv2.COLOR_BGR2GRAY)
    dark_rows  = np.where(right_gray.min(axis=1) < 30)[0]
    if len(dark_rows) == 0:
        y0 = 0 if plate_idx == 0 else h // 2
        y1 = h // 2 if plate_idx == 0 else h
        return cv2.cvtColor(img[y0:y1, w // 2:w], cv2.COLOR_BGR2RGB)
    # Encuentra el gap entre placa A y placa B (salto > 10 filas)
    gaps = np.where(np.diff(dark_rows) > 10)[0]
    if len(gaps) >= 1:
        split_y = dark_rows[gaps[-1]]   # fin del bloque A
        next_y  = dark_rows[gaps[-1]+1] # inicio del bloque B
    else:
        split_y = h // 2
        next_y  = h // 2
    if plate_idx == 0:
        y0 = int(dark_rows[0])
        y1 = int(split_y) + 1
    else:
        y0 = int(next_y)
        y1 = int(dark_rows[-1]) + 1
    return cv2.cvtColor(img[y0:y1, w // 2:w], cv2.COLOR_BGR2RGB)


def fig_to_b64(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def make_output(b64_str):
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {
            "image/png": b64_str,
            "text/plain": ["<Figure>"]
        }
    }


# --- Generar paneles y recopilar outputs por imagen ---
panel_outputs = {}  # stem -> list of outputs (one figure per imagen)

for img_path in sorted(IMAGES_DIR.glob('actinomicetos_*.jpeg')):
    stem = img_path.stem
    print(f'Generando paneles para {stem}...')

    img = cv2.imread(str(img_path))
    plates = detect_plates(img)
    counts = CELLSAM_COUNTS.get(stem, (None, None))
    gt_vals = GT.get(stem, (None, None))

    seg_png = SEG_DIR / f'{stem}_count.png'

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(stem.replace('actinomicetos_', 'Actinomicetos '),
                 fontsize=15, fontweight='bold', y=1.01)

    for col, title in enumerate(['Original', 'Contraste (CLAHE)', 'Segmentacion CellSAM']):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=10)

    for row, (cx, cy, r) in enumerate(plates):
        placa_id = 'AB'[row]
        gt_v = gt_vals[row] if gt_vals else None
        cs_v = counts[row] if counts else None

        crop, pmask = crop_plate(img, cx, cy, r, shrink=1.0)
        original_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        original_rgb[pmask == 0] = 0
        clahe_rgb = apply_clahe(crop, pmask)

        axes[row, 0].imshow(original_rgb)
        axes[row, 1].imshow(clahe_rgb)

        seg_img = extract_seg_from_postprocess(seg_png, row)
        if seg_img is not None:
            axes[row, 2].imshow(seg_img)
        else:
            axes[row, 2].text(0.5, 0.5, 'No disponible',
                              ha='center', va='center', transform=axes[row, 2].transAxes)

        axes[row, 0].set_ylabel(f'Placa {placa_id}', fontsize=12,
                                fontweight='bold', labelpad=10)

        gt_str = f'GT = {gt_v}' if gt_v is not None else ''
        cs_str = f'CellSAM = {cs_v}' if cs_v is not None else ''
        axes[row, 2].set_xlabel(f'{cs_str}    {gt_str}', fontsize=10, labelpad=6)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    b64 = fig_to_b64(fig, dpi=110)
    plt.close(fig)

    panel_outputs[stem] = [make_output(b64)]
    print(f'  OK ({len(b64) // 1024} KB)')

# --- Embeberlos en el notebook ---
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

for cell in nb['cells']:
    if cell['id'] == 'code-pipeline':
        cell['outputs'] = []
        cell['execution_count'] = 1
        for stem in sorted(panel_outputs.keys()):
            cell['outputs'].extend(panel_outputs[stem])
        print(f'\nEmbebidos {len(panel_outputs)} paneles en code-pipeline.')
        break

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Notebook guardado.')
