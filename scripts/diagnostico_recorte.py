"""
Diagnostica calidad de deteccion de placas:
  - Superpone circulo detectado sobre imagen original
  - Compara shrink=0.86 vs shrink=1.0
  - Muestra cuantas colonias caen en zona excluida
Guarda en results/diagnostico/
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops

IMAGES_DIR = Path('images/placas')
OUT_DIR    = Path('results/diagnostico')
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT = {
    'actinomicetos_1': (63, 64), 'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60), 'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25), 'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67), 'actinomicetos_8': (24, 13),
}

SHRINKS = [0.86, 0.93, 1.00]


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
            detected = True
        else:
            cx, cy, r = hw // 2 + ox, hh // 2 + oy, int(min(hh, hw) * 0.43)
            detected = False
        plates.append((cx, cy, r, detected))
    return plates


def crop_plate(img_bgr, cx, cy, r, shrink=1.0):
    r_use = int(r * shrink)
    x1 = max(0, cx - r_use); y1 = max(0, cy - r_use)
    x2 = min(img_bgr.shape[1], cx + r_use)
    y2 = min(img_bgr.shape[0], cy + r_use)
    crop = img_bgr[y1:y2, x1:x2].copy()
    hc, wc = crop.shape[:2]
    mask = np.zeros((hc, wc), dtype=np.uint8)
    cv2.circle(mask, (cx - x1, cy - y1), r_use, 255, -1)
    return crop, mask


for img_path in sorted(IMAGES_DIR.glob('actinomicetos_*.jpeg')):
    stem = img_path.stem
    img  = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    plates = detect_plates(img)
    gt_vals = GT.get(stem, (None, None))

    # --- Figura 1: circulo detectado sobre imagen completa ---
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    for i, (cx, cy, r, det) in enumerate(plates):
        color = (0, 220, 0) if det else (255, 80, 80)
        cv2.circle(vis, (cx, cy), r,           color, 4)
        cv2.circle(vis, (cx, cy), int(r*0.86), (255, 165, 0), 2)  # shrink=0.86
        cv2.circle(vis, (cx, cy), 6,            (255, 0, 0), -1)
        label = f'{"AB"[i]} r={r}px{"" if det else " FALLBACK"}'
        cv2.putText(vis, label, (cx - 60, cy - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # linea divisoria
    if h > w:
        cv2.line(vis, (0, h // 2), (w, h // 2), (200, 200, 0), 2)
    else:
        cv2.line(vis, (w // 2, 0), (w // 2, h), (200, 200, 0), 2)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(vis)
    ax.set_title(f'{stem}  |  GT: A={gt_vals[0]}  B={gt_vals[1]}\n'
                 f'Verde = circulo detectado   Naranja = shrink 0.86   Rojo = centro', fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{stem}_circulo.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'{stem}: A r={plates[0][2]}px{"" if plates[0][3] else " FALLBACK"}  '
          f'B r={plates[1][2]}px{"" if plates[1][3] else " FALLBACK"}')

    # --- Figura 2: efecto del shrink en el recorte ---
    fig, axes = plt.subplots(2, len(SHRINKS) + 1, figsize=(5 * (len(SHRINKS) + 1), 11))
    fig.suptitle(f'{stem}  GT: A={gt_vals[0]}  B={gt_vals[1]}', fontsize=13, fontweight='bold')

    for row, (cx, cy, r, det) in enumerate(plates):
        placa_id = 'AB'[row]
        # columna 0: original sin mask con circulo
        orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
        cv2.circle(orig, (cx, cy), r, (0, 220, 0), 3)
        cv2.circle(orig, (cx, cy), int(r * 0.86), (255, 165, 0), 2)
        pad = int(r * 1.15)
        y1 = max(0, cy - pad); y2 = min(h, cy + pad)
        x1 = max(0, cx - pad); x2 = min(w, cx + pad)
        roi = orig[y1:y2, x1:x2]
        axes[row, 0].imshow(roi)
        axes[row, 0].set_title(f'Placa {placa_id} original\nVerde=full  Naranja=0.86', fontsize=9)
        axes[row, 0].axis('off')

        for col, shrink in enumerate(SHRINKS):
            crop, mask = crop_plate(img, cx, cy, r, shrink)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).copy()
            crop_rgb[mask == 0] = 0

            # detectar blobs simples para estimar colonias visibles
            gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray_c = cv2.bitwise_and(gray_c, gray_c, mask=mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
            tophat = cv2.morphologyEx(gray_c, cv2.MORPH_TOPHAT, kernel)
            _, bw   = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)
            bw      = cv2.bitwise_and(bw, bw, mask=mask)
            _, labels = cv2.connectedComponents(bw)
            props = regionprops(labels)
            n_blobs = sum(1 for p in props if 300 <= p.area <= 50000)

            axes[row, col + 1].imshow(crop_rgb)
            axes[row, col + 1].set_title(
                f'shrink={shrink}  crop={crop.shape[1]}x{crop.shape[0]}px\n'
                f'blobs tophat = {n_blobs}  GT={gt_vals[row]}', fontsize=9)
            axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{stem}_shrink.png', dpi=120, bbox_inches='tight')
    plt.close()

print(f'\nDiagnostico guardado en: {OUT_DIR}/')
