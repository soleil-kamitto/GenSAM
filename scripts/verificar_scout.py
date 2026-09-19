"""
Verificacion rapida del nuevo scout (sin CellSAM).
Muestra n_adapt y thr elegido por placa, comparando con umbral optimo del sweep.
"""
import numpy as np
import cv2
from pathlib import Path
from skimage.measure import regionprops

IMAGES_DIR      = Path('images/placas')
SHRINK          = 1.0
MIN_COLONY_AREA = 300
MAX_COLONY_AREA = 50000
MIN_SOLIDITY    = 0.50
DENSE_THR       = 55
MODERATE_THR    = 40

GT = {
    'actinomicetos_1': (63, 64), 'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60), 'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25), 'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67), 'actinomicetos_8': (24, 13),
}
# threshold optimo por placa (del sweep 05_bbox_sweep, shrink=0.86, postprocess=False)
OPTIMAL = {
    'actinomicetos_1': (0.80, 0.80), 'actinomicetos_2': (0.80, 0.80),
    'actinomicetos_3': (0.80, 0.80), 'actinomicetos_4': (0.80, 0.80),
    'actinomicetos_5': (0.10, 0.80), 'actinomicetos_6': (0.80, 0.10),
    'actinomicetos_7': (0.10, 0.80), 'actinomicetos_8': (0.80, 0.80),
}


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
    r_use = int(r * SHRINK)
    x1 = max(0, cx-r_use); y1 = max(0, cy-r_use)
    x2 = min(img_bgr.shape[1], cx+r_use); y2 = min(img_bgr.shape[0], cy+r_use)
    crop = img_bgr[y1:y2, x1:x2].copy()
    hc, wc = crop.shape[:2]
    mask = np.zeros((hc, wc), dtype=np.uint8)
    cv2.circle(mask, (cx-x1, cy-y1), r_use, 255, -1)
    crop[mask==0] = 0
    return crop, mask


def scout(crop_bgr, plate_mask):
    gray  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.bitwise_and(gray, gray, mask=plate_mask)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enh   = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 71, -8)
    thresh = cv2.bitwise_and(thresh, thresh, mask=plate_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)
    _, labels = cv2.connectedComponents(thresh)
    props = regionprops(labels)
    valid = [p for p in props if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY]
    n = len(valid)
    if n >= DENSE_THR:    thr = 0.80
    elif n >= MODERATE_THR: thr = 0.75
    else:                   thr = 0.65
    return n, thr


print(f'shrink={SHRINK}  dense>={DENSE_THR}->0.80  mod>={MODERATE_THR}->0.75  else->0.65\n')
print(f'{"Imagen":<22} {"Pl":>2}  {"GT":>4}  {"adapt":>5}  {"thr_nuevo":>9}  {"thr_opt":>7}')
print('-' * 58)

for img_path in sorted(IMAGES_DIR.glob('actinomicetos_*.jpeg')):
    stem = img_path.stem
    img  = cv2.imread(str(img_path))
    plates = detect_plates(img)
    gt_ab  = GT.get(stem, (None,None))
    opt_ab = OPTIMAL.get(stem, (None,None))
    for i, (cx,cy,r) in enumerate(plates):
        crop, mask = crop_plate(img, cx, cy, r)
        n_adapt, thr = scout(crop, mask)
        print(f'{stem:<22} {"AB"[i]:>2}  {gt_ab[i]:>4}  {n_adapt:>5}  '
              f'{thr:>9.2f}  {opt_ab[i]:>7.2f}')
