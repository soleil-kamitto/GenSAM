"""
Experimento de preprocesado para las fotos propias (images/mis_fotos/).

Motivacion: la pipeline funciona bien en placas de buen contraste (MC73-A: 13/13,
RC73-A-2.5: 204) pero falla en las lavadas por el contraluz, donde las colonias
son beige palido sobre agar palido y hay un gradiente fuerte de iluminacion
(M2-A-3 devolvio 0 con decenas de colonias visibles, MC73-C devolvio 1 de ~5).
El normalize de CellSAM es global y no corrige ese gradiente de baja frecuencia.

Compara 4 variantes sobre 3 imagenes representativas:
  A  baseline                     thr=0.65   (la configuracion actual)
  B  baseline                     thr=0.40
  C  flat-field + CLAHE           thr=0.65
  D  flat-field + CLAHE           thr=0.40

Las imagenes incluyen dos casos de fallo y uno que ya funciona bien, para
detectar si la correccion arregla los primeros sin romper el ultimo.

Uso:
    python scripts/exp_prepro_mis_fotos.py

Guarda:
    results/colonias/experimentos/08_prepro_mis_fotos/resultados.csv
    results/colonias/experimentos/08_prepro_mis_fotos/<imagen>_<variante>.png
"""
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

MIN_COLONY_AREA = 300
MAX_COLONY_AREA = 50000
MIN_SOLIDITY    = 0.50
SHRINK          = 0.86
MAX_CROP_DIM    = 1200
INK_HUE_MIN, INK_HUE_MAX, INK_SAT_MIN = 60, 140, 40

OUT_DIR = Path('results/colonias/experimentos/08_prepro_mis_fotos')

# imagen -> conteo visual aproximado, para orientar la lectura de resultados
IMGS = {
    'M2-A-3': 'falla total (n=0), muchas colonias palidas en cadena',
    'MC73-C': 'falla parcial (n=1), ~5 colonias visibles',
    'MC73-A': 'ya correcto (n=13), control de no regresion',
}

VARIANTS = [
    ('A_base_065',  False, 0.65),
    ('B_base_040',  False, 0.40),
    ('C_flat_065',  True,  0.65),
    ('D_flat_040',  True,  0.40),
]


def detect_plate(img_bgr):
    h, w = img_bgr.shape[:2]
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(h, w), param1=60, param2=25,
        minRadius=int(min(h, w) * 0.30), maxRadius=int(min(h, w) * 0.52),
    )
    if circles is not None:
        b = np.round(circles[0][0]).astype(int)
        return int(b[0]), int(b[1]), int(b[2])
    return w // 2, h // 2, int(min(h, w) * 0.43)


def crop_plate(img_bgr, cx, cy, r):
    r_use = int(r * SHRINK)
    x1 = max(0, cx - r_use);  y1 = max(0, cy - r_use)
    x2 = min(img_bgr.shape[1], cx + r_use); y2 = min(img_bgr.shape[0], cy + r_use)
    crop = img_bgr[y1:y2, x1:x2].copy()
    hc, wc = crop.shape[:2]
    mask = np.zeros((hc, wc), dtype=np.uint8)
    cv2.circle(mask, (cx - x1, cy - y1), r_use, 255, -1)
    crop[mask == 0] = 0
    longest = max(hc, wc)
    if longest > MAX_CROP_DIM:
        s = MAX_CROP_DIM / longest
        nw, nh = int(round(wc * s)), int(round(hc * s))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return crop, mask


def flat_field(crop_bgr, plate_mask):
    """
    Corrige el gradiente de iluminacion del contraluz: estima el fondo con un
    desenfoque gaussiano de sigma grande y lo divide, dejando solo la estructura
    de alta frecuencia (las colonias). Despues aplica CLAHE para realzar el
    contraste local, que es justo lo que falta en las placas lavadas.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = int(max(crop_bgr.shape[:2]) * 0.15) | 1     # kernel impar, ~15% del lado
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    corrected = gray / (bg + 1e-6)

    inside = plate_mask > 0
    if inside.sum() == 0:
        return crop_bgr
    lo, hi = np.percentile(corrected[inside], [1, 99])
    corrected = np.clip((corrected - lo) / (hi - lo + 1e-6), 0, 1)
    corrected = (corrected * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    corrected = clahe.apply(corrected)
    corrected[~inside] = 0
    return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)


def is_ink(hsv, label_mask, label):
    region = label_mask == label
    h = float(np.median(hsv[..., 0][region]))
    s = float(np.median(hsv[..., 1][region]))
    return INK_HUE_MIN <= h <= INK_HUE_MAX and s >= INK_SAT_MIN


def segment(model, crop_for_model, crop_original, plate_mask, thr):
    rgb = cv2.cvtColor(crop_for_model, cv2.COLOR_BGR2RGB)
    try:
        seg, _, _ = segment_cellular_image(
            rgb, model=model, normalize=True, postprocess=True,
            bbox_threshold=thr, device='cpu')
        if seg is None:
            seg = np.zeros(rgb.shape[:2], dtype=np.int32)
    except (AttributeError, TypeError, ValueError):
        seg = np.zeros(crop_for_model.shape[:2], dtype=np.int32)
    seg = seg.copy()
    seg[plate_mask == 0] = 0
    # el filtro de tinta se evalua siempre sobre la imagen original en color,
    # porque la variante flat-field devuelve una imagen en escala de grises
    hsv = cv2.cvtColor(crop_original, cv2.COLOR_BGR2HSV)
    valid = [p for p in regionprops(seg)
             if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY
             and not is_ink(hsv, seg, p.label)]
    return seg, valid


def draw_overlay(crop_bgr, label_mask, valid_props):
    rng = np.random.default_rng(seed=42)
    ov = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for p in valid_props:
        color = rng.uniform(0.2, 1.0, size=3)
        reg = label_mask == p.label
        ov[reg] = ov[reg] * 0.3 + color * 0.7
    return ov


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Cargando modelo CellSAM...')
    model = get_model()
    print('Listo.\n')

    header = f'{"imagen":<10} {"variante":<12} {"conteo":>6}'
    print(header)
    print('-' * len(header))

    rows = []
    for stem, nota in IMGS.items():
        img = cv2.imread(f'images/mis_fotos/{stem}.jpg')
        cx, cy, r = detect_plate(img)
        crop, plate_mask = crop_plate(img, cx, cy, r)
        flat = flat_field(crop, plate_mask)

        for vname, use_flat, thr in VARIANTS:
            crop_model = flat if use_flat else crop
            seg, valid = segment(model, crop_model, crop, plate_mask, thr)
            n = len(valid)
            print(f'{stem:<10} {vname:<12} {n:>6}')
            rows.append({'image': stem, 'variante': vname,
                         'flat_field': use_flat, 'bbox_threshold': thr,
                         'conteo': n, 'nota': nota})

            fig, axes = plt.subplots(1, 3, figsize=(17, 6))
            fig.suptitle(f'{stem}  |  {vname}  |  n={n}',
                         fontsize=13, fontweight='bold')
            axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[0].set_title('original', fontsize=11); axes[0].axis('off')
            axes[1].imshow(cv2.cvtColor(crop_model, cv2.COLOR_BGR2RGB))
            axes[1].set_title('entrada al modelo', fontsize=11); axes[1].axis('off')
            axes[2].imshow(draw_overlay(crop, seg, valid))
            axes[2].set_title(f'deteccion n={n}', fontsize=11); axes[2].axis('off')
            plt.tight_layout()
            plt.savefig(OUT_DIR / f'{stem}_{vname}.png', dpi=110,
                        bbox_inches='tight')
            plt.close()

    print('-' * len(header))
    pd.DataFrame(rows).to_csv(OUT_DIR / 'resultados.csv', index=False)
    print(f'\nCSV: {OUT_DIR / "resultados.csv"}')
    print(f'PNGs: {OUT_DIR}/')


if __name__ == '__main__':
    main()
