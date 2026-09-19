"""
Conteo de colonias para las fotos propias del usuario (images/mis_fotos/),
una placa por imagen (a diferencia del dataset actinomicetos_N, que trae dos
placas por foto).

Pipeline (lo mejor encontrado en los experimentos anteriores):
  1. Deteccion de la placa por Hough circle sobre la imagen completa.
  2. Recorte circular de la placa (shrink=1.0, incluye toda la placa).
  3. Scout: CLAHE + umbral adaptativo -> estimacion clasica rapida de densidad.
  4. Segun esa estimacion se elige bbox_threshold para CellSAM:
       n_adapt < 40   -> 0.65  (placa escasa o dificil de ver)
       40 <= n_adapt < 55 -> 0.75  (densidad moderada)
       n_adapt >= 55  -> 0.80  (placa densa, colonias bien visibles)
  5. CellSAM corre con normalize=True, postprocess=True y ese bbox_threshold.
  6. Filtro de regiones por area (300-50000 px) y solidity (>=0.50).

Uso:
    python scripts/contar_mis_fotos.py [images/mis_fotos/]

Guarda:
    results/colonias/mis_fotos/summary.csv
    results/colonias/mis_fotos/<nombre>_count.png
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
SHRINK          = 0.92   # radio de recorte, elegido con la curva de recuperacion
                          # medida sobre el ground truth manual (856 colonias):
                          #   0.86 -> 79.0 % de las colonias dentro del recorte
                          #   0.90 -> 88.9 %
                          #   0.92 -> 96.0 %   <- valor actual
                          #   1.00 -> 100 %
                          #
                          # medido contra el conteo manual (scripts/exp_shrink.py):
                          #            MAE contable   MAE global   acierto
                          #   0.86         5.46         15.40       73.0 %
                          #   0.92         4.08         11.33       80.1 %
                          # el 0.92 gana sobre todo en las placas densas, donde
                          # RC73-A-2.5 pasa de -109 a -77 de error. El costo es que
                          # MC73-A y MC73-D dejan de ser exactas (13->16 y 9->12)
                          # por falsos positivos de condensacion en el borde.
                          #
                          # el radio completo (1.0) se probo y es peor: MAE contable
                          # 5.77, porque la condensacion pegada al vidrio forma
                          # gotas redondas que ningun filtro de forma distingue de
                          # una colonia. 0.92 se queda por dentro de esa banda.
RIM_BAND         = 0.86  # sin uso mientras SHRINK=0.86 (el recorte ya excluye todo
                          # lo que quedaria en esta banda); queda listo si se retoma
                          # el experimento con un descriptor mejor que la forma
RIM_MIN_SOLIDITY = 0.65
RIM_MAX_ECC      = 0.85
MAX_CROP_DIM    = 1200   # las fotos propias vienen a resolucion de camara (hasta
                          # 4096px); se reescala el recorte de la placa a esta
                          # escala, la misma con la que se calibro la pipeline
                          # (dataset actinomicetos_N, placas de ~800-1000px)
SUPPORTED       = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
# la carpeta de salida se calcula en main() a partir de la carpeta de entrada

DENSE_THR    = 55
MODERATE_THR = 40

# rechazo de rotulacion con marcador azul: medido sobre las fotos, la tinta cae
# en hue 71-105 con saturacion alta, mientras que las colonias estan en hue 39-44
INK_HUE_MIN  = 60
INK_HUE_MAX  = 140
INK_SAT_MIN  = 40


def flat_field(crop_bgr, plate_mask):
    """
    Corrige el gradiente de iluminacion del contraluz: estima el fondo con un
    desenfoque gaussiano de sigma grande y lo divide, dejando la estructura de
    alta frecuencia (las colonias). Luego CLAHE para realzar contraste local.

    Medido en scripts/exp_prepro_mis_fotos.py: sin esta correccion M2-A-3 daba 0
    colonias (con decenas visibles) y MC73-C daba 1 de ~5. Bajar el umbral por si
    solo no arreglaba nada. Con la correccion pasan a 29 y 4. El control MC73-A
    da 13 con y sin correccion, o sea que no degrada lo que ya funcionaba.
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
    """True si la region parece rotulacion con marcador y no una colonia."""
    region = label_mask == label
    h = float(np.median(hsv[..., 0][region]))
    s = float(np.median(hsv[..., 1][region]))
    return INK_HUE_MIN <= h <= INK_HUE_MAX and s >= INK_SAT_MIN


# ── deteccion y recorte (una sola placa por imagen) ────────────────────────────
def detect_plate(img_bgr):
    h, w = img_bgr.shape[:2]
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(h, w), param1=60, param2=25,
        minRadius=int(min(h, w) * 0.30),
        maxRadius=int(min(h, w) * 0.52),
    )
    if circles is not None:
        best = np.round(circles[0][0]).astype(int)
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])
    else:
        cx, cy, r = w // 2, h // 2, int(min(h, w) * 0.43)
    return cx, cy, r


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

    # reescala si el recorte excede MAX_CROP_DIM (fotos de camara vienen mucho
    # mas grandes que las del dataset con el que se calibraron los umbrales)
    longest = max(hc, wc)
    if longest > MAX_CROP_DIM:
        scale = MAX_CROP_DIM / longest
        new_w, new_h = int(round(wc * scale)), int(round(hc * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return crop, mask


# ── scout: estimacion clasica con CLAHE + umbral adaptativo ───────────────────
def scout(crop_bgr, plate_mask):
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

    # El scout corre sobre la imagen ya corregida por flat-field, porque sobre la
    # original el estimador daba valores de ruido (0-20 en las 15 fotos) y la
    # regla adaptativa quedaba inerte: elegia el mismo umbral siempre.
    if n_adapt >= DENSE_THR:
        # las placas densas ya salian bien con 0.65 (RC73-A-2.5 verificada), no
        # se toca para no introducir falsos positivos entre colonias pegadas
        thr, razon = 0.65, f'densa (adapt={n_adapt})'
    else:
        # 0.40 es lo que rescata las placas palidas sin degradar el control
        thr, razon = 0.40, f'escasa/dificil (adapt={n_adapt})'

    return n_adapt, thr, razon


# ── CellSAM ─────────────────────────────────────────────────────────────────
def segment(model, crop_model, crop_original, plate_mask, bbox_threshold):
    """crop_model es la imagen corregida que entra al modelo; crop_original se
    usa solo para el filtro de tinta, que necesita el color real."""
    crop_bgr = crop_model
    crop_rgb = cv2.cvtColor(crop_model, cv2.COLOR_BGR2RGB)
    try:
        seg_mask, _, _ = segment_cellular_image(
            crop_rgb, model=model,
            normalize=NORMALIZE, postprocess=POSTPROCESS,
            bbox_threshold=bbox_threshold, device='cpu',
        )
        if seg_mask is None:
            seg_mask = np.zeros(crop_rgb.shape[:2], dtype=np.int32)
    except (AttributeError, TypeError, ValueError):
        # ValueError: postprocess_predictions de CellSAM falla con np.max sobre
        # una lista vacia cuando la placa esta esteril y no detecta nada
        seg_mask = np.zeros(crop_bgr.shape[:2], dtype=np.int32)
    seg_mask = seg_mask.copy()
    seg_mask[plate_mask == 0] = 0
    hsv = cv2.cvtColor(crop_original, cv2.COLOR_BGR2HSV)

    # radio relativo del centroide de cada region, para las reglas del borde
    hc, wc = crop_original.shape[:2]
    pcx, pcy = wc / 2.0, hc / 2.0
    plate_r = min(hc, wc) / 2.0

    def pasa_borde(p):
        """En la banda exterior exige forma de colonia (redonda y solida),
        porque ahi viven el menisco y la condensacion (arcos alargados)."""
        cy_, cx_ = p.centroid
        rad = np.hypot(cx_ - pcx, cy_ - pcy) / plate_r
        if rad <= RIM_BAND:
            return True
        return p.solidity >= RIM_MIN_SOLIDITY and p.eccentricity <= RIM_MAX_ECC

    props = regionprops(seg_mask)
    base = [p for p in props
            if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
            and p.solidity >= MIN_SOLIDITY]
    valid = [p for p in base if not is_ink(hsv, seg_mask, p.label)
             and pasa_borde(p)]
    n_ink = sum(1 for p in base if is_ink(hsv, seg_mask, p.label))
    n_rim = sum(1 for p in base if not is_ink(hsv, seg_mask, p.label)
                and not pasa_borde(p))
    return seg_mask, valid, n_ink, n_rim


def draw_overlay(crop_bgr, label_mask, valid_props):
    rng = np.random.default_rng(seed=42)
    overlay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for p in valid_props:
        color  = rng.uniform(0.2, 1.0, size=3)
        region = label_mask == p.label
        overlay[region] = overlay[region] * 0.3 + color * 0.7
    return overlay


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('images/mis_fotos')
    # la salida se deriva del nombre de la carpeta de entrada, para que correr
    # un lote nuevo no sobrescriba los resultados de otro
    out_dir = Path('results/colonias') / images_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in SUPPORTED
                    and not p.name.startswith('ground'))

    print(f'Imagenes: {len(images)}  shrink={SHRINK}\n')
    print('Cargando modelo CellSAM...')
    model = get_model()
    print('Listo.\n')

    header = (f'{"imagen":<20} {"adapt":>5} {"thr":>5} {"conteo":>6} '
              f'{"tinta":>6} {"borde":>6}  razon')
    print(header)
    print('-' * len(header))

    rows = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        cx, cy, r = detect_plate(img)
        crop, plate_mask = crop_plate(img, cx, cy, r)

        flat = flat_field(crop, plate_mask)
        n_adapt, thr, razon = scout(flat, plate_mask)
        seg_mask, valid, n_ink, n_rim = segment(model, flat, crop,
                                                plate_mask, thr)
        n = len(valid)

        print(f'{img_path.stem:<20} {n_adapt:>5} {thr:>5.2f} {n:>6} '
              f'{n_ink:>6} {n_rim:>6}  {razon}')
        rows.append({
            'image': img_path.stem, 'count': n,
            'n_adapt': n_adapt, 'bbox_threshold': thr,
            'descartadas_tinta': n_ink, 'descartadas_borde': n_rim,
        })

        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        fig.suptitle(img_path.name, fontsize=13, fontweight='bold')
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        axes[0].imshow(crop_rgb)
        axes[0].set_title('Placa original', fontsize=11)
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(flat, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Corregida (entrada al modelo)', fontsize=11)
        axes[1].axis('off')
        axes[2].imshow(draw_overlay(crop, seg_mask, valid))
        axes[2].set_title(f'n={n}  thr={thr:.2f}  adapt={n_adapt}', fontsize=11)
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f'{img_path.stem}_count.png', dpi=120, bbox_inches='tight')
        plt.close()

    print('-' * len(header))
    pd.DataFrame(rows).to_csv(out_dir / 'summary.csv', index=False)
    print(f'\nCSV guardado: {out_dir / "summary.csv"}')
    print(f'PNGs:         {out_dir}/')


if __name__ == '__main__':
    main()
