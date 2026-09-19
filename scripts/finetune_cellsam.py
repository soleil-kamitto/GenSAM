"""
Fine-tuning del mask decoder de CellSAM para colonias de actinomicetos.

Solo se entrena el mask decoder de SAM (el componente que convierte bounding
boxes en mascaras). El encoder de imagenes y AnchorDETR se mantienen congelados.

Pasos:
  1. Evaluar conteos ANTES del fine-tuning.
  2. Generar pseudo-etiquetas usando el propio CellSAM con normalize=True.
  3. Pre-computar embeddings del encoder (una vez, ya que esta congelado).
  4. Fine-tuning del mask decoder con esas pseudo-etiquetas.
  5. Evaluar conteos DESPUES del fine-tuning.

Uso:
    python scripts/finetune_cellsam.py images/placas/ [epochs]

Guarda:
    results/finetuned/cellsam_finetuned.pt
"""

import sys
import csv
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from skimage.measure import regionprops
from skimage.measure import label as relabel
from torchvision import tv_tensors

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from cellSAM import get_model, segment_cellular_image


# ── PARAMETROS ────────────────────────────────────────────────────────────────
MIN_COLONY_AREA = 300
MAX_COLONY_AREA = 50000
MIN_SOLIDITY    = 0.50
BBOX_THRESHOLD  = 0.4
LR              = 1e-5
SAM_SIZE        = 1024
SUPPORTED       = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
OUT_DIR         = Path('results/colonias/finetuning')


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


# ── FUNCIONES DE PERDIDA ──────────────────────────────────────────────────────

def dice_loss(logit, target, smooth=1.0):
    p     = logit.sigmoid()
    inter = (p * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (p.sum() + target.sum() + smooth)


def bce_dice(logit, target):
    return (0.5 * F.binary_cross_entropy_with_logits(logit, target)
            + 0.5 * dice_loss(logit, target))


# ── PSEUDO-ETIQUETAS ──────────────────────────────────────────────────────────

def get_pseudo_labels(model, crop_rgb, plate_mask):
    """Corre CellSAM con normalize=True y filtra colonias validas."""
    try:
        mask, _, _ = segment_cellular_image(
            crop_rgb, model=model,
            normalize=True, postprocess=False,
            bbox_threshold=BBOX_THRESHOLD, device='cpu',
        )
        if mask is None:
            return np.zeros(crop_rgb.shape[:2], dtype=np.int32)
    except (AttributeError, TypeError):
        return np.zeros(crop_rgb.shape[:2], dtype=np.int32)

    mask = mask.copy()
    mask[plate_mask == 0] = 0

    clean = np.zeros_like(mask)
    for p in regionprops(mask):
        if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA and p.solidity >= MIN_SOLIDITY:
            clean[mask == p.label] = p.label

    return relabel(clean > 0).astype(np.int32)


def prepare_samples(model, images_dir):
    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in SUPPORTED
                    and not p.name.startswith('ground'))
    samples = []
    for img_path in images:
        print(f'  {img_path.name}')
        img = cv2.imread(str(img_path))
        for i, (cx, cy, r) in enumerate(detect_plates(img)):
            crop, plate_mask = crop_plate(img, cx, cy, r)
            crop_rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inst_mask = get_pseudo_labels(model, crop_rgb, plate_mask)
            n = inst_mask.max()
            print(f'    Placa {"AB"[i]}: {n} colonias (pseudo-etiquetas)')
            samples.append({'crop_rgb': crop_rgb, 'instance_mask': inst_mask})
    return samples


# ── FINE-TUNING ───────────────────────────────────────────────────────────────

def finetune(model, samples, num_epochs):
    # Congelar todo
    for p in model.parameters():
        p.requires_grad = False
    # Solo entrenar el mask decoder
    for p in model.model_cp.mask_decoder.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f'Parametros entrenables: {n_trainable:,} / {n_total:,}')

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Pre-computar embeddings del encoder (congelado → se hace una sola vez)
    print('Pre-computando embeddings del encoder...')
    precomp = []
    for s in samples:
        crop_rgb = s['crop_rgb']
        h, w = crop_rgb.shape[:2]
        img_tv = tv_tensors.Image(torch.from_numpy(crop_rgb).permute(2, 0, 1).float())
        with torch.no_grad():
            emb, paddings = model.generate_embeddings([img_tv])
        precomp.append({'emb': emb, 'padh': paddings[0][0], 'padw': paddings[0][1],
                        'h': h, 'w': w})
    print(f'Listos. {len(precomp)} placas.\n')

    model.train()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        n_updates  = 0

        for s, pc in zip(samples, precomp):
            instance_mask = s['instance_mask']
            emb           = pc['emb']          # (1, 256, 64, 64)
            padh, padw    = pc['padh'], pc['padw']
            h_orig, w_orig = pc['h'], pc['w']

            props = regionprops(instance_mask)
            if not props:
                continue

            # Escala de coordenadas originales → 1024x1024
            sx = SAM_SIZE / w_orig
            sy = SAM_SIZE / h_orig

            for prop in props:
                r0, c0, r1, c1 = prop.bbox

                # Bounding box en espacio SAM (1024x1024)
                box = torch.tensor(
                    [[c0 * sx, r0 * sy, c1 * sx, r1 * sy]], dtype=torch.float32
                ).unsqueeze(0)  # (1, 1, 4)

                # Mascara GT binaria en resolucion original
                gt = torch.from_numpy(
                    (instance_mask == prop.label).astype(np.float32)
                )  # (h_orig, w_orig)

                # Forward del mask decoder (entrenable)
                sparse_e, dense_e = model.model_cp.prompt_encoder(
                    points=None, boxes=box, masks=None
                )
                low_res, _ = model.model_cp.mask_decoder(
                    image_embeddings=emb[0].unsqueeze(0),
                    image_pe=model.model_cp.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_e,
                    dense_prompt_embeddings=dense_e,
                    multimask_output=False,
                )  # (1, 1, 256, 256)

                # Upsample a resolucion original
                pred = model.model_cp.postprocess_masks(
                    low_res,
                    input_size=(SAM_SIZE - padh, SAM_SIZE - padw),
                    original_size=(h_orig, w_orig),
                )[0, 0]  # (h_orig, w_orig)

                loss = bce_dice(pred, gt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_updates  += 1

        scheduler.step()
        avg = total_loss / max(n_updates, 1)
        print(f'  Epoca {epoch:2d}/{num_epochs}  loss={avg:.4f}  (colonias={n_updates})')


# ── EVALUACION ────────────────────────────────────────────────────────────────

def count_colonies(model, images_dir):
    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in SUPPORTED
                    and not p.name.startswith('ground'))
    results = {}
    model.eval()
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
                    bbox_threshold=BBOX_THRESHOLD, device='cpu',
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


def print_results(counts, gt):
    errs = []
    print(f'\n{"imagen":<25} {"A_pred":>6} {"A_gt":>6} {"err_A":>6} '
          f'{"B_pred":>6} {"B_gt":>6} {"err_B":>6}')
    print('-' * 62)
    for stem, (ca, cb) in counts.items():
        ga, gb = gt.get(stem, (None, None))
        if ga is not None:
            errs += [abs(ca - ga), abs(cb - gb)]
            print(f'{stem:<25} {ca:>6} {ga:>6} {ca-ga:>+6} '
                  f'{cb:>6} {gb:>6} {cb-gb:>+6}')
        else:
            print(f'{stem:<25} {ca:>6} {"?":>6} {"?":>6} '
                  f'{cb:>6} {"?":>6} {"?":>6}')
    mae = np.mean(errs) if errs else None
    if mae is not None:
        print(f'\nError absoluto medio: {mae:.1f} colonias/placa')
    return mae


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('images/placas')
    num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_path = images_dir / 'ground_truth.csv'
    gt = {}
    if gt_path.exists():
        with open(gt_path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                stem = Path(row['image']).stem
                gt[stem] = (int(row['plate_A']), int(row['plate_B']))
        print(f'Ground truth cargado: {len(gt)} imagenes\n')

    print('Cargando modelo CellSAM...')
    model = get_model()
    print('Modelo listo.\n')

    # Paso 1: Evaluacion ANTES
    print('=' * 62)
    print('  ANTES del fine-tuning')
    print('=' * 62)
    mae_before = print_results(count_colonies(model, images_dir), gt)

    # Paso 2: Pseudo-etiquetas
    print('\n' + '=' * 62)
    print('  Generando pseudo-etiquetas')
    print('=' * 62)
    samples = prepare_samples(model, images_dir)
    total_colonias = sum(int(s['instance_mask'].max()) for s in samples)
    print(f'\nTotal: {len(samples)} placas, {total_colonias} colonias para entrenamiento')

    # Paso 3: Fine-tuning
    print('\n' + '=' * 62)
    print(f'  Fine-tuning  ({num_epochs} epocas, lr={LR})')
    print('=' * 62)
    finetune(model, samples, num_epochs)

    # Paso 4: Evaluacion DESPUES
    print('\n' + '=' * 62)
    print('  DESPUES del fine-tuning')
    print('=' * 62)
    mae_after = print_results(count_colonies(model, images_dir), gt)

    # Guardar pesos
    save_path = OUT_DIR / 'cellsam_finetuned.pt'
    torch.save(model.state_dict(), save_path)
    print(f'\nPesos guardados: {save_path}')

    if mae_before is not None and mae_after is not None:
        delta = mae_before - mae_after
        print(f'\nResumen  MAE antes={mae_before:.1f}  despues={mae_after:.1f}  '
              f'cambio={delta:+.1f} colonias/placa')


if __name__ == '__main__':
    main()
