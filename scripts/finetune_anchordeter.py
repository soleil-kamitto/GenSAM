"""
Fine-tuning de AnchorDETR para detectar colonias sin normalize=False.

Sin CLAHE (normalize=False), AnchorDETR no encuentra nada porque el contraste
es bajo. Este script lo entrena con imagenes SIN CLAHE usando como etiquetas
las detecciones del modelo CON CLAHE (normalize=True).

Optimizacion clave: el backbone (SAM ViT-B) es congelado y sus features se
pre-computan una sola vez, evitando el backward por la red grande en cada paso.
Solo se entrenan el transformer y la capa de proyeccion de AnchorDETR.

Uso:
    python scripts/finetune_anchordeter.py images/placas/ [epochs]

Guarda:
    results/colonias/finetuning/cellsam_anchordeter.pt
"""

import sys
import csv
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from cellSAM import get_model, segment_cellular_image
from cellSAM.AnchorDETR.util.misc import nested_tensor_from_tensor_list


# ── PARAMETROS ────────────────────────────────────────────────────────────────
MIN_COLONY_AREA = 300
MAX_COLONY_AREA = 50000
MIN_SOLIDITY    = 0.50
BBOX_THRESHOLD  = 0.4
LR              = 5e-6
SUPPORTED       = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
OUT_DIR         = Path('results/colonias/finetuning')


# ── BOUNDING BOX UTILS ────────────────────────────────────────────────────────

def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def giou_matrix(b1, b2):
    a1 = (b1[:, 2]-b1[:, 0]).clamp(0) * (b1[:, 3]-b1[:, 1]).clamp(0)
    a2 = (b2[:, 2]-b2[:, 0]).clamp(0) * (b2[:, 3]-b2[:, 1]).clamp(0)
    lt   = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb   = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    inter = (rb - lt).clamp(0).prod(-1)
    union = a1[:, None] + a2[None, :] - inter
    iou   = inter / union.clamp(1e-6)
    lt_e  = torch.min(b1[:, None, :2], b2[None, :, :2])
    rb_e  = torch.max(b1[:, None, 2:], b2[None, :, 2:])
    enc   = (rb_e - lt_e).clamp(0).prod(-1)
    return iou - (enc - union) / enc.clamp(1e-6)


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


# ── GT BOXES CON normalize=True ───────────────────────────────────────────────

def get_gt_boxes(model, crop_rgb, plate_mask):
    try:
        mask, _, _ = segment_cellular_image(
            crop_rgb, model=model,
            normalize=True, postprocess=False,
            bbox_threshold=BBOX_THRESHOLD, device='cpu',
        )
        if mask is None:
            return np.zeros((0, 4), dtype=np.float32)
    except (AttributeError, TypeError):
        return np.zeros((0, 4), dtype=np.float32)

    mask = mask.copy()
    mask[plate_mask == 0] = 0
    h, w = crop_rgb.shape[:2]
    boxes = []
    for p in regionprops(mask):
        if not (MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA and p.solidity >= MIN_SOLIDITY):
            continue
        r0, c0, r1, c1 = p.bbox
        boxes.append([
            (c0 + c1) / 2 / w,
            (r0 + r1) / 2 / h,
            (c1 - c0) / w,
            (r1 - r0) / h,
        ])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


# ── PERDIDA DE DETECCION ──────────────────────────────────────────────────────

def detection_loss(outputs, gt_t):
    n_gt        = gt_t.shape[0]
    pred_logits = outputs['pred_logits'][0]
    pred_boxes  = outputs['pred_boxes'][0]

    if n_gt == 0:
        return 0.01 * pred_logits.sigmoid().sum()

    with torch.no_grad():
        px = cxcywh_to_xyxy(pred_boxes)
        gx = cxcywh_to_xyxy(gt_t)
        cost = (5 * torch.cdist(pred_boxes, gt_t, p=1)
                - 2 * giou_matrix(px, gx)).cpu().numpy()
        ri, ci = linear_sum_assignment(cost)
        ri = torch.tensor(ri, dtype=torch.long)
        ci = torch.tensor(ci, dtype=torch.long)

    loss_l1  = F.l1_loss(pred_boxes[ri], gt_t[ci])
    mp, mg   = cxcywh_to_xyxy(pred_boxes[ri]), cxcywh_to_xyxy(gt_t[ci])
    loss_giou = (1 - giou_matrix(mp, mg).diag()).mean()

    alpha, gamma = 0.25, 2.0
    target = torch.zeros_like(pred_logits)
    target[ri, 0] = 1.0
    p   = pred_logits.sigmoid()
    ce  = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    p_t = p * target + (1 - p) * (1 - target)
    fw  = (alpha * target + (1 - alpha) * (1 - target)) * (1 - p_t).pow(gamma)
    loss_ce = (fw * ce).sum() / n_gt

    return 5.0 * loss_l1 + 2.0 * loss_giou + 2.0 * loss_ce


# ── FINE-TUNING ───────────────────────────────────────────────────────────────

def finetune(model, samples, num_epochs):
    for p in model.parameters():
        p.requires_grad = False

    detr = model.cellfinder.decode_head
    for p in detr.transformer.parameters():
        p.requires_grad = True
    for p in detr.input_proj.parameters():
        p.requires_grad = True

    n_tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f'Parametros entrenables: {n_tr:,} / {n_all:,}')

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Pre-computar features del backbone (congelado) con imagenes RAW
    # Se hace solo UNA vez antes del loop de entrenamiento
    print('Pre-computando features del backbone con imagenes sin CLAHE...')
    precomp = []
    model.eval()
    with torch.no_grad():
        for idx, s in enumerate(samples):
            img_raw   = torch.from_numpy(
                s['crop_rgb'].astype(np.float32).transpose(2, 0, 1)
            )
            imgs_detr = model.sam_bbox_preprocessing([img_raw], percentile=False)
            nested    = nested_tensor_from_tensor_list(imgs_detr)
            bb_feats  = detr.backbone(nested)

            srcs_pre, masks_pre = [], []
            for feat in bb_feats:
                src, mask = feat.decompose()
                srcs_pre.append(src.detach().clone())
                masks_pre.append(mask.detach().clone())

            precomp.append({
                'srcs':  srcs_pre,
                'masks': masks_pre,
                'gt_t':  torch.tensor(s['gt_boxes'], dtype=torch.float32),
            })
            print(f'  {idx+1}/{len(samples)}  bboxes_gt={len(s["gt_boxes"])}', flush=True)

    print('Features listas.\n')

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        model.train()

        for pc in precomp:
            # input_proj sobre features pre-computadas (entrenable)
            srcs = torch.cat([
                detr.input_proj[l](src).unsqueeze(1)
                for l, src in enumerate(pc['srcs'])
            ], dim=1)

            # transformer (entrenable)
            out_class, out_coord = detr.transformer(srcs, pc['masks'])

            outputs = {
                'pred_logits': out_class[-1].unsqueeze(0),
                'pred_boxes':  out_coord[-1].unsqueeze(0),
                'aux_outputs': [
                    {'pred_logits': c.unsqueeze(0), 'pred_boxes': b.unsqueeze(0)}
                    for c, b in zip(out_class[:-1], out_coord[:-1])
                ],
            }

            loss = detection_loss(outputs, pc['gt_t'])
            for aux in outputs['aux_outputs']:
                loss = loss + 0.4 * detection_loss(aux, pc['gt_t'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / max(len(precomp), 1)
        print(f'  Epoca {epoch:2d}/{num_epochs}  loss={avg:.4f}', flush=True)


# ── EVALUACION ────────────────────────────────────────────────────────────────

def count_colonies(model, images_dir, normalize):
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
                    normalize=normalize, postprocess=False,
                    bbox_threshold=BBOX_THRESHOLD, device='cpu',
                )
                if mask is None:
                    mask = np.zeros(crop.shape[:2], dtype=np.int32)
            except (AttributeError, TypeError):
                mask = np.zeros(crop.shape[:2], dtype=np.int32)
            mask = mask.copy()
            mask[plate_mask == 0] = 0
            n = sum(1 for p in regionprops(mask)
                    if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                    and p.solidity >= MIN_SOLIDITY)
            counts.append(n)
        results[img_path.stem] = counts
    return results


def print_results(counts, gt, label=''):
    errs = []
    tag = f'  [{label}]' if label else ''
    print(f'\n{"imagen":<25} {"A":>6} {"A_gt":>6} {"errA":>6} '
          f'{"B":>6} {"B_gt":>6} {"errB":>6}{tag}')
    print('-' * 65)
    for stem, (ca, cb) in counts.items():
        ga, gb = gt.get(stem, (None, None))
        if ga is not None:
            errs += [abs(ca - ga), abs(cb - gb)]
            print(f'{stem:<25} {ca:>6} {ga:>6} {ca-ga:>+6} {cb:>6} {gb:>6} {cb-gb:>+6}')
        else:
            print(f'{stem:<25} {ca:>6} {"?":>6} {"?":>6} {cb:>6} {"?":>6} {"?":>6}')
    mae = np.mean(errs) if errs else None
    if mae is not None:
        print(f'\nMAE: {mae:.1f} colonias/placa')
    return mae


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('images/placas')
    num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_path = images_dir / 'ground_truth.csv'
    gt = {}
    if gt_path.exists():
        with open(gt_path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                stem = Path(row['image']).stem
                gt[stem] = (int(row['plate_A']), int(row['plate_B']))
        print(f'Ground truth: {len(gt)} imagenes\n')

    print('Cargando modelo CellSAM...')
    model = get_model()
    print('Listo.\n')

    print('=== REFERENCIA: normalize=True ===')
    print_results(count_colonies(model, images_dir, normalize=True), gt, 'normalize=True')

    print('\n=== ANTES: normalize=False (sin fine-tuning) ===')
    mae_before = print_results(count_colonies(model, images_dir, normalize=False), gt, 'normalize=False')

    print('\n=== Generando GT bboxes con normalize=True ===')
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
            gt_boxes  = get_gt_boxes(model, crop_rgb, plate_mask)
            print(f'    Placa {"AB"[i]}: {len(gt_boxes)} bboxes')
            samples.append({'crop_rgb': crop_rgb, 'gt_boxes': gt_boxes})

    print(f'\nTotal: {sum(len(s["gt_boxes"]) for s in samples)} bboxes en {len(samples)} placas')

    print(f'\n=== Fine-tuning AnchorDETR ({num_epochs} epocas, lr={LR}) ===')
    finetune(model, samples, num_epochs)

    print('\n=== DESPUES: normalize=False ===')
    counts_after = count_colonies(model, images_dir, normalize=False)
    mae_after = print_results(counts_after, gt, 'normalize=False')

    # Guardar CSV para el grafico comparativo
    import pandas as pd
    rows = []
    for stem, (ca, cb) in counts_after.items():
        rows.append({'image': stem, 'plate_A': ca, 'plate_B': cb})
    csv_path = OUT_DIR / 'anchordeter_finetuned_summary.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'CSV guardado: {csv_path}')

    save_path = OUT_DIR / 'cellsam_anchordeter.pt'
    torch.save(model.state_dict(), save_path)
    print(f'Pesos guardados: {save_path}')

    if mae_before is not None and mae_after is not None:
        print(f'\nResumen normalize=False:  antes={mae_before:.1f}  despues={mae_after:.1f}  '
              f'mejora={mae_before - mae_after:+.1f}')
    print('Referencia normalize=True: MAE~12.8')


if __name__ == '__main__':
    main()
