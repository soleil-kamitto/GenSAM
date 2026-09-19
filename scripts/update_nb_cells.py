import json

NB = r'c:\Users\Sol\cellsam_project\cellsam\notebooks\cellsam_actinomicetos_clean.ipynb'
nb = json.load(open(NB, encoding='utf-8'))

SCOUT_MD = """\
## 6. Segmentacion con CellSAM - Scout adaptativo

En lugar de un `bbox_threshold` fijo, se usa un **clasificador rapido de vision clasica** (top-hat morfologico) para estimar la densidad de colonias en cada placa en milisegundos. Con esa estimacion se asigna automaticamente el threshold optimo:

| Densidad estimada (area media por objeto) | `bbox_threshold` asignado |
|------------------------------------------|--------------------------|
| Colonias muy juntas (area media > 6000 px2) | 0.50 |
| Densidad moderada (area media > 3500 px2)   | 0.65 |
| Colonias separadas                           | 0.80 |

El resto de parametros: `normalize=True`, `postprocess=True`. Los filtros de area (300-50000 px2) y solidez (>= 0.50) eliminan artefactos de borde.
"""

SCOUT_CODE = """\
MIN_COLONY_AREA = 300
MAX_COLONY_AREA = 50000
MIN_SOLIDITY    = 0.50


def apply_clahe(crop_bgr, plate_mask):
    lab   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced[plate_mask == 0] = 0
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)


def scout_threshold(crop_bgr, plate_mask):
    gray   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray   = cv2.bitwise_and(gray, gray, mask=plate_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, bw  = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)
    bw     = cv2.bitwise_and(bw, bw, mask=plate_mask)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    areas = stats[1:, cv2.CC_STAT_AREA]
    mean_area = float(areas.mean()) if len(areas) > 0 else 0
    if mean_area > 6000:
        return 0.50
    elif mean_area > 3500:
        return 0.65
    else:
        return 0.80


def segment_cellsam(crop_bgr, plate_mask):
    thr      = scout_threshold(crop_bgr, plate_mask)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mask, _, _ = segment_cellular_image(
        crop_rgb, model=model,
        normalize=True, postprocess=True,
        bbox_threshold=thr, device='cpu',
    )
    mask = mask.copy()
    mask[plate_mask == 0] = 0
    props = regionprops(mask)
    valid = [p for p in props
             if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY]
    return mask, valid, thr


def draw_instance_overlay(crop_bgr, valid_props, label_mask, plate_mask):
    rng     = np.random.default_rng(42)
    overlay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay[plate_mask == 0] = 0
    colors  = rng.uniform(0.3, 1.0, size=(len(valid_props), 3))
    for prop, color in zip(valid_props, colors):
        region = label_mask == prop.label
        overlay[region] = overlay[region] * 0.35 + color * 0.65
    return np.clip(overlay, 0, 1)


print('Funciones scout adaptativo definidas.')
"""

PIPELINE_CODE = """\
image_files = sorted(IMAGES_DIR.glob('actinomicetos_*.jpeg'))
results = []

for img_path in image_files:
    stem   = img_path.stem
    img    = cv2.imread(str(img_path))
    plates = detect_plates(img)
    panels = []
    counts = []

    for idx, (cx, cy, r) in enumerate(plates):
        placa_id    = 'AB'[idx]
        gt_val      = gt.get(stem, (None, None))[idx]
        crop, pmask = crop_plate(img, cx, cy, r, shrink=1.0)
        lmask, valid, thr = segment_cellsam(crop, pmask)
        n_col = len(valid)
        counts.append(n_col)

        original_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        original_rgb[pmask == 0] = 0
        clahe_rgb   = apply_clahe(crop, pmask)
        overlay_rgb = draw_instance_overlay(crop, valid, lmask, pmask)
        panels.append((original_rgb, clahe_rgb, overlay_rgb, placa_id, n_col, gt_val, thr))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(stem.replace('actinomicetos_', 'Actinomicetos '),
                 fontsize=15, fontweight='bold', y=1.01)
    for col, title in enumerate(['Original', 'Contraste (CLAHE)', 'Segmentacion CellSAM']):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=10)
    for row, (orig, clahe, overlay, placa_id, n_col, gt_val, thr) in enumerate(panels):
        axes[row, 0].imshow(orig)
        axes[row, 1].imshow(clahe)
        axes[row, 2].imshow(overlay)
        axes[row, 0].set_ylabel(f'Placa {placa_id}', fontsize=12, fontweight='bold', labelpad=10)
        gt_str = f'GT={gt_val}' if gt_val is not None else ''
        axes[row, 2].set_xlabel(f'CellSAM={n_col}  thr={thr}  {gt_str}', fontsize=10, labelpad=6)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{stem}_3panel.png', dpi=120, bbox_inches='tight')
    plt.show()

    results.append({'image': stem,
                    'cellsam_A': counts[0], 'cellsam_B': counts[1],
                    'gt_A': gt.get(stem, (None, None))[0],
                    'gt_B': gt.get(stem, (None, None))[1]})

print('Procesamiento completado.')
"""

for cell in nb['cells']:
    if cell['id'] == 'cellsam-08':
        cell['source'] = SCOUT_MD
    elif cell['id'] == 'code-cellsam':
        cell['source'] = SCOUT_CODE
        cell['outputs'] = []
    elif cell['id'] == 'code-pipeline':
        cell['source'] = PIPELINE_CODE
        # keep outputs (already embedded)

json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('Celdas actualizadas correctamente.')
