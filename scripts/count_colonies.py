"""
CellSAM-based macroscopic colony counter for Petri dish images.
Usage: python scripts/count_colonies.py <image_path>
Example: python scripts/count_colonies.py images/placas/actinomicetos_1.jpeg

Each image must contain two Petri dishes (landscape: left/right,
portrait: top/bottom). Both plates are replicates of the same sample;
their counts should be close — the script reports a reliability indicator.

CellSAM handles segmentation. Classical CV is used only for plate detection
and cropping. Results are saved to results/colonies/.

Tuning: adjust the PARAMETERS block below if counts are consistently off.
"""

import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops

from cellSAM import get_model, segment_cellular_image


# ── PARAMETERS ────────────────────────────────────────────────────────────────
MIN_COLONY_AREA = 300    # px² — blobs smaller than this are noise
MAX_COLONY_AREA = 50000  # px² — blobs larger than this are merged/artifacts
MIN_SOLIDITY    = 0.50   # filter crescents/arcs (glass rim artifacts)
NORMALIZE       = True   # CellSAM: percentile normalisation + CLAHE
                          # set False if image is already well-normalised
POSTPROCESS     = False  # CellSAM: extra postprocessing for noisy images

# ── MODEL (loaded once, reused across plates/images) ─────────────────────────
_model = None


def _get_model():
    global _model
    if _model is None:
        print("  Loading CellSAM model (first time only)...")
        _model = get_model()
        print("  Model ready.")
    return _model


# ── PLATE DETECTION ───────────────────────────────────────────────────────────

def detect_plates(img_bgr):
    """
    Detect one Petri dish per image half using HoughCircles.

    Portrait (h > w): top/bottom halves.
    Landscape (w >= h): left/right halves.

    Returns [(cx, cy, r), ...] ordered top-bottom or left-right.
    """
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
        gray    = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(hh, hw),
            param1=60,
            param2=25,
            minRadius=int(min(hh, hw) * 0.30),
            maxRadius=int(min(hh, hw) * 0.52),
        )

        if circles is not None:
            best = np.round(circles[0][0]).astype(int)
            cx, cy, r = int(best[0]) + ox, int(best[1]) + oy, int(best[2])
        else:
            print(f"  Warning: no circle in half {i+1}, using fallback.")
            cx = hw // 2 + ox
            cy = hh // 2 + oy
            r  = int(min(hh, hw) * 0.43)

        plates.append((cx, cy, r))

    return plates


def crop_plate(img_bgr, cx, cy, r, shrink=0.86):
    """
    Crop a circular plate region from the image.
    Returns (crop_bgr, circular_mask, x1, y1).
    Pixels outside the circle are set to 0.
    """
    r_use = int(r * shrink)
    x1 = max(0, cx - r_use)
    y1 = max(0, cy - r_use)
    x2 = min(img_bgr.shape[1], cx + r_use)
    y2 = min(img_bgr.shape[0], cy + r_use)

    crop   = img_bgr[y1:y2, x1:x2].copy()
    hc, wc = crop.shape[:2]
    cx_l   = cx - x1
    cy_l   = cy - y1

    mask = np.zeros((hc, wc), dtype=np.uint8)
    cv2.circle(mask, (cx_l, cy_l), r_use, 255, -1)
    crop[mask == 0] = 0

    return crop, mask, x1, y1


# ── CELLSAM SEGMENTATION ──────────────────────────────────────────────────────

def segment_colonies(crop_bgr, plate_mask):
    """
    Segment colonies in a cropped plate using CellSAM.

    CellSAM receives an RGB crop with the circular plate region intact and
    black outside. Its output mask is then filtered by area and solidity to
    remove noise and rim artifacts.

    Returns (label_mask, valid_regionprops, all_detections_binary).
    """
    model    = _get_model()
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    mask, _, _ = segment_cellular_image(
        crop_rgb,
        model=model,
        normalize=NORMALIZE,
        postprocess=POSTPROCESS,
        device="cpu",
    )

    # Zero out anything outside the circular plate
    mask = mask.copy()
    mask[plate_mask == 0] = 0

    # Binary of all CellSAM detections (for visualization)
    binary_all = (mask > 0).astype(np.uint8) * 255

    # Filter by area and solidity
    props = regionprops(mask)
    valid = [p for p in props
             if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY]

    return mask, valid, binary_all


# ── VISUALIZATION ─────────────────────────────────────────────────────────────

def draw_overlay(crop_bgr, valid_props, label_mask):
    overlay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).copy().astype(np.float32) / 255.0
    for prop in valid_props:
        region = label_mask == prop.label
        overlay[region] = overlay[region] * 0.45 + np.array([0.15, 0.85, 0.35]) * 0.55
    return overlay


# ── MAIN PROCESSING ───────────────────────────────────────────────────────────

def process_image(img_path, output_dir):
    """
    Detect plates, run CellSAM segmentation, save visualisation.
    Returns list of colony counts [plate_A, plate_B].
    """
    img_path   = Path(img_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")

    h_img, w_img = img.shape[:2]
    orientation  = "portrait" if h_img > w_img else "landscape"
    print(f"  {w_img}x{h_img} ({orientation})")

    plates      = detect_plates(img)
    plate_names = ["A", "B"]
    counts      = []

    n_plates = len(plates)
    fig, axes = plt.subplots(n_plates, 3, figsize=(15, 6 * n_plates))
    if n_plates == 1:
        axes = axes[np.newaxis, :]

    for i, (cx, cy, r) in enumerate(plates):
        name = plate_names[i] if i < len(plate_names) else str(i + 1)
        print(f"  Plate {name}: circle at ({cx},{cy}) r={r}")

        crop, mask, _, _ = crop_plate(img, cx, cy, r)
        label_mask, valid, binary_all = segment_colonies(crop, mask)
        n = len(valid)
        counts.append(n)
        print(f"  Plate {name}: {n} colonies")

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        overlay  = draw_overlay(crop, valid, label_mask)

        axes[i, 0].imshow(crop_rgb)
        axes[i, 0].set_title(f"Plate {name} — original", fontsize=13)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(binary_all, cmap="gray")
        axes[i, 1].set_title(f"Plate {name} — CellSAM detections", fontsize=13)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Plate {name} — {n} colonies (filtered)", fontsize=13,
                             fontweight="bold")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = output_dir / f"{img_path.stem}_colony_count.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    if len(counts) == 2 and all(c > 0 for c in counts):
        diff_pct = abs(counts[0] - counts[1]) / max(counts) * 100
        tag = "CONFIABLE" if diff_pct <= 10 else ("ACEPTABLE" if diff_pct <= 20 else "REVISAR")
        print(f"  Diferencia entre placas: {diff_pct:.1f}%  [{tag}]")

    return counts


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img_path   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("images/placa.jpg")
    output_dir = Path("results/colonies")

    print(f"\n[1] Processing: {img_path}")
    counts = process_image(img_path, output_dir)

    print(f"\n[2] Saved: {output_dir}/{img_path.stem}_colony_count.png")
    print("\n" + "=" * 45)
    print("  COLONY COUNT SUMMARY")
    print("=" * 45)
    for i, n in enumerate(counts):
        print(f"  Plate {chr(65+i)}: {n:>4} colonies")
    if len(counts) == 2 and all(c > 0 for c in counts):
        diff_pct = abs(counts[0] - counts[1]) / max(counts) * 100
        tag = "CONFIABLE" if diff_pct <= 10 else ("ACEPTABLE" if diff_pct <= 20 else "REVISAR")
        print(f"  Diferencia: {diff_pct:.1f}%  [{tag}]")
    print("=" * 45)
