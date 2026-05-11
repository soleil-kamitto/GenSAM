"""
Macroscopic colony counter for Petri dish images.
Usage: python scripts/count_colonies.py <image_path>
Example: python scripts/count_colonies.py images/placa_lab.jpg

Each image must contain two Petri dishes (stacked vertically for portrait
images, or side by side for landscape). Both plates are replicates of the
same sample; their counts should be close — the script reports a reliability
indicator based on the difference between them.

Segmentation uses LAB color distance from the agar background, so it detects
white, cream, and colored colonies without needing per-plate tuning.

Results are saved to results/colonies/.

Tuning: adjust the PARAMETERS block below if counts are consistently off.
"""

import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, label as sk_label


# ── PARAMETERS ────────────────────────────────────────────────────────────────
MIN_COLONY_AREA    = 300   # px² — blobs smaller than this are noise
MAX_COLONY_AREA    = 50000 # px² — blobs larger than this are merged/artifacts
MIN_SOLIDITY       = 0.50  # filter crescents/arcs (glass rim artifacts)
WATERSHED_MIN_DIST = 20    # px — min distance between colony centers
                            # increase if overcounting, decrease if undercounting
MAX_HOLE_AREA      = 400   # px² — only fill holes this small (colony center dots)
OPEN_ITERATIONS    = 1     # morphological open passes to remove speckle
# Top-hat channel (catches white/cream colonies brighter than background)
TOPHAT_KERNEL      = 81    # px — must be larger than the biggest colony
TOPHAT_THRESHOLD   = 10    # intensity cutoff on top-hat output (0-255)
# Saturation channel (catches colored colonies: pink, orange, yellow, etc.)
SAT_THRESHOLD      = 35    # HSV saturation cutoff (0-255); lower = more sensitive


def detect_plates(img_bgr):
    """
    Detect one Petri dish per image half.

    For portrait images (h > w): top half / bottom half.
    For landscape images (w >= h): left half / right half.

    This prevents HoughCircles from finding two circles in the same half.
    Returns [(cx, cy, r), ...] ordered: top-then-bottom or left-then-right.
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

        min_r = int(min(hh, hw) * 0.30)
        max_r = int(min(hh, hw) * 0.52)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(hh, hw),      # allow only one circle per half
            param1=60,
            param2=25,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is not None:
            best = np.round(circles[0][0]).astype(int)
            cx, cy, r = int(best[0]) + ox, int(best[1]) + oy, int(best[2])
        else:
            print(f"    Warning: no circle found in half {i+1}, using fallback.")
            cx = hw // 2 + ox
            cy = hh // 2 + oy
            r  = int(min(hh, hw) * 0.43)

        plates.append((cx, cy, r))

    return plates


def crop_plate(img_bgr, cx, cy, r, shrink=0.86):
    """
    Return (crop, circular_mask, x_offset, y_offset).
    `shrink` < 1.0 trims the glass rim.
    """
    r_use = int(r * shrink)
    x1 = max(0, cx - r_use)
    y1 = max(0, cy - r_use)
    x2 = min(img_bgr.shape[1], cx + r_use)
    y2 = min(img_bgr.shape[0], cy + r_use)

    crop  = img_bgr[y1:y2, x1:x2].copy()
    hc, wc = crop.shape[:2]

    cx_l = cx - x1
    cy_l = cy - y1
    mask = np.zeros((hc, wc), dtype=np.uint8)
    cv2.circle(mask, (cx_l, cy_l), r_use, 255, -1)
    crop[mask == 0] = 0

    return crop, mask, x1, y1


def segment_colonies(crop_bgr, plate_mask):
    """
    Hybrid segmentation combining two independent channels:

    1. Top-hat on L (LAB lightness): detects white/cream colonies that are
       brighter than the local agar background regardless of color.
    2. Saturation threshold in HSV: detects colored colonies (pink, orange,
       yellow) that stand out from the low-saturation agar.

    The two binary masks are combined with OR so both colony types are found
    in the same image without parameter conflict.

    Returns (label_image, list_of_valid_regionprops).
    """
    blurred = cv2.GaussianBlur(crop_bgr, (5, 5), 0)

    # ── Channel 1: top-hat on L (luminance) for white/cream colonies ─────────
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    L   = cv2.bitwise_and(lab[:, :, 0], lab[:, :, 0], mask=plate_mask)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    tophat  = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, kernel)
    _, bin_lum = cv2.threshold(tophat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY)

    # ── Channel 2: saturation in HSV for colored colonies ────────────────────
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    sat = cv2.bitwise_and(hsv[:, :, 1], hsv[:, :, 1], mask=plate_mask)
    _, bin_col = cv2.threshold(sat, SAT_THRESHOLD, 255, cv2.THRESH_BINARY)

    # ── Combine: colony if bright OR colored ─────────────────────────────────
    binary = cv2.bitwise_or(bin_lum, bin_col)
    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

    # ── Fill only small holes (colony center dots, NOT agar gaps) ────────────
    inv_binary  = (~binary.astype(bool))
    inv_labeled = sk_label(inv_binary)
    binary_filled = binary.copy()
    for region_id in range(1, inv_labeled.max() + 1):
        hole = inv_labeled == region_id
        if hole.sum() <= MAX_HOLE_AREA:
            binary_filled[hole] = 255

    # ── Open to remove speckle ───────────────────────────────────────────────
    k3            = np.ones((3, 3), np.uint8)
    binary_filled = cv2.morphologyEx(binary_filled, cv2.MORPH_OPEN, k3,
                                     iterations=OPEN_ITERATIONS)
    binary_filled = cv2.bitwise_and(binary_filled, binary_filled, mask=plate_mask)

    # ── Distance transform → one watershed seed per colony ───────────────────
    dist   = ndimage.distance_transform_edt(binary_filled)
    coords = peak_local_max(dist, min_distance=WATERSHED_MIN_DIST,
                            threshold_rel=0.15,
                            labels=binary_filled.astype(bool))

    if len(coords) == 0:
        _, labels_out = cv2.connectedComponents(binary_filled)
        props = regionprops(labels_out)
        valid = [p for p in props
                 if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                 and p.solidity >= MIN_SOLIDITY]
        return labels_out, valid, binary_filled

    local_max = np.zeros_like(dist, dtype=bool)
    local_max[tuple(coords.T)] = True
    markers   = sk_label(local_max)
    labels_ws = watershed(-dist, markers, mask=binary_filled.astype(bool))

    props = regionprops(labels_ws)
    valid = [p for p in props
             if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY]

    return labels_ws, valid, binary_filled


def draw_overlay(crop_bgr, valid_props, labels_ws):
    overlay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).copy().astype(np.float32) / 255.0
    for prop in valid_props:
        mask_col = labels_ws == prop.label
        overlay[mask_col] = overlay[mask_col] * 0.45 + np.array([0.15, 0.85, 0.35]) * 0.55
    return overlay


def process_image(img_path, output_dir):
    """
    Process a single image: detect plates, count colonies, save visualization.
    Returns list of counts, one per plate.
    """
    img_path   = Path(img_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")

    h_img, w_img = img.shape[:2]
    orientation  = "portrait" if h_img > w_img else "landscape"
    print(f"    {w_img}x{h_img} ({orientation})")

    plates      = detect_plates(img)
    plate_names = ["A", "B"]
    counts      = []

    n_plates = len(plates)
    fig, axes = plt.subplots(n_plates, 3, figsize=(15, 6 * n_plates))
    if n_plates == 1:
        axes = axes[np.newaxis, :]

    for i, (cx, cy, r) in enumerate(plates):
        name = plate_names[i] if i < len(plate_names) else str(i + 1)
        crop, mask, _, _ = crop_plate(img, cx, cy, r)
        labels_ws, valid, binary_raw = segment_colonies(crop, mask)
        n                = len(valid)
        counts.append(n)
        print(f"    Plate {name}: {n} colonies")

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        overlay  = draw_overlay(crop, valid, labels_ws)

        axes[i, 0].imshow(crop_rgb)
        axes[i, 0].set_title(f"Plate {name} — original", fontsize=13)
        axes[i, 0].axis("off")

        # Show the raw binary before watershed so we can see what the
        # segmentation step detects before area/solidity filtering.
        axes[i, 1].imshow(binary_raw, cmap="gray")
        axes[i, 1].set_title(f"Plate {name} — binary (pre-filter)", fontsize=13)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Plate {name} — {n} colonies", fontsize=13, fontweight="bold")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = output_dir / f"{img_path.stem}_colony_count.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Reliability check between duplicate plates ────────────────────────────
    if len(counts) == 2 and all(c > 0 for c in counts):
        diff_pct = abs(counts[0] - counts[1]) / max(counts) * 100
        tag = "CONFIABLE" if diff_pct <= 10 else ("ACEPTABLE" if diff_pct <= 20 else "REVISAR")
        print(f"    Diferencia entre placas: {diff_pct:.1f}%  [{tag}]")

    return counts


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img_path   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("images/placa.jpg")
    output_dir = Path("results/colonies")

    print(f"\n[1] Loading: {img_path}")
    counts = process_image(img_path, output_dir)

    print(f"\n[2] Saved: results/colonies/{img_path.stem}_colony_count.png")
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
