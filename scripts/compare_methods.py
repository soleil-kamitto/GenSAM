"""
Compare CellSAM vs classical CV colony counting on all plate images.

Usage:
    python scripts/compare_methods.py images/placas/

Outputs (in results/colonies/):
    comparison_chart.png       — bar chart: counts by method across all images
    <image>_comparison.png     — per-image side-by-side masks for both methods
    comparison_summary.csv     — raw counts table
"""

import sys
import csv
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

from cellSAM import get_model, segment_cellular_image


# ── SHARED FILTERING PARAMETERS ───────────────────────────────────────────────
MIN_COLONY_AREA    = 300
MAX_COLONY_AREA    = 50000
MIN_SOLIDITY       = 0.50

# ── CLASSICAL CV PARAMETERS ───────────────────────────────────────────────────
TOPHAT_KERNEL      = 81    # px — must be larger than biggest colony
TOPHAT_THRESHOLD   = 10    # top-hat intensity cutoff
SAT_THRESHOLD      = 60    # HSV saturation cutoff for colored colonies
MAX_HOLE_AREA      = 400   # px² — max hole to fill (colony center dots)
OPEN_ITERATIONS    = 1     # morphological open iterations
WATERSHED_MIN_DIST = 20    # px — min distance between colony seeds

# ── CELLSAM PARAMETERS ────────────────────────────────────────────────────────
NORMALIZE          = True  # percentile normalisation + CLAHE
POSTPROCESS        = False

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

_model = None


def _get_model():
    global _model
    if _model is None:
        print("  Loading CellSAM model (first time only)...")
        _model = get_model()
        print("  Model ready.")
    return _model


# ── PLATE DETECTION & CROP (shared) ──────────────────────────────────────────

def detect_plates(img_bgr):
    h, w    = img_bgr.shape[:2]
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
            cx = hw // 2 + ox
            cy = hh // 2 + oy
            r  = int(min(hh, hw) * 0.43)

        plates.append((cx, cy, r))

    return plates


def crop_plate(img_bgr, cx, cy, r, shrink=0.86):
    r_use = int(r * shrink)
    x1 = max(0, cx - r_use);  y1 = max(0, cy - r_use)
    x2 = min(img_bgr.shape[1], cx + r_use)
    y2 = min(img_bgr.shape[0], cy + r_use)

    crop    = img_bgr[y1:y2, x1:x2].copy()
    hc, wc  = crop.shape[:2]
    mask    = np.zeros((hc, wc), dtype=np.uint8)
    cv2.circle(mask, (cx - x1, cy - y1), r_use, 255, -1)
    crop[mask == 0] = 0
    return crop, mask


# ── CLASSICAL SEGMENTATION ───────────────────────────────────────────────────

def segment_classical(crop_bgr, plate_mask):
    """
    Full classical CV pipeline:
      1. Gaussian blur 5x5
      2. White top-hat on grayscale (kernel 81px) — bright colonies on dark agar
      3. Fixed threshold on top-hat output (TOPHAT_THRESHOLD=10)
      4. HSV saturation threshold (SAT_THRESHOLD=60) — colored colonies
      5. OR of both binary masks
      6. Fill small holes <=400px2 (colony center dots)
      7. Morphological open — remove speckle
      8. Watershed — separate touching colonies
    Returns (label_image, valid_regionprops, binary_mask).
    """
    # 1+2. Blur → top-hat on grayscale
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray    = cv2.bitwise_and(gray, gray, mask=plate_mask)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (TOPHAT_KERNEL, TOPHAT_KERNEL))
    tophat  = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

    # 3. Threshold on top-hat
    _, bin_lum = cv2.threshold(tophat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    bin_lum    = cv2.bitwise_and(bin_lum, bin_lum, mask=plate_mask)

    # 4. HSV saturation for colored colonies
    hsv        = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat        = cv2.bitwise_and(hsv[:, :, 1], hsv[:, :, 1], mask=plate_mask)
    _, bin_col = cv2.threshold(sat, SAT_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 5. OR combination
    binary = cv2.bitwise_or(bin_lum, bin_col)
    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

    # 6. Fill small holes
    inv_labeled   = sk_label(~binary.astype(bool))
    binary_filled = binary.copy()
    for rid in range(1, inv_labeled.max() + 1):
        hole = inv_labeled == rid
        if hole.sum() <= MAX_HOLE_AREA:
            binary_filled[hole] = 255

    # 7. Morphological open
    k3            = np.ones((3, 3), np.uint8)
    binary_filled = cv2.morphologyEx(binary_filled, cv2.MORPH_OPEN, k3,
                                     iterations=OPEN_ITERATIONS)
    binary_filled = cv2.bitwise_and(binary_filled, binary_filled, mask=plate_mask)

    # 8. Watershed
    dist   = ndimage.distance_transform_edt(binary_filled)
    coords = peak_local_max(dist, min_distance=WATERSHED_MIN_DIST,
                            threshold_rel=0.15, labels=binary_filled.astype(bool))

    if len(coords) == 0:
        _, labels_out = cv2.connectedComponents(binary_filled)
        props = regionprops(labels_out)
        valid = [p for p in props if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
                 and p.solidity >= MIN_SOLIDITY]
        return labels_out, valid, binary_filled

    local_max              = np.zeros_like(dist, dtype=bool)
    local_max[tuple(coords.T)] = True
    markers                = sk_label(local_max)
    labels_ws              = watershed(-dist, markers, mask=binary_filled.astype(bool))

    props = regionprops(labels_ws)
    valid = [p for p in props if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
             and p.solidity >= MIN_SOLIDITY]
    return labels_ws, valid, binary_filled


# ── CELLSAM SEGMENTATION ─────────────────────────────────────────────────────

def segment_cellsam(crop_bgr, plate_mask):
    """
    CellSAM segmentation with area/solidity filtering.
    Returns (label_mask, valid_regionprops, binary_mask).
    """
    model    = _get_model()
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    mask, _, _ = segment_cellular_image(
        crop_rgb, model=model,
        normalize=NORMALIZE, postprocess=POSTPROCESS, device="cpu",
    )
    mask = mask.copy()
    mask[plate_mask == 0] = 0

    binary = (mask > 0).astype(np.uint8) * 255
    props  = regionprops(mask)
    valid  = [p for p in props if MIN_COLONY_AREA <= p.area <= MAX_COLONY_AREA
              and p.solidity >= MIN_SOLIDITY]
    return mask, valid, binary


# ── OVERLAY HELPER ────────────────────────────────────────────────────────────

def draw_overlay(crop_bgr, valid_props, label_mask):
    overlay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for prop in valid_props:
        region = label_mask == prop.label
        overlay[region] = overlay[region] * 0.45 + np.array([0.15, 0.85, 0.35]) * 0.55
    return overlay


# ── PER-IMAGE COMPARISON ──────────────────────────────────────────────────────

def process_image(img_path, output_dir):
    """
    Run both methods on one image and save a side-by-side comparison PNG.
    Returns {'classical': [nA, nB], 'cellsam': [nA, nB]}.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open: {img_path}")

    plates      = detect_plates(img)
    plate_names = ["A", "B"]

    counts = {"classical": [], "cellsam": []}

    # 2 plates × 3 cols (original | classical | cellsam)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(img_path.name, fontsize=14, fontweight="bold")

    for i, (cx, cy, r) in enumerate(plates):
        name = plate_names[i]
        crop, mask = crop_plate(img, cx, cy, r)

        lbl_cl, val_cl, bin_cl = segment_classical(crop, mask)
        lbl_cs, val_cs, bin_cs = segment_cellsam(crop, mask)

        n_cl = len(val_cl)
        n_cs = len(val_cs)
        counts["classical"].append(n_cl)
        counts["cellsam"].append(n_cs)

        print(f"  Plate {name}  classical={n_cl}  cellsam={n_cs}")

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(crop_rgb)
        axes[i, 0].set_title(f"Plate {name} — original", fontsize=12)
        axes[i, 0].axis("off")

        ov_cl = draw_overlay(crop, val_cl, lbl_cl)
        axes[i, 1].imshow(ov_cl)
        axes[i, 1].set_title(f"Classical CV — {n_cl} colonias", fontsize=12)
        axes[i, 1].axis("off")

        ov_cs = draw_overlay(crop, val_cs, lbl_cs)
        axes[i, 2].imshow(ov_cs)
        axes[i, 2].set_title(f"CellSAM — {n_cs} colonias", fontsize=12)
        axes[i, 2].axis("off")

    plt.tight_layout()
    out = output_dir / f"{img_path.stem}_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()

    return counts


# ── SUMMARY CHART ─────────────────────────────────────────────────────────────

def make_chart(results, output_dir):
    """
    Bar chart: classical vs CellSAM counts for every image and plate.
    """
    images   = [r["name"] for r in results]
    cl_A     = [r["classical"][0] for r in results]
    cl_B     = [r["classical"][1] for r in results]
    cs_A     = [r["cellsam"][0]   for r in results]
    cs_B     = [r["cellsam"][1]   for r in results]

    x     = np.arange(len(images))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Comparativa: Classical CV vs CellSAM", fontsize=14, fontweight="bold")

    for ax, cl_vals, cs_vals, plate in [
        (ax1, cl_A, cs_A, "Placa A"),
        (ax2, cl_B, cs_B, "Placa B"),
    ]:
        bars1 = ax.bar(x - width / 2, cl_vals, width, label="Classical CV",
                       color="#4C72B0", alpha=0.85)
        bars2 = ax.bar(x + width / 2, cs_vals, width, label="CellSAM",
                       color="#DD8452", alpha=0.85)

        ax.bar_label(bars1, padding=3, fontsize=9)
        ax.bar_label(bars2, padding=3, fontsize=9)
        ax.set_ylabel("Colonias contadas")
        ax.set_title(plate)
        ax.legend()
        ax.set_ylim(0, max(max(cl_vals + cs_vals, default=1) * 1.2, 10))
        ax.grid(axis="y", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels([img.replace("actinomicetos_", "actin_") for img in images],
                         rotation=30, ha="right", fontsize=9)

    plt.tight_layout()
    out = output_dir / "comparison_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("images/placas")
    output_dir = Path("results/colonies")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in SUPPORTED_EXT)
    if not images:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) — running both methods\n")

    results = []
    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}")
        try:
            counts = process_image(img_path, output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            counts = {"classical": [None, None], "cellsam": [None, None]}
        results.append({"name": img_path.stem, **counts})
        print()

    # Summary CSV
    csv_path = output_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "classical_A", "classical_B", "cellsam_A", "cellsam_B"])
        for r in results:
            w.writerow([r["name"],
                        r["classical"][0], r["classical"][1],
                        r["cellsam"][0],   r["cellsam"][1]])
    print(f"  CSV saved: {csv_path}")

    # Chart
    make_chart(results, output_dir)

    # Console table
    print("\n" + "=" * 65)
    print(f"  {'Image':<22} {'Cl-A':>6} {'Cl-B':>6} {'CS-A':>6} {'CS-B':>6}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['name']:<22} {str(r['classical'][0]):>6} {str(r['classical'][1]):>6}"
              f" {str(r['cellsam'][0]):>6} {str(r['cellsam'][1]):>6}")
    print("=" * 65)


if __name__ == "__main__":
    main()
