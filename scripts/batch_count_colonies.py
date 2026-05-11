"""
Batch colony counter — processes all plate images in a folder.
Usage: python scripts/batch_count_colonies.py <images_folder> [ground_truth.csv]
Example:
    python scripts/batch_count_colonies.py images/placas/
    python scripts/batch_count_colonies.py images/placas/ images/placas/ground_truth.csv

Ground truth CSV format (optional, for error calculation):
    filename,plate_A,plate_B
    actinomicetos1.jpeg,63,64
    actinomicetos2.jpeg,55,58

Results are saved to:
    results/colonies/          — one visualization PNG per image
    results/colonies/summary.csv  — count table for all images
"""

import sys
import csv
from pathlib import Path

# Import processing functions from count_colonies
sys.path.insert(0, str(Path(__file__).parent))
from count_colonies import process_image

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_ground_truth(csv_path):
    """Load ground truth CSV → {filename_stem: (count_A, count_B)}."""
    gt = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = Path(row["filename"]).stem
            gt[stem] = (int(row["plate_A"]), int(row["plate_B"]))
    return gt


def main():
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("images")
    gt_csv     = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    output_dir = Path("results/colonies")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image files
    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXT
    )

    if not images:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) in {images_dir}\n")

    # Load ground truth if provided
    gt = load_ground_truth(gt_csv) if gt_csv and gt_csv.exists() else {}

    # Process all images
    results = []
    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}")
        try:
            counts = process_image(img_path, output_dir)
            a = counts[0] if len(counts) > 0 else None
            b = counts[1] if len(counts) > 1 else None
        except Exception as e:
            print(f"    ERROR: {e}")
            a, b = None, None

        if a is not None and b is not None and a > 0 and b > 0:
            diff_pct = abs(a - b) / max(a, b) * 100
            reliability = "CONFIABLE" if diff_pct <= 10 else ("ACEPTABLE" if diff_pct <= 20 else "REVISAR")
        else:
            diff_pct    = None
            reliability = "—"

        row = {"image": img_path.name, "plate_A": a, "plate_B": b,
               "diff_pct": f"{diff_pct:.1f}%" if diff_pct is not None else "—",
               "reliability": reliability}

        # Compare with ground truth if available
        if img_path.stem in gt:
            gt_a, gt_b = gt[img_path.stem]
            row["gt_A"]  = gt_a
            row["gt_B"]  = gt_b
            row["err_A"] = (a - gt_a) if a is not None else None
            row["err_B"] = (b - gt_b) if b is not None else None
            row["pct_A"] = f"{abs(a-gt_a)/gt_a*100:.1f}%" if a is not None else "—"
            row["pct_B"] = f"{abs(b-gt_b)/gt_b*100:.1f}%" if b is not None else "—"

        results.append(row)
        print()

    # Save CSV summary
    summary_path = output_dir / "summary.csv"
    has_gt       = bool(gt)
    fieldnames   = ["image", "plate_A", "plate_B", "diff_pct", "reliability"]
    if has_gt:
        fieldnames += ["gt_A", "gt_B", "err_A", "err_B", "pct_A", "pct_B"]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Print summary table
    print("=" * 65)
    print("  BATCH SUMMARY")
    print("=" * 65)
    if has_gt:
        print(f"  {'Image':<30} {'A':>5} {'B':>5} {'ErrA':>6} {'ErrB':>6} {'Conf.':>10}")
        print("  " + "-" * 65)
        for r in results:
            ea = f"{r['err_A']:+d}" if r.get("err_A") is not None else "—"
            eb = f"{r['err_B']:+d}" if r.get("err_B") is not None else "—"
            print(f"  {r['image']:<30} {str(r['plate_A']):>5} {str(r['plate_B']):>5} {ea:>6} {eb:>6} {r['reliability']:>10}")
    else:
        print(f"  {'Image':<30} {'A':>7} {'B':>7} {'Dif.%':>7} {'Conf.':>10}")
        print("  " + "-" * 63)
        for r in results:
            print(f"  {r['image']:<30} {str(r['plate_A']):>7} {str(r['plate_B']):>7} {r['diff_pct']:>7} {r['reliability']:>10}")
    print("=" * 65)
    print(f"\n  CSV saved: {summary_path}")
    print(f"  Visualizations: {output_dir}/")


if __name__ == "__main__":
    main()
