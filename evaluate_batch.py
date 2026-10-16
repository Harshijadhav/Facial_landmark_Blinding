"""
Command-line batch evaluation (same code as the Evaluation tab).

Examples:
    python evaluate_batch.py --input data/eval
    python evaluate_batch.py --input data/eval --sweep 8 16 24 30 --mode both
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from src.evaluation import evaluate_images, paired_tests, summarize_conditions, summarize_robustness

MODES = {"rgb": "Colour pattern", "luminance": "Brightness only", "anchors": "Anchor-targeted"}


def load_images(folder, limit):
    items = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            data = np.fromfile(os.path.join(folder, name), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                items.append((name, img))
        if len(items) >= limit:
            break
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description="Batch evaluation of the landmark-blinding defense")
    ap.add_argument("--input", required=True, help="folder with .jpg/.png face images")
    ap.add_argument("--out", default="results", help="folder for the CSV files")
    ap.add_argument("--strength", type=float, default=16, help="perturbation strength in percent")
    ap.add_argument("--sweep", type=float, nargs="+", help="several strengths, e.g. 8 16 24 30")
    ap.add_argument("--jpeg", type=int, default=75, help="JPEG quality")
    ap.add_argument("--max-side", type=int, default=1280, help="longest image side, 0 = full size")
    ap.add_argument("--mode", choices=["rgb", "luminance", "anchors", "both", "all"], default="rgb")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--robustness", action="store_true", help="also run the post-processing tests (slower)")
    args = ap.parse_args(argv)

    from src.detector import FaceDetector
    from src.landmarks import LandmarkDetector
    face_detector, landmark_detector = FaceDetector(), LandmarkDetector()

    items = load_images(args.input, args.limit)
    if not items:
        raise SystemExit(f"No images found in {args.input}")
    print(f"Loaded {len(items)} images from {args.input}")

    strengths = args.sweep if args.sweep else [args.strength]
    modes = {"both": ["rgb", "luminance"], "all": ["rgb", "luminance", "anchors"]}.get(args.mode, [args.mode])
    max_side = args.max_side or None
    os.makedirs(args.out, exist_ok=True)

    sweep_rows = []
    for mode in modes:
        for s in strengths:
            tag = f"{mode}_s{int(round(s))}"
            print(f"\n=== pattern: {MODES[mode]} | strength {s:g}% | JPEG {args.jpeg} ===")

            def progress(i, n, name):
                print(f"\r  image {min(i + 1, n)} of {n}", end="", flush=True)

            per_image, robust, skipped = evaluate_images(
                items, landmark_detector, face_detector, s / 100.0, args.jpeg,
                max_side=max_side, robustness=args.robustness, progress=progress, mode=mode)
            print()
            summary = summarize_conditions(per_image)
            tests = paired_tests(per_image)
            folder = os.path.join(args.out, tag)
            os.makedirs(folder, exist_ok=True)
            per_image.to_csv(os.path.join(folder, "per_image.csv"), index=False)
            summary.to_csv(os.path.join(folder, "summary.csv"), index=False)
            tests.to_csv(os.path.join(folder, "paired_tests.csv"), index=False)
            if not robust.empty:
                summarize_robustness(robust).to_csv(os.path.join(folder, "robustness.csv"), index=False)
            print(summary.to_string(index=False))
            if not tests.empty:
                print(tests.to_string(index=False))
            if skipped:
                print("Skipped (no face found):", ", ".join(skipped))

            if not summary.empty:
                by = summary.set_index("Condition")
                ours = by.loc["Ours"]
                row = {
                    "Pattern": MODES[mode], "Strength %": s, "Images": int(ours["Images"]),
                    "Ours error %": ours["Mean key-point error (% eye dist)"],
                    "JPEG only error %": by.loc["JPEG only", "Mean key-point error (% eye dist)"] if "JPEG only" in by.index else np.nan,
                    "Noise error %": by.loc["Random noise", "Mean key-point error (% eye dist)"] if "Random noise" in by.index else np.nan,
                    "Ours PSNR dB": ours["Mean PSNR (dB)"], "Ours SSIM": ours["Mean SSIM"],
                    "Ours disrupted %": ours["Disrupted (%)"],
                }
                if not tests.empty:
                    t = tests[tests["Comparison"] == "Ours vs Random noise"]
                    row["p vs noise"] = float(t["p-value (Wilcoxon)"].iloc[0]) if len(t) else np.nan
                sweep_rows.append(row)

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(args.out, "sweep_summary.csv"), index=False)
    print("\n=== Summary of all runs ===")
    print(sweep.to_string(index=False))
    print(f"\nCSV files saved in: {args.out}")


if __name__ == "__main__":
    main()
