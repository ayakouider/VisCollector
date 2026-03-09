import argparse
import json
import os
import re
import shutil
import sys
import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_BLUR_THRESHOLD   = 100.0
DEFAULT_DARK_THRESHOLD   = 30 
DEFAULT_BRIGHT_THRESHOLD = 225
DEFAULT_REPORTS_DIR      = "./dataset/reports"
BLURRY_SUBDIR            = "blurry"


def calculate_laplacian_variance(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def analyze_frame(image, blur_threshold, dark_threshold, bright_threshold):
    blur_score = calculate_laplacian_variance(image)
    brightness = mean_brightness(image)

    is_blurry  = blur_score < blur_threshold
    is_dark    = brightness < dark_threshold
    is_bright  = brightness > bright_threshold
    is_rejected = is_blurry or is_dark or is_bright

    reasons = []
    if is_blurry: reasons.append(f"blurry (score={blur_score:.1f} < {blur_threshold})")
    if is_dark:   reasons.append(f"dark (brightness={brightness:.1f})")
    if is_bright: reasons.append(f"bright (brightness={brightness:.1f})")

    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "is_blurry":  is_blurry,
        "is_dark":    is_dark,
        "is_bright":  is_bright,
        "is_rejected": is_rejected,
        "reasons":    reasons,
    }


def clean_id(path: str):
    basename = os.path.basename(path.rstrip("/\\"))
    clean    = re.sub(r"[^\w\s-]", "", basename)
    clean    = re.sub(r"[\s]+", "_", clean.strip())
    return clean


def get_image_files(frames_dir: str) -> list:
    """Returns a plain list of file path STRINGS — not dicts."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return [
        os.path.join(frames_dir, f)
        for f in sorted(os.listdir(frames_dir))
        if os.path.isfile(os.path.join(frames_dir, f))
        and os.path.splitext(f)[1].lower() in extensions
    ]


def suggest_threshold(scores: list):
    return float(np.percentile(scores, 15))


def blur_filter(
    frames_dir: str,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    dark_threshold: int   = DEFAULT_DARK_THRESHOLD,
    bright_threshold: int = DEFAULT_BRIGHT_THRESHOLD,
    delete: bool          = False,
    reports_dir: str      = DEFAULT_REPORTS_DIR,
    preview: bool         = False,
):
    image_files = get_image_files(frames_dir)  # list of strings
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {frames_dir}")

    video_id = clean_id(frames_dir)

    print(f"\n[->] Assessing {len(image_files)} frames")
    print(f"[OK] Blur threshold   : {blur_threshold} (lower = stricter)")
    print(f"[OK] Dark threshold   : mean brightness < {dark_threshold}")
    print(f"[OK] Bright threshold : mean brightness > {bright_threshold}")
    print(f"[OK] Mode             : {'DELETE (move to blurry/)' if delete else 'REPORT ONLY'}\n")

    results     = []
    blur_scores = []
    rejected    = []   # list of dicts {path, blur_score, ...}
    accepted    = []   # list of dicts

    with tqdm(image_files, desc="Assessing frames", unit="frame") as pbar:
        for path in pbar:                        # path is a plain STRING
            image = cv2.imread(path)
            if image is None:
                continue

            analysis = analyze_frame(image, blur_threshold, dark_threshold, bright_threshold)
            analysis["path"]     = path                      # store path inside dict
            analysis["filename"] = os.path.basename(path)
            results.append(analysis)
            blur_scores.append(analysis["blur_score"])

            if analysis["is_rejected"]:
                rejected.append(analysis)        # dict goes into rejected list
            else:
                accepted.append(analysis)

            pbar.set_postfix(
                rejected=len(rejected),
                blur=f"{analysis['blur_score']:.0f}"
            )

    suggested_threshold = suggest_threshold(blur_scores)

    # Preview mode
    if preview and rejected:
        print(f"\n[->] Preview mode: press any key to advance, 'q' to quit")
        for item in rejected[:20]:               # item is a DICT
            frame_path = item["path"]            # extract string path from dict
            image = cv2.imread(frame_path)
            label = f"REJECTED | blur={item['blur_score']:.1f} | {item['filename']}"
            cv2.putText(image, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Rejected Frame", image)
            if cv2.waitKey(0) == ord('q'):
                break
        cv2.destroyAllWindows()

    # Move rejected frames
    moved_count = 0
    if delete and rejected:
        blurry_dir = os.path.join(frames_dir, BLURRY_SUBDIR)
        os.makedirs(blurry_dir, exist_ok=True)
        print(f"\n[->] Moving {len(rejected)} rejected frames to: {blurry_dir}")
        for item in tqdm(rejected, desc="Moving rejected frames", unit="frame"):
            frame_path = item["path"]            # extract string path from dict
            dest = os.path.join(blurry_dir, os.path.basename(frame_path))
            shutil.move(frame_path, dest)
            moved_count += 1
        print(f"[OK] Moved {moved_count} frames.")

    # Save report
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{video_id}_quality_report.json")

    report = {
        "video_id"           : video_id,
        "frames_dir"         : os.path.abspath(frames_dir),
        "total_frames"       : len(results),
        "kept_frames"        : len(accepted),
        "rejected_frames"    : len(rejected),
        "rejection_rate_pct" : round(len(rejected) / len(results) * 100, 1) if results else 0,
        "blur_threshold_used": blur_threshold,
        "suggested_threshold": round(suggested_threshold, 2),
        "blur_score_min"     : round(min(blur_scores), 2) if blur_scores else 0,
        "blur_score_max"     : round(max(blur_scores), 2) if blur_scores else 0,
        "blur_score_mean"    : round(float(np.mean(blur_scores)), 2) if blur_scores else 0,
        "frames_moved"       : moved_count,
        "frame_details"      : results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report, report_path


def main():
    parser = argparse.ArgumentParser(description="Blur Filter - Vision Data Agent (Phase 4)")
    parser.add_argument("--frames_dir",      type=str,   required=True)
    parser.add_argument("--threshold",       type=float, default=DEFAULT_BLUR_THRESHOLD)
    parser.add_argument("--dark_threshold",  type=int,   default=DEFAULT_DARK_THRESHOLD)
    parser.add_argument("--bright_threshold",type=int,   default=DEFAULT_BRIGHT_THRESHOLD)
    parser.add_argument("--delete",          action="store_true")
    parser.add_argument("--preview",         action="store_true")
    parser.add_argument("--reports_dir",     type=str,   default=DEFAULT_REPORTS_DIR)

    args = parser.parse_args()

    if not os.path.isdir(args.frames_dir):
        print(f"[X] Frames directory not found: {args.frames_dir}")
        sys.exit(1)

    try:
        report, report_path = blur_filter(
            frames_dir       = args.frames_dir,
            blur_threshold   = args.threshold,
            dark_threshold   = args.dark_threshold,
            bright_threshold = args.bright_threshold,
            delete           = args.delete,
            reports_dir      = args.reports_dir,
            preview          = args.preview,
        )

        print(f"\n{'─' * 50}")
        print(f"  Total frames       : {report['total_frames']}")
        print(f"  Kept (sharp)       : {report['kept_frames']}")
        print(f"  Rejected           : {report['rejected_frames']} ({report['rejection_rate_pct']}%)")
        print(f"  Blur score range   : {report['blur_score_min']} - {report['blur_score_max']}")
        print(f"  Blur score mean    : {report['blur_score_mean']}")
        print(f"  Suggested threshold: {report['suggested_threshold']}")
        print(f"  Report saved to    : {report_path}")
        print(f"{'─' * 50}")

        if not args.delete:
            print(f"\n[!] Dry run - no files were changed.")
            print(f"[->] Re-run with --delete to move rejected frames.")
            print(f"[->] Re-run with --threshold {report['suggested_threshold']:.0f} for suggested threshold.\n")

    except Exception as e:
        print(f"\n[X] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()