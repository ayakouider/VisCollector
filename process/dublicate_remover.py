import argparse
import json
import os
import re
import shutil
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
import imagehash
import torch
import open_clip


DEFAULT_FRAMES_DIR = "./dataset/frames"
DEFAULT_REPORTS_DIR = "./dataset/reports"
DEFAULT_PHASH_THRESHOLD = 8
DEFAULT_CLIP_THRESHOLD  = 0.97
DEFAULT_METHOD = "phash"
DUPLICATES_SUBDIR = "duplicates"

def clean_id(path: str) -> str:
    basename = os.path.basename(path.rstrip("/\\"))
    clean    = re.sub(r"[^\w\s-]", "", basename)
    clean    = re.sub(r"[\s]+", "_", clean.strip())
    return clean

def get_image_files(frames_dir: str) -> list:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return [
        os.path.join(frames_dir, f)
        for f in sorted(os.listdir(frames_dir))
        if os.path.isfile(os.path.join(frames_dir, f))          # skip subdirectories
        and os.path.splitext(f)[1].lower() in extensions
    ]

def calculate_phash(image_path: str):
    try:
        with Image.open(image_path).convert("RGB") as img:
            return imagehash.phash(img)
    except Exception as e:
        print(f"[!] Error calculating pHash for {image_path}: {e}")
        return None

def calculate_duplicate_phash(image_files: list, threshold: int):
    print(f"[→] Computing perceptual hashes for {len(image_files)} frames...")

    hashes = {}
    for path in tqdm(image_files, desc="Computing pHash", unit="frame"):
        try:
            hashes[path] = calculate_phash(path)
        except Exception as e:
            print(f"[!] Could not hash {os.path.basename(path)}: {e}")

    print(f"[✓] Hashes computed. Comparing pairs...\n")

    to_remove  = set()
    pairs      = []
    paths_list = list(hashes.keys())

    for i in tqdm(range(len(paths_list)), desc="Finding duplicates", unit="frame"):
        if paths_list[i] in to_remove:
            continue
        for j in range(i + 1, len(paths_list)):
            if paths_list[j] in to_remove:
                continue

            dist = hashes[paths_list[i]] - hashes[paths_list[j]]
            if dist <= threshold:
                to_remove.add(paths_list[j])
                pairs.append({
                    'accepted': os.path.basename(paths_list[i]),
                    'rejected': os.path.basename(paths_list[j]),
                    'distance': int(dist)
                })
    
    to_keep = [p for p in paths_list if p not in to_remove]
    return{
        'keep': to_keep,
        'remove': list(to_remove),
        'pairs': pairs
    }

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[→] Loading CLIP model (device: {device})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.to(device)
    print(f"[✓] CLIP model loaded.\n")
    return model, preprocess, device

def calculate_clip_embeddings(image_files:list, model, preprocess, device):
    embeddings= {}
    with torch.no_grad():
        for path in tqdm(image_files, desc="Computing CLIP embeddings", unit="frame"):
            try:
                img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                emb = model.encode_image(img)
                emb = emb / emb.norm(dim=-1, keepdim=True)  # Normalize
                embeddings[path] = emb.cpu().numpy().squeeze()
            except Exception as e:
                print(f"[!] Could not embed {os.path.basename(path)}: {e}")
    return embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) :
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def find_duplicates_clip(image_files:list,threshold:float):
    model, preprocess, device = load_clip_model()
    embeddings = calculate_clip_embeddings(image_files, model, preprocess, device)

    print(f"[✓] Embeddings computed. Comparing pairs...\n")

    to_remove  = set()
    pairs      = []
    paths_list = list(embeddings.keys())

    for i in tqdm(range(len(paths_list)), desc="Finding duplicates", unit="frame"):
        if paths_list[i] in to_remove:
            continue
        for j in range(i + 1, len(paths_list)):
            if paths_list[j] in to_remove:
                continue
            sim = cosine_similarity(embeddings[paths_list[i]], embeddings[paths_list[j]])
            if sim >= threshold:
                to_remove.add(paths_list[j])
                pairs.append({
                    "kept"      : os.path.basename(paths_list[i]),
                    "removed"   : os.path.basename(paths_list[j]),
                    "similarity": round(sim, 4),
                })

    to_keep = [p for p in paths_list if p not in to_remove]

    return {
        "keep"  : to_keep,
        "remove": list(to_remove),
        "pairs" : pairs,
    }

def remove_duplicates(
    frames_dir: str,
    method: str = DEFAULT_METHOD,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
    clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
    delete: bool = False,
    reports_dir: str = DEFAULT_REPORTS_DIR,
):
    image_files = get_image_files(frames_dir)
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {frames_dir}")
    
    video_id = clean_id(frames_dir)
    print(f"\n[→] Running duplicate detection on {len(image_files)} frames")
    print(f"[✓] Method     : {method.upper()}")

    if method == "phash":
        print(f"[✓] Threshold  : Hamming distance ≤ {phash_threshold}")
    else:
        print(f"[✓] Threshold  : Cosine similarity ≥ {clip_threshold}")
    print(f"[✓] Mode       : {'DELETE (move to duplicates/)' if delete else 'REPORT ONLY'}\n")

    if method == "phash":
        result = calculate_duplicate_phash(image_files, phash_threshold)
    elif method == "clip":
        result = find_duplicates_clip(image_files, clip_threshold)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'phash' or 'clip'.")
    
    to_keep   = result["keep"]
    to_remove = result["remove"]
    pairs     = result["pairs"]

    moved_count = 0
    if delete and to_remove:
        dup_dir= os.path.join(frames_dir, DUPLICATES_SUBDIR)
        os.makedirs(dup_dir, exist_ok=True)
        print(f"\n[→] Moving {len(to_remove)} duplicates to: {dup_dir}")
        for path in tqdm(to_remove, desc="Moving duplicates", unit="frame"):
            dest = os.path.join(dup_dir, os.path.basename(path))
            shutil.move(path, dest)
            moved_count += 1
        print(f"[✓] Moved {moved_count} duplicates.")

    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{video_id}_dedup_report.json")

    report = {
        "video_id"           : video_id,
        "frames_dir"         : os.path.abspath(frames_dir),
        "method"             : method,
        "threshold"          : phash_threshold if method == "phash" else clip_threshold,
        "total_frames"       : len(image_files),
        "unique_frames"      : len(to_keep),
        "duplicate_frames"   : len(to_remove),
        "duplicate_rate_pct" : round(len(to_remove) / len(image_files) * 100, 1) if image_files else 0,
        "frames_moved"       : moved_count,
        "duplicate_pairs"    : pairs[:100],  # Cap at 100 pairs in report
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report, report_path

def main():
    parser = argparse.ArgumentParser(
        description="Duplicate Remover — Vision Data Agent (Phase 4)"
    )
    parser.add_argument("--frames_dir", type=str, required=True,
        help="Directory of extracted frames (e.g. ./dataset/frames/my_video)")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD,
        choices=["phash", "clip"],
        help=f"Detection method: 'phash' (fast) or 'clip' (semantic). Default: {DEFAULT_METHOD}")
    parser.add_argument("--threshold", type=float, default=None,
        help=f"Detection threshold. pHash: Hamming distance (default {DEFAULT_PHASH_THRESHOLD}, lower=stricter). "
             f"CLIP: cosine similarity (default {DEFAULT_CLIP_THRESHOLD}, higher=stricter).")
    parser.add_argument("--delete", action="store_true",
        help="Move duplicates to a duplicates/ subfolder. Without this, only a report is generated.")
    parser.add_argument("--reports_dir", type=str, default=DEFAULT_REPORTS_DIR,
        help=f"Directory to save dedup report (default: {DEFAULT_REPORTS_DIR})")

    args = parser.parse_args()

    if not os.path.isdir(args.frames_dir):
        print(f"[✗] Frames directory not found: {args.frames_dir}")
        sys.exit(1)

    # Set threshold defaults based on method
    if args.threshold is None:
        threshold = DEFAULT_PHASH_THRESHOLD if args.method == "phash" else DEFAULT_CLIP_THRESHOLD
    else:
        threshold = args.threshold

    phash_threshold = int(threshold)   if args.method == "phash" else DEFAULT_PHASH_THRESHOLD
    clip_threshold  = float(threshold) if args.method == "clip"  else DEFAULT_CLIP_THRESHOLD

    try:
        report, report_path = remove_duplicates(
            frames_dir       = args.frames_dir,
            method           = args.method,
            phash_threshold  = phash_threshold,
            clip_threshold   = clip_threshold,
            delete           = args.delete,
            reports_dir      = args.reports_dir,
        )

        print(f"\n{'─' * 50}")
        print(f"  Total frames       : {report['total_frames']}")
        print(f"  Unique frames kept : {report['unique_frames']}")
        print(f"  Duplicates found   : {report['duplicate_frames']} ({report['duplicate_rate_pct']}%)")
        print(f"  Report saved to    : {report_path}")
        print(f"{'─' * 50}")

        if not args.delete:
            print(f"\n[!] Dry run — no files were changed.")
            print(f"[→] Re-run with --delete to move duplicates.\n")

    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as e:
        print(f"\n[✗] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()