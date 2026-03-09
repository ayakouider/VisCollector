from __future__ import annotations
import argparse
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agent.discover import discover
from agent.data_manager import (
    register_video, update_video, is_processed,
    print_stats, Status
)

DEFAULT_DATASET_DIR = "./dataset"
DEFAULT_MEDIA_DIR = "./dataset/media"
DEFAULT_FRAMES_DIR = "./dataset/frames"
DEFAULT_FACES_DIR = "./dataset/faces"
DEFAULT_AUDIO_DIR  = "./dataset/audio"
DEFAULT_TRANSCRIPTS_DIR = "./dataset/transcripts"
FRAME_INTERVAL_S = 2    
BLUR_THRESHOLD = 100.0
PHASH_THRESHOLD = 8

def step_download(video_id:str, url:str,dataset_dir:str):
    from process.yt_collector import download_video

    print(f"\n[agent] ▶ STEP 1/5  Download  →  {url}")
    try:
        video_path=download_video(url,output_dir=DEFAULT_MEDIA_DIR)
        update_video(video_id, {
            "status": Status.Downloaded,
            "video_path": video_path
        }, dataset_dir)
        print(f"[agent] ✓ Downloaded: {video_path}")
        return True, video_path
    except Exception as e:
        update_video(video_id,{
            "status": Status.Failed,
            "error": str(e)
        }, dataset_dir)
        print(f"[agent] ✗ Download failed: {e}")
        return False, None
    
def step_extract_frames(video_id:str, video_path:str, dataset_dir:str):
    from process.frame_extractor import extract_frames,clean_vid_name,build_output_dir

    print(f"\n[agent] ▶ STEP 2/5  Extract Frames  →  {os.path.basename(video_path)}")
    try:
        # FIX: Don't overwrite video_id! Use a different variable name
        vid_name = clean_vid_name(video_path)  # ← for folder naming only
        frames_dir = build_output_dir(DEFAULT_FRAMES_DIR, video_id=vid_name)
        
        saved = extract_frames(video_path, output_dir=frames_dir, interval_s=FRAME_INTERVAL_S)
        
        # Use the ORIGINAL video_id (URL hash) for manifest lookups
        update_video(video_id, {
            "frames_dir": frames_dir,
            "frame_count": len(saved)
        }, dataset_dir)
        
        print(f"[agent] ✓ Extracted {len(saved)} frames → {frames_dir}")
        return True, frames_dir
    except Exception as e:
        update_video(video_id, {
            "status": Status.Failed,
            "error": str(e)
        }, dataset_dir)
        print(f"[agent] ✗ Frame extraction failed: {e}")
        return False, None
        
def step_extract_faces(video_id:str,frames_dir:str,dataset_dir:str):
    from process.face_extractor import extract_faces
        
    print(f"\n[agent] ▶ STEP 3/5  Extract Faces  →  {frames_dir}")
    try:
        saved_paths,frames_with_faces,total = extract_faces(frames_dir, output_dir=DEFAULT_FACES_DIR)
        faces_dir=os.path.join(DEFAULT_FACES_DIR, os.path.basename(frames_dir))
        update_video(video_id,{
            "faces_dir": faces_dir,
            "face_count": len(saved_paths)
            }, dataset_dir)
        print(f"[agent] ✓ Extracted {len(saved_paths)} faces → {faces_dir}")
        return True, faces_dir
    except Exception as e:
        update_video(video_id,{
            "status": Status.Failed,
            "error": str(e)
            }, dataset_dir)
        print(f"[agent] ✗ Face extraction failed: {e}")
        return False, None
        
def step_blur(video_id:str,frames_dir:str, dataset_dir:str):
    from process.blur import blur_filter

    print(f"\n[agent] ▶ STEP 4/5  Blur Filter  →  {frames_dir}")
    try:
        report,_= blur_filter(frames_dir, blur_threshold=BLUR_THRESHOLD, delete=True)
        kept= report["kept_frames"]
        update_video(video_id,{
            "kept_frames": kept
            }, dataset_dir)
        print(f"[agent] ✓ Blur filter: kept {kept}, removed {report['rejected_frames']}")
        return True, kept
    except Exception as e:
        print(f"[agent] ✗ Blur filter failed: {e}")
        return False, 0

def step_dupre_filter(video_id:str,frames_dir:str,dataset_dir:str):
    from process.dublicate_remover import remove_duplicates

    print(f"\n[agent] ▶ STEP 5/5  Dedup Filter  →  {frames_dir}")
    try:
        report,_= remove_duplicates(frames_dir,method="phash",phash_threshold=PHASH_THRESHOLD,delete=True)
        unique= report["unique_frames"]
        update_video(video_id,{
            "status": Status.Processed,
            "kept_frames": unique
        }, dataset_dir)
        print(f"[agent] ✓ Dedup: kept {unique}, removed {report['duplicate_frames']}")
        return True, unique
    except Exception as e:
        print(f"[agent] ✗ Dedup filter failed: {e}")
        return False, 0

def process_video(
        url:str,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        skip_blur: bool  = False,
        skip_dedup: bool = False,
        dry_run: bool    = False):
    
    if is_processed(url,dataset_dir):
        return {"url": url, "status": Status.Skipped}
    
    video_id, _ = register_video(url, dataset_dir)
    result={"url":url, "video_id": video_id, "status": Status.Failed}

    if dry_run:
        print(f"[agent] DRY RUN — would process: {url}")
        update_video(video_id, {"status": Status.Skipped}, dataset_dir)
        result["status"] = Status.Skipped
        return result
    
    ok, video_path = step_download(video_id, url, dataset_dir)
    if not ok:
        return result
    
    ok,frames_dir = step_extract_frames(video_id, video_path, dataset_dir)
    if not ok:
        return result
    
    ok,faces_dir = step_extract_faces(video_id, frames_dir, dataset_dir)
    if not ok:
        return result
    
    update_video(video_id,{"status": Status.Processed}, dataset_dir)

    if not skip_blur:
        step_blur(video_id, frames_dir, dataset_dir)

    if not skip_dedup:
        step_dupre_filter(video_id, frames_dir, dataset_dir)

    update_video(video_id,{"status": Status.Complete}, dataset_dir)
    result["status"] = Status.Complete
    result.update({
        "video_path": video_path,
        "frames_dir": frames_dir,
        "faces_dir" : faces_dir
    })

    print(f"\n[agent] ✓ COMPLETE: {url}")
    return result

def run_agent(
    urls: list[str] | None = None,
    query: str | None = None,
    channel_url: str | None = None,
    max_results: int = 5,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    skip_blur: bool = False,
    skip_dedup: bool = False,
    dry_run: bool = False,
    # NEW: Smart search parameters for short videos
    smart_search: bool = True,
    min_duration: int = 15,        # 15 seconds minimum
    max_duration: int = 180,       # 3 minutes maximum
    sort_by: str = "relevance",
):
    print("\n" + "═" * 55)
    print("  VISION DATA AGENT — Starting")
    print("═" * 55)
    
    # Pass smart search parameters to discover
    discovery_urls = discover(
        urls=urls, 
        query=query, 
        channel_url=channel_url, 
        max_results=max_results,
        # NEW: Pass duration filters
        smart_search=smart_search,
        min_duration=min_duration,
        max_duration=max_duration,
        sort_by=sort_by,
    )
    
    if not discovery_urls:
        print("[agent] No URLs found. Exiting.")
        return []
    
    print(f"[agent] {len(discovery_urls)} URL(s) to process.\n")
    
    results = []
    for i, url in enumerate(discovery_urls, 1):
        print(f"\n{'─' * 55}")
        print(f"[agent] Video {i}/{len(discovery_urls)}: {url}")
        print(f"{'─' * 55}")
        
        try:
            result = process_video(
                url=url,
                dataset_dir=dataset_dir,
                skip_blur=skip_blur,
                skip_dedup=skip_dedup,
                dry_run=dry_run
            )
            results.append(result)
        except Exception as e:
            print(f"[agent] ✗ Unexpected error on {url}:\n{traceback.format_exc()}")
            results.append({"url": url, "status": Status.Failed, "error": str(e)})
    
    print("\n" + "═" * 55)
    print("  AGENT RUN COMPLETE")
    print("═" * 55)
    
    completed = sum(1 for r in results if r["status"] == Status.Complete)
    failed    = sum(1 for r in results if r["status"] == Status.Failed)
    skipped   = sum(1 for r in results if r["status"] == Status.Skipped)
    
    print(f"  Processed : {len(discovery_urls)}")
    print(f"  Complete  : {completed}")
    print(f"  Failed    : {failed}")
    print(f"  Skipped   : {skipped}")
    
    print_stats(dataset_dir)
    return results
def main():
    parser = argparse.ArgumentParser(
        description="Vision Data Agent — Automated dataset builder"
    )

    # Input sources (one required)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--urls",    nargs="+", type=str,
        help="One or more YouTube URLs to process")
    source.add_argument("--query",   type=str,
        help="YouTube search query (e.g. 'misinformation clips 2024')")
    source.add_argument("--channel", type=str,
        help="YouTube channel or playlist URL")

    # Options
    parser.add_argument("--max_results", type=int, default=5,
        help="Max videos to discover from search/channel (default: 5)")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR,
        help=f"Root dataset directory (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--no_blur",  action="store_true",
        help="Skip blur filtering step")
    parser.add_argument("--no_dedup", action="store_true",
        help="Skip duplicate removal step")
    parser.add_argument("--dry_run",  action="store_true",
        help="Discover URLs and report only — no downloads or processing")
    parser.add_argument("--stats",    action="store_true",
        help="Print current dataset stats and exit")

    args = parser.parse_args()

    # Stats-only mode
    if args.stats:
        print_stats(args.dataset_dir)
        sys.exit(0)

    run_agent(
        urls        = args.urls,
        query       = args.query,
        channel_url = args.channel,
        max_results = args.max_results,
        dataset_dir = args.dataset_dir,
        skip_blur   = args.no_blur,
        skip_dedup  = args.no_dedup,
        dry_run     = args.dry_run,
    )


if __name__ == "__main__":
    main()