from __future__ import annotations
import json 
import os
import hashlib
from datetime import datetime

DEFAULT_DATASET_DIR = "./dataset"
MANIFEST_FILENAME   = "manifest.json"

def _empty_file(file_path):
    return {
        "created_at"  : datetime.now().isoformat(),
        "updated_at"  : datetime.now().isoformat(),
        "total_videos": 0,
        "total_frames": 0,
        "total_faces" : 0,
        "videos"      : {}
    }

def _file_path(dataset_dir:str):
    return os.path.join(dataset_dir, MANIFEST_FILENAME)

def load_file(dataset_dir:str=DEFAULT_DATASET_DIR):
    path = _file_path(dataset_dir)
    if os.path.exists(path):
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    else:
        return _empty_file(path)

def save_file(data:dict, dataset_dir:str=DEFAULT_DATASET_DIR):
    os.makedirs(dataset_dir, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    path=_file_path(dataset_dir)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path

def url_to_vid_id(url:str):
    return hashlib.md5(url.encode()).hexdigest()[:10]

class Status:
    Pending = "pending",
    Downloaded = "downloaded",
    Processed = "processed",
    Filtered = "filtered",
    Complete = "complete"
    Failed = "failed"
    Skipped = "skipped"

def register_video(
        url: str,
    dataset_dir: str = DEFAULT_DATASET_DIR,
):
    manager=load_file(dataset_dir)
    vid_id = url_to_vid_id(url)

    if vid_id not in manager["videos"]:
        manager["videos"][vid_id] = {
            "video_id"    : vid_id,
            "url"         : url,
            "status"      : Status.Pending,
            "registered_at": datetime.now().isoformat(),
            "video_path"  : None,
            "frames_dir"  : None,
            "faces_dir"   : None,
            "audio_path"  : None,
            "transcript_path": None,
            "frame_count" : 0,
            "face_count"  : 0,
            "kept_frames" : 0,
            "error"       : None,
        }
        manager["total_videos"] += 1
        save_file(manager, dataset_dir)
        print(f"[dataset] Registered: {vid_id} → {url}")
    else:
        print(f"[dataset] Already registered: {vid_id}")

    return vid_id, manager["videos"][vid_id]

def update_video(
    video_id: str,
    updates: dict,
    dataset_dir: str = DEFAULT_DATASET_DIR
):
    manager = load_file(dataset_dir)
    if video_id not in manager["videos"]:
        raise KeyError(f"[dataset] Unknown video_id: {video_id}")
    
    entry = manager["videos"][video_id]
    entry.update(updates)
    entry["updated_at"] = datetime.now().isoformat()

    manager["total_frames"] = sum( 
        v.get("frame_count",0) for v in manager["videos"].values()
    )

    manager["total_faces"] = sum(
        v.get("face_count",0) for v in manager["videos"].values()
    )

    save_file(manager, dataset_dir)
    return entry


def is_processed(
    url: str,
    dataset_dir: str = DEFAULT_DATASET_DIR
):
    manager=load_file(dataset_dir)
    vid_id = url_to_vid_id(url)
    entry=manager["videos"].get(vid_id)
    if entry and entry["status"] == Status.Complete:
        print(f"[dataset] Skipping already complete: {url}")
        return True
    return False

def get_stats(dataset_dir: str = DEFAULT_DATASET_DIR):
    manager=load_file(dataset_dir)
    videos = manager["videos"].values()

    status_counts = {}
    for v in videos:
        s = v.get("status", "unknown")
        if isinstance(s, list):
            s = s[0] if s else "unknown"
        status_counts[s] = status_counts.get(s, 0) + 1
    return {
        "total_videos" : manager["total_videos"],
        "total_frames" : manager["total_frames"],
        "total_faces"  : manager["total_faces"],
        "by_status"    : status_counts,
        "manifest_path": _file_path(dataset_dir)
    }

def print_stats(dataset_dir:str = DEFAULT_DATASET_DIR):
    stats= get_stats(dataset_dir)
    print(f"\n{'─' * 45}")
    print(f"  DATASET SUMMARY")
    print(f"{'─' * 45}")
    print(f"  Total videos : {stats['total_videos']}")
    print(f"  Total frames : {stats['total_frames']}")
    print(f"  Total faces  : {stats['total_faces']}")
    print(f"  By status    :")
    for status, count in stats["by_status"].items():
        print(f"    {status:<15}: {count}")
    print(f"  Manifest     : {stats['manifest_path']}")
    print(f"{'─' * 45}\n")