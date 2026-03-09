import argparse
import os
import sys
import re
import cv2
from tqdm import tqdm

DEFAULT_INTERVAL_SECONDS = 2 
DEFAULT_FRAMES_DIR = "./dataset/frames"
DEFAULT_FRAME_SIZE = None 
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_JPEG_QUALITY = 95

def clean_vid_name(video_path:str):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    clean = re.sub(r'[^\w\s-]', '', basename)
    clean = re.sub(r'[-\s+]','_',clean.strip())
    return clean


def build_output_dir(frame_dir:str,video_id:str):
    output_dir = os.path.join(frame_dir,video_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_video_features(cap:cv2.VideoCapture):
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps > 0 else 0
    return {
        'fps':fps,
        'total_frames':total_frames,
        'width':width,
        'height':height,
        'duration_s':duration_s
    }

def resize_frame(frame, size:int):
    h,w = frame.shape[:2]
    if w < h :
        new_w = size
        new_h = int(h*size/w)
    else:
        new_h = size
        new_w = int(w*size/h)
    return cv2.resize(frame,(new_w,new_h),interpolation=cv2.INTER_LANCZOS4)

def build_params(fmt:str, quality:int):
    if fmt == 'jpg':
        return[cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == 'png':
        return[cv2.IMWRITE_PNG_COMPRESSION,3]
    else:
        return []
    
def extract_frames(
        video_path:str,
        output_dir:str,
        interval_s:int=DEFAULT_INTERVAL_SECONDS,
        size:int = None,
        fmt:str = DEFAULT_IMAGE_FORMAT,
        quality:int = DEFAULT_JPEG_QUALITY
):
    cap= cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'[!] Error opening video file: {video_path}')
    
    stats= get_video_features(cap)
    fps = stats['fps']
    total_frames = stats['total_frames']
    duration_s = stats['duration_s']

    frame_interval = max(1,int(fps * interval_s))

    expected_count = max(1, int(duration_s / interval_s))

    print(f'\n[✓] Video: {os.path.basename(video_path)}')
    print(f'[✓] Resolution: {stats["width"]}x{stats["height"]}')
    print(f'[✓] FPS: {fps:.2f}')
    print(f'[✓] Duration   : {int(duration_s // 60)}m {int(duration_s % 60)}s')
    print(f'[✓] Interval   : every {interval_s}s  (every {frame_interval} frames)')
    print(f'[✓] Expected frames: ~{expected_count}')
    print(f'[✓] Output     : {os.path.abspath(output_dir)}\n')

    params = build_params(fmt, quality)
    saved_paths = []
    frame_count = 0
    saved_count = 0

    with tqdm(total= expected_count, desc='Extracting frames',unit='frame') as pbar:
        while True:
            ret,frame = cap.read()

            if not ret:
                break


            if frame_count % frame_interval == 0:
                if size:
                    frame = resize_frame(frame,size)

                file_name = f'frame_{saved_count+1:04d}.{fmt}'
                save_path = os.path.join(output_dir,file_name)

                success = cv2.imwrite(save_path,frame,params)
                if success:
                    saved_paths.append(save_path)
                    saved_count += 1
                    pbar.update(1)
                else:
                    print(f'[!] Failed to save frame {file_name}')

            frame_count += 1

    cap.release()
    return saved_paths

def print_summary(saved_paths:list, output_dir:str):
    if not saved_paths:
        print("\n[✗] No frames were saved.")
        return

    # Calculate total size of saved frames
    total_bytes = sum(os.path.getsize(p) for p in saved_paths)
    total_mb    = total_bytes / (1024 * 1024)

    print(f"\n{'─' * 45}")
    print(f"  Frames saved : {len(saved_paths)}")
    print(f"  Total size   : {total_mb:.1f} MB")
    print(f"  Location     : {os.path.abspath(output_dir)}")
    print(f"  First frame  : {os.path.basename(saved_paths[0])}")
    print(f"  Last frame   : {os.path.basename(saved_paths[-1])}")
    print(f"{'─' * 45}")


def main():
    parser = argparse.ArgumentParser(
        description="Frame Extractor — Vision Data Agent (Phase 2)"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video file (e.g. ./dataset/media/my_video.mp4)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"Extract 1 frame every N seconds (default: {DEFAULT_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=DEFAULT_FRAMES_DIR,
        help=f"Root directory for frames output (default: {DEFAULT_FRAMES_DIR})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Resize shortest side to N pixels (e.g. 224 for CLIP). Default: keep original.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=DEFAULT_IMAGE_FORMAT,
        choices=["jpg", "png"],
        help=f"Output image format (default: {DEFAULT_IMAGE_FORMAT})",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG quality 0–100 (default: {DEFAULT_JPEG_QUALITY}, only applies to jpg)",
    )

    args = parser.parse_args()

    # Validate video path
    if not os.path.exists(args.video):
        print(f"[✗] Video file not found: {args.video}")
        sys.exit(1)

    # Derive video_id from filename
    video_id   = clean_vid_name(args.video)
    output_dir = build_output_dir(args.frames_dir, video_id)

    print(f"[→] Video ID: {video_id}")

    try:
        saved_paths = extract_frames(
            video_path = args.video,
            output_dir = output_dir,
            interval_s = args.interval,
            size       = args.size,
            fmt        = args.format,
            quality    = args.quality,
        )
        print_summary(saved_paths, output_dir)

    except IOError as e:
        print(f"\n[✗] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


    