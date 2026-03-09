import argparse
import os
import sys
import yt_dlp

DEFAULT_OUTPUT_DIR = "./dataset/media"

def download_options(output_dir: str):
    return{
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': False,
        'retries': 3,
        'writeinfojson': True,
        'write_thumbnail': True,
        "postprocessors": [
            {
                "key": "FFmpegEmbedSubtitle",
            },
            {
                "key": "FFmpegMetadata",
                "add_metadata": True,
            },
        ],

    }

def check_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[✓] Output directory ready: {os.path.abspath(output_dir)}")

def get_video_info(url: str):
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info= ydl.extract_info(url, download=False)
        return info
    
def download_video(url: str, output_dir: str):
    check_output_dir(output_dir)

    print(f"\n[→] Fetching video info for: {url}")
    try:
        info= get_video_info(url)
        print(f"[✓] Title    : {info.get('title', 'Unknown')}")
        print(f"[✓] Uploader : {info.get('uploader', 'Unknown')}")
        duration_s = info.get("duration", 0)
        print(f"[✓] Duration : {duration_s // 60}m {duration_s % 60}s")
        print(f"[✓] Saving to: {os.path.abspath(output_dir)}\n")
    except Exception as e:
        print(f"[!] Could not fetch video info: {e}")
        print("[→] Attempting download anyway...\n")


    dl_opt= download_options(output_dir)
    with yt_dlp.YoutubeDL(dl_opt) as ydl:
        ydl.download([url])

    downloaded_files= [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.mp4')
    ]

    if downloaded_files:
        latest= max(downloaded_files, key=os.path.getmtime)
        print(f'\n[✓] Download complete: {latest}')
        return latest
    else:
        print("\n[!] Download finished but could not locate the .mp4 file.")
        return output_dir

def main():
    parser = argparse.ArgumentParser(
        description="YouTube Video Collector — Vision Data Agent (Phase 1)"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="YouTube video URL to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the video (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    if not args.url.startswith("http"):
        print("[✗] Invalid URL. Please provide a full YouTube URL.")
        sys.exit(1)

    try:
        saved_path = download_video(url=args.url, output_dir=args.output_dir)
        print(f"\n[✓] Video saved to: {saved_path}")
        print("[→] Ready for Phase 2: frame extraction with FFmpeg.")
    except yt_dlp.utils.DownloadError as e:
        print(f"\n[✗] Download failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()