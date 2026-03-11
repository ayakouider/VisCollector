# VisCollector: For vision data collection
A web interface that transforms youtube videos into datasets for Computer Vision research.

## Features:
#### AI powered interface:
     - Natural Language chat powered by Google Gemini.
     - Search ny query, direct URL, or Yotube channel.
     - Intelligent filtering for relevant, high quality videos(30s - 20 mins)
#### Automated processing pipeline:
     1. Video Download
     2. Frame extraction - extracts frames at optimal intervals.
     3. Face detection - identifies and crops faces using ResNet SSD.
     4. Quality filtering - removes blurry and low quality frames.
     5. Duplicate remover - liminates redundant data using perceptual hashing.
#### Real Time Tracking:
     - Live progress updates via WebSocket.
     - Job status monitoring.
     - Dataset statistics dashboard.
     - Download frames and faces as ZIP files.

## Project Structure:
```
VisCollector/
├── process/                  # Video processing modules
|    └── yt_collector.py       # Youtube Video downloader
|    └── frame_extractor.py    # Frame extraction from videos
|    └── face_extractor.py     # Face detection and cropping
|    └── blur.py               # Blur filtering
|    └── duplicate_remover.py  # Duplicate removal (pHash/CLIP)
├── agent/
|    └── agent.py          # Main pipeline orchestrator
|    └── data_manager.py   # Dataset manifest tracking
|    └── discover.py       # Smart video discovery
├── webapp/
|    └── app.html  # React web interface
|    └── app.py    # Flask REST API + WebSocket server
```
### How it works:
1. Discovery Phase:
   - Smart search uses YouTube's API with quality scoring
   - Filters videos by duration (30s-10min by default)
   - Ranks by relevance, views, engagement, and recency
   - Removes live streams and irrelevant content
     
2. Processing pipeline:
```
   Video URL
    ↓
[1] Download (yt-dlp) → saves MP4
    ↓
[2] Extract Frames (OpenCV) → 1 frame every 2-3 seconds
    ↓
[3] Detect Faces (ResNet SSD) → crops all faces
    ↓
[4] Filter Blur (Laplacian variance) → removes low-quality
    ↓
[5] Remove Duplicates (pHash) → eliminates redundancy
    ↓
Clean Dataset Ready!
 ```
3. Real-Time Updates:
   - WebSocket connection shows live progress
   - Download results immediately when complete

### Output Format:
```
dataset/
├── manifest.json                 # Metadata for all videos
├── media/
│   └── video_id.mp4             # Downloaded video
├── frames/
│   └── video_id/
│       ├── frame_0001.jpg       # Extracted frames
│       ├── frame_0002.jpg
│       └── ...
├── faces/
│   └── video_id/
│       ├── face_0001.jpg        # Detected faces
│       ├── face_0002.jpg
│       └── ...
└── reports/
    └── video_id_quality.json    # Quality analysis
```
#### Manifest Format:
```
json {
"642f070910": {
    "url": "https://youtube.com/watch?v=XXX",
    "video_id": "642f070910",
    "title": "Video Title",
    "status": "complete",
    "timestamp": "2026-03-11T10:30:00",
    "frames_count": 245,
    "faces_count": 89
  }
}
```

