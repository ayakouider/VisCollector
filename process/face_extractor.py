import argparse
import os
import re
import sys
import urllib.request
import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_FACES_DIR = "./dataset/faces"
DEFAULT_CONFIDENCE = 0.65
DEFAULT_PADDING = 20
DEFAULT_MIN_FACE_SIZE = 50
DEFAULT_IMAGE_FORMAT  = "jpg"
DEFAULT_JPEG_QUALITY    = 95

MODEL_DIR    = "./models"
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
WEIGHTS_URL  = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
WEIGHTS_PATH  = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

def download_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)

    files = [
        (PROTOTXT_PATH,PROTOTXT_URL,"deploy.prototxt"),
        (WEIGHTS_PATH,WEIGHTS_URL,"res10_300x300_ssd_iter_140000.caffemodel")
    ]

    for path, url, name in files:
        if not os.path.exists(path):
            print(f"[→] Downloading model file: {name}")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"[✓] Saved to: {path}")
            except Exception as e:
                raise RuntimeError(
                    
                    f"Failed to download model file: {name}\n"
                    f"Error: {e}\n"
                    f"Please manually download from:\n{url}\n"
                    f"and save to: {path}"
                )      
        else:
            print(f"[✓] Model file found: {name}")

def load_face_detector():
    download_model_files()
    net= cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)
    print("[✓] Face detector loaded (ResNet SSD)\n")
    return net

def clean_id(path: str) -> str:
    basename = os.path.basename(path.rstrip("/\\"))
    clean    = re.sub(r"[^\w\s-]", "", basename)
    clean    = re.sub(r"[\s]+", "_", clean.strip())
    return clean

def get_image_files(frames_dir: str):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [
        os.path.join(frames_dir, f)
        for f in sorted(os.listdir(frames_dir))
        if os.path.splitext(f)[1].lower() in extensions
    ]
    return files

def detect_faces(net, image:np.array, min_conf:float):
    h,w= image.shape[:2]

    blob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),
                               scalefactor=1.0,
                               size=(300,300),
                                 mean=(104.0,177.0,123.0))
    net.setInput(blob)
    detections= net.forward()

    faces= []
    for i in range(detections.shape[2]):
        confidence= float(detections[0,0,i,2])
        if confidence < min_conf:
            continue

        box = detections[0,0,i,3:7]* np.array([w,h,w,h])
        x1,y1,x2,y2= box.astype("int")
        faces.append((x1,y1,x2,y2,confidence))

    return faces

def crop_face(
    image: np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    padding: int,
    min_size: int
):
    h,w=image.shape[:2]

    x1 = max(0,x1 - padding)
    y1 = max(0,y1 - padding)
    x2 = min(w,x2 + padding)
    y2 = min(h,y2 + padding)

    face_w= x2 - x1
    face_h= y2 - y1

    if face_w < min_size or face_h < min_size:
        return None
    
    return image[y1:y2,x1:x2]

def extract_faces(
    frames_dir: str,
    output_dir: str = DEFAULT_FACES_DIR,
    min_confidence: float = DEFAULT_CONFIDENCE,
    padding: int = DEFAULT_PADDING,
    min_size: int = DEFAULT_MIN_FACE_SIZE,
    fmt: str = DEFAULT_IMAGE_FORMAT,
    quality: int = DEFAULT_JPEG_QUALITY,
):
    
    image_files = get_image_files(frames_dir)
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {frames_dir}")
    
    video_id = clean_id(frames_dir)
    faces_dir= os.path.join(output_dir, video_id)
    os.makedirs(faces_dir, exist_ok=True)

    print(f"[→] Scanning {len(image_files)} frames for faces")
    print(f"[✓] Min confidence : {min_confidence}")
    print(f"[✓] Min face size  : {min_size}x{min_size}px")
    print(f"[✓] Padding        : {padding}px")
    print(f"[✓] Output         : {os.path.abspath(faces_dir)}\n")

    net = load_face_detector()
    params= (
        [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg"
        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )

    saved_paths=[]
    face_counter=0
    frames_with_faces=0

    with tqdm(image_files, desc="Detecting faces", unit="frame") as pbar:
        for frame_path in pbar:
            image = cv2.imread(frame_path)
            if image is None:
                continue

            faces = detect_faces(net, image, min_confidence)

            if faces:
                frames_with_faces += 1

            for (x1, y1, x2, y2, confidence) in faces:
                crop = crop_face(image, x1, y1, x2, y2, padding, min_size)
                if crop is None:
                    continue

                face_counter += 1
                filename  = f"face_{face_counter:04d}.{fmt}"
                save_path = os.path.join(faces_dir, filename)

                cv2.imwrite(save_path, crop, params)
                saved_paths.append(save_path)

            pbar.set_postfix(faces=face_counter)

    return saved_paths, frames_with_faces, len(image_files)

def main():
    parser = argparse.ArgumentParser(
        description="Face Extractor — Vision Data Agent (Phase 4)"
    )
    parser.add_argument("--frames_dir", type=str, required=True,
        help="Directory of extracted frames (e.g. ./dataset/frames/my_video)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_FACES_DIR,
        help=f"Root output directory for faces (default: {DEFAULT_FACES_DIR})")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help=f"Min detection confidence 0–1 (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING,
        help=f"Padding around face crop in pixels (default: {DEFAULT_PADDING})")
    parser.add_argument("--min_size", type=int, default=DEFAULT_MIN_FACE_SIZE,
        help=f"Min face size in pixels (default: {DEFAULT_MIN_FACE_SIZE})")
    parser.add_argument("--format", type=str, default=DEFAULT_IMAGE_FORMAT,
        choices=["jpg", "png"],
        help=f"Output image format (default: {DEFAULT_IMAGE_FORMAT})")

    args = parser.parse_args()

    if not os.path.isdir(args.frames_dir):
        print(f"[✗] Frames directory not found: {args.frames_dir}")
        sys.exit(1)

    try:
        saved_paths, frames_with_faces, total_frames = extract_faces(
            frames_dir     = args.frames_dir,
            output_dir     = args.output_dir,
            min_confidence = args.confidence,
            padding        = args.padding,
            min_size       = args.min_size,
            fmt            = args.format,
        )

        print(f"\n{'─' * 45}")
        print(f"  Frames scanned     : {total_frames}")
        print(f"  Frames with faces  : {frames_with_faces}")
        print(f"  Total faces saved  : {len(saved_paths)}")
        if saved_paths:
            total_mb = sum(os.path.getsize(p) for p in saved_paths) / (1024 * 1024)
            print(f"  Total size         : {total_mb:.1f} MB")
        print(f"{'─' * 45}\n")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[✗] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

