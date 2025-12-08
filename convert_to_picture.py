import os
from pathlib import Path
import cv2

# Paths
VIDEO_ROOT = Path("data/all_clips/output_clips")
FRAME_ROOT = Path("data/frames")

CLASSES = ["Cry", "HandsUp", "Still", "TongueOut", "Yawn"]
N_FRAMES = 4

##  Extracts 4 evenly spaced frames from each video and saves them as images - this is used for our image 
## version of the CNN, which is trained on individual frames rather than video sequences, and intakes frames later on

## used AI to help write these function

def extract_frames_from_video(video_path, out_dir, n_frames=N_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"No frames in {video_path}")
        cap.release()
        return

    indices = [
        int(frame_count * (i + 1) / (n_frames + 1))
        for i in range(n_frames)
    ]

    stem = video_path.stem  
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            print(f"Failed to read frame {idx} from {video_path}")
            continue

        out_path = out_dir / f"{stem}_f{i+1}.jpg"
        cv2.imwrite(str(out_path), frame)

    cap.release()

def main():
    for cls in CLASSES:
        video_dir = VIDEO_ROOT / cls
        out_dir = FRAME_ROOT / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname in os.listdir(video_dir):
            if not fname.lower().endswith(".mp4"):
                continue
            video_path = video_dir / fname
            print("Extracting frames from:", video_path)
            extract_frames_from_video(video_path, out_dir)

if __name__ == "__main__":
    main()
    print("Done extracting frames.")
