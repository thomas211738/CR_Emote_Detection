import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

random.seed(42)

INPUT_ROOT = Path("data/frames/original_frames")
OUTPUT_ROOT = Path("data/frames")

CLASSES = ["Cry", "HandsUp", "Still", "TongueOut", "Yawn"]

SPLITS = {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15,
}

def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

## used AI to help write these function
## Splits the videos in train val and test
## puts the videos into folders based on their class and split
## keeps all frames from the same video together in the same split
def split_class_by_video(cls_name: str):
    in_dir = INPUT_ROOT / cls_name

    # Group frames by 'video id' (prefix before _f)
    video_to_files = defaultdict(list)

    for fname in os.listdir(in_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        stem = Path(fname).stem  
        if "_f" in stem:
            video_id = stem.split("_f")[0]  # "Cry6"
        else:
            video_id = stem  

        video_to_files[video_id].append(fname)

    video_ids = list(video_to_files.keys())
    if not video_ids:
        print(f"[WARN] No videos found for class {cls_name} in {in_dir}")
        return

    random.shuffle(video_ids)

    n_videos = len(video_ids)
    n_train = int(n_videos * SPLITS["train"])
    n_val = int(n_videos * SPLITS["validation"])
    n_test = n_videos - n_train - n_val

    split_vids = {
        "train": video_ids[:n_train],
        "validation": video_ids[n_train:n_train + n_val],
        "test": video_ids[n_train + n_val:],
    }

    print(
        f"Class {cls_name}: videos={n_videos}, "
        f"train={len(split_vids['train'])}, "
        f"val={len(split_vids['validation'])}, "
        f"test={len(split_vids['test'])}"
    )

    # Copy all frames from each video into the chosen split
    for split_name, vids in split_vids.items():
        out_dir = OUTPUT_ROOT / split_name / cls_name
        make_dir(out_dir)

        for vid in vids:
            for fname in video_to_files[vid]:
                src = in_dir / fname
                dst = out_dir / fname
                shutil.copy2(src, dst)

def main():
    
    for cls in CLASSES:
        split_class_by_video(cls)
    print("Done splitting")

if __name__ == "__main__":
    main()

