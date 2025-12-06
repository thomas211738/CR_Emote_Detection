import os
import subprocess
from pathlib import Path

# Root folders
INPUT_ROOT = Path("data/all_clips/original_clips")
OUTPUT_ROOT = Path("data/all_clips/output_clips")

# Your emote class names (must match folder names)
CLASSES = ["Cry", "HandsUp", "Still", "TongueOut", "Yawn"]

# Desired output settings
OUT_SIZE = "224x224"  # width x height
OUT_FPS = 15

def convert_all():
    for cls in CLASSES:
        in_dir = INPUT_ROOT / cls
        out_dir = OUTPUT_ROOT / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname in os.listdir(in_dir):
            # Skip non-video files just in case
            if not fname.lower().endswith((".mov", ".mp4", ".m4v")):
                continue

            in_path = in_dir / fname

            # Make a clean .mp4 filename
            stem = in_path.stem  # filename without extension
            out_path = out_dir / f"{stem}.mp4"

            # ffmpeg command: resize, set fps, re-encode to mp4
            cmd = [
                "ffmpeg",
                "-y",                    # overwrite without asking
                "-i", str(in_path),      # input
                "-vf", "scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:-1:-1:color=black",
                "-r", str(OUT_FPS),      # fps
                "-c:v", "libx264",       # video codec
                "-preset", "fast",
                "-crf", "23",
                str(out_path),
            ]

            print("Converting:", in_path, "->", out_path)
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    convert_all()
    print("Done converting all videos!")
