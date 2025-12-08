import os
import subprocess
from pathlib import Path

# Root folders
INPUT_ROOT = Path("data/all_clips/original_clips")
OUTPUT_ROOT = Path("data/all_clips/output_clips")


CLASSES = ["Cry", "HandsUp", "Still", "TongueOut", "Yawn"]


OUT_SIZE = "224x224"  
OUT_FPS = 15

## converts the videos from .mov to .mp4 and resizes them this is needed for the model since we break
## down the videos into frames of 224x224 pixels later on, also it makes sure the videos are smaller and same fps
## standard for easier processing later on
## used AI to help write these function
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

            
            stem = in_path.stem  
            out_path = out_dir / f"{stem}.mp4"

            # ffmpeg commannds
            cmd = [
                "ffmpeg",
                "-y",                    
                "-i", str(in_path),      
                "-vf", "scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:-1:-1:color=black",
                "-r", str(OUT_FPS),      # fps
                "-c:v", "libx264",       
                "-preset", "fast",
                "-crf", "23",
                str(out_path),
            ]

            print("Converting:", in_path, "->", out_path)
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    convert_all()
    print("Done converting all videos!")
