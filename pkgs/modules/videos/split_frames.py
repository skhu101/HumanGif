
import os
import subprocess


def split_frames(video_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    p = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y",
        "-i", video_file,
        "-c:v", "png",
        f"{output_dir}/%04d.png"
    ])
    ret = p.wait()
    assert ret == 0
