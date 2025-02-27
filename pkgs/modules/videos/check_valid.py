
import subprocess


def check_valid(video_file: str) -> bool:
    p = subprocess.Popen(["ffprobe", "-v", "error", video_file])
    ret = p.wait()
    return ret == 0
