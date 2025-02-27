import asyncio
import json
import os
import subprocess
import sys
import cv2
from pathlib import Path

import aioshutil
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from scenedetect import AdaptiveDetector, detect

from modules.videos.check_valid import check_valid
from utils.video import get_moved_area_mask

def get_frame_info(video_file: str):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps

def down_resolution(video_file: str, height: int, output_file: str):
    """down resolution. If already less than 'height', copy to dest

    Args:
        video_file (str): video
        height (int): target resolution height
        output_file (str): output file path

    Returns:
        _type_: _description_
    """
    cap = cv2.VideoCapture(video_file)
    origin_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    if origin_height < height:
        asyncio.run(aioshutil.copyfile(video_file, output_file))
        return output_file

    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-i", video_file,
                          "-c:a", "libmp3lame", "-c:v", "libx264",
                          "-vf", f"scale=-2:{str(height)}", output_file])
    ret = p.wait()

    assert ret == 0

    return output_file

def has_audio(video_file: str):
    clip = VideoFileClip(video_file)
    res = clip.audio is not None
    clip.close()
    return res

def split_wav_audio(video_file: str, output_file: str):
    sb = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-i", video_file, "-vn", "-c:a", "pcm_s16le", output_file])
    ret = sb.wait()
    if ret != 0:
        print(f"{video_file} error")
    return output_file

def split_video(video_file: str, segments: list[tuple[float, float]], output_dir: str, name_template: str="$NAME-Split-$FRAME_START-$FRAME_END.$EXT"):
    size = os.stat(video_file).st_size
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    cap.release()

    output_files = []
    video_name = Path(video_file).name.split(".")[0]
    ext = Path(video_file).name.split(".")[1]

    def parse_name(template: str, name, frame_start, frame_end, ext):
        template = template.replace("$NAME", name)
        template = template.replace("$FRAME_START", str(frame_start))
        template = template.replace("$FRAME_END", str(frame_end))
        template = template.replace("$EXT", ext)
        return template

    os.makedirs(output_dir, exist_ok=True)

    if len(segments) <= 0:
        output_file = os.path.join(output_dir, parse_name(name_template, video_name, 0, frame_count, ext))
        asyncio.run(aioshutil.copy(video_file, output_file))
        output_files.append(output_file)
        return output_files

    for seg in segments:
        (start, end) = seg
        if start >= duration or end <= 0:
            continue
        end = min(end, duration)
        frame_start = int(start * fps)
        frame_end = int(end * fps)
        output_file = os.path.join(output_dir, parse_name(name_template, video_name, frame_start, frame_end, ext))

        if os.path.exists(output_file):
            if check_valid(output_file):
                print(f"{video_file} exists. skip...")
                continue

        try:
            p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", video_file, "-t", str(end - start),
                                #   "-c:v", "libx264", "-c:a", "libmp3lame",
                                  output_file])
            assert p.wait() == 0
            output_files.append(output_file)
        except Exception as e:
            print(f"split failed! {str(e)}")

    return output_files

def detect_scene(
    video_file: str,
    adaptive_threshold=0.5,
    start_time: float | None=None,
    end_time: float | None=None,
    min_scene_duration=2.0,
    max_scene_duration=None,
    drop_frames=0,
    output_json_file: str=None
):
    max_scene_duration = max_scene_duration if max_scene_duration else sys.maxsize
    frame_count, fps = get_frame_info(video_file)
    start_frame = 0
    end_frame = frame_count

    if start_time:
        start_frame = int(start_time * fps)

    if end_time:
        end_frame = int(end_time * fps)

    scene_list = detect(video_file, AdaptiveDetector(adaptive_threshold=adaptive_threshold, window_width=1), start_time=start_time, end_time=end_time)

    segments: list[tuple[float, float]] = []
    for start, end in scene_list:
        scene_duration = end.get_seconds() - start.get_seconds()
        if scene_duration >= min_scene_duration and scene_duration <= max_scene_duration:
            segments.append((start.get_seconds(), end.get_seconds()))
        elif scene_duration > max_scene_duration:
            num_splits = int(scene_duration / max_scene_duration) + 1
            split_duration = scene_duration / num_splits
            for i in range(num_splits):
                split_start = start + i * split_duration
                split_end = start + (i + 1) * split_duration
                if i == num_splits - 1:
                    split_end = end
                segments.append((split_start.get_seconds(), split_end.get_seconds()))

    result = {
        "video_file": video_file,
        "segments": segments
    }

    if output_json_file:
        with open(output_json_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(result))

    return result


def detect_motion(video_file: str, output_json_file: str=None):
    frames = get_frames(video_file)

    masks = get_moved_area_mask(frames)

    motion_detected = bool(np.all(masks == 255))

    result = {
        "video_file": video_file,
        "motion_detected": motion_detected
    }
    
    if output_json_file:
        with open(output_json_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(result))
        
    return result

def get_frames(video_file: str, frame_num: int = -1):
    cap = cv2.VideoCapture(video_file)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_num == -1:
        frame_num = count
    else:
        frame_num = min(frame_num, count)
    frames = []
    for _ in range(frame_num):
        ret, ref_frame = cap.read()
        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        if not ret:
            raise ValueError("Failed to read video file")
        frames.append(ref_frame)
    return frames
