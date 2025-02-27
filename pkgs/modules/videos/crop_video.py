
from pathlib import Path
import subprocess
import sys

import cv2

def crop_square(video_file: str, output_file: str):
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened()

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    w = min(video_width, video_height)
    h = w
    x = (video_width - w) // 2
    y = (video_height - h) // 2

    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-i", video_file, "-vf", f"crop={w}:{h}:{x}:{y}", "-c:a", "copy", output_file])
    ret = p.wait()
    if ret != 0:
        print(f"ERROR: {video_file} crop_to_resize ret={ret}")
    return output_file

def crop_video(video_file: str, output_file: str, bboxes: list[tuple[int, int, int, int]], expansion_ratio=1.5, square=True):
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened()

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    assert frame_count == len(bboxes)

    output_filename = Path(output_file).name.split(".")[0]
    start_frame = 0

    output_files = []
    for bxes in split_bboxes(bboxes):
        end_frame = start_frame + len(bxes)
        start_sec = start_frame / fps
        end_sec = end_frame / fps
        x, y, w, h = calc_max_bbox(bxes)

        x = x - int((expansion_ratio - 1) * w / 2)
        y = y - int((expansion_ratio - 1) * h / 2)
        w = min(video_width, int(w * expansion_ratio))
        h = min(video_height, int(h * expansion_ratio))
        x = max(0, x)
        y = max(0, y)

        if square:
            if w > h:
                y -= (w - h) // 2
            else:
                x -= (h - w) // 2
            w = max(w, h)
            h = w
        filename = str(Path(output_file).parent / f"{output_filename}-face-{start_frame}-{end_frame}.mp4")
        p = subprocess.Popen([
            "ffmpeg", "-v", "error", "-y",
            "-i", video_file,
            "-ss", str(start_sec),
            "-t", str(end_sec - start_sec),
            "-vf", f"crop={w}:{h}:{x}:{y}",
            "-c:a", "copy", filename
        ])
        ret = p.wait()
        if ret != 0:
            print(f"ERROR: {video_file} crop_video ret={ret}")

        start_frame += len(bxes)
        output_files.append(filename)

    return output_files

def calc_max_bbox(bboxes: list[tuple[int, int, int ,int]]):
    # 所有bbox取并集
    start_x = sys.maxsize
    start_y = sys.maxsize
    end_x = 0
    end_y = 0
    width = 0
    height = 0

    for bbox in bboxes:
        current_start_x = bbox[0]
        current_start_y = bbox[1]
        current_width = bbox[2]
        current_height = bbox[3]
        current_end_x = current_start_x + current_width
        current_end_y = current_start_y + current_height
        start_x = min(current_start_x, start_x)
        start_y = min(current_start_y, start_y)
        end_x = max(current_end_x, end_x)
        end_y = max(current_end_y, end_y)

    width = end_x - start_x
    height = end_y - start_y

    return (start_x, start_y, width, height)

def split_bboxes(bboxes: list[tuple[int, int, int, int]], distance=0.3) -> list[list[tuple[int, int, int, int]]]:
    """根据bbox的移动距离，分割bbox列表

    Args:
        bboxes (list[tuple[int, int, int, int]]): _description_
        distance (float, optional): _description_. Defaults to 0.4.

    Returns:
        list[list[tuple[int, int, int, int]]]: _description_
    """
    results: list[list[tuple[int, int, int, int]]] = []
    base_bbox = bboxes[0]
    calc_distance = lambda x, y: abs((x - y) / x)

    cache: list[tuple[int, int, int, int]] = []
    for bbox in bboxes:
        max_distance = max(
            calc_distance(bbox[0], base_bbox[0]),
            calc_distance(bbox[1], base_bbox[1]),
            calc_distance(bbox[2], base_bbox[2]),
            calc_distance(bbox[3], base_bbox[3]),
        )
        if max_distance > distance:
            # 切分列表
            if len(cache) > 0:
                results.append(cache)
            base_bbox = bbox
            cache = []
        cache.append(bbox)
    
    if len(cache) > 0:
        results.append(cache)

    return results

def draw_bboxes(video_file: str, output_file: str, bboxes: list[tuple[int, int, int, int]]):
    # 打开视频文件
    cap = cv2.VideoCapture(video_file)

    # 获取视频的总帧数和帧的FPS（每秒帧数）
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # 定义编解码器和文件格式
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    color = (0, 255, 0)
    thickness = 1

    frame_index = 0
    while True:
        if frame_index >= len(bboxes):
            break;

        ret, frame = cap.read()
        if not ret:
            break

        rect = bboxes[frame_index]
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, thickness)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    
if __name__ == "__main__":
    import json

    from constants import CACHE_DIR

    json_obj = None
    with open("/home/leeway/workspace/data_processors/.cache/output/stage11/BV1A54y1C7Da-no_overlap-0-2594-scene-0-2594.json", "r", encoding="utf-8") as f:
        json_obj = json.loads(f.read())

    bboxes = list(map(lambda x: (x["bounding_box"]["origin_x"], x["bounding_box"]["origin_y"], x["bounding_box"]["width"], x["bounding_box"]["height"]), json_obj["segments"][0]["detections"]))
    output_file = CACHE_DIR / "test.mp4"
    # output_file = CACHE_DIR / "test_bbox.mp4"
    crop_video("/home/leeway/workspace/data_processors/.cache/output/stage9/BV1A54y1C7Da-no_overlap-0-2594-scene-0-2594.mp4", output_file, bboxes, 1.5)
    # draw_bboxes("/home/leeway/workspace/data_processors/.cache/output/stage9/BV1A54y1C7Da-no_overlap-0-2594-scene-0-2594.mp4", str(output_file), bboxes)
