import copy
import json

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def detect_face(video_file: str, least_frames, score_threshold, save_path: str=None):
    cap = cv2.VideoCapture(video_file)

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path="pretrained_models/blaze_face_short_range.tflite"),
        running_mode=VisionRunningMode.IMAGE,
    )

    results_list = []
    with FaceDetector.create_from_options(options) as detector:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            face_detector_result = detector.detect(mp_image)
            results_list.append(face_detector_result)

    # 处理检测结果
    # 初始化临时和总结果列表
    # 初始化总结果
    # res_all = {"aa": video_file, "segments": []}
    # 初始化当前片段结果
    res = {"detections": [], "start_frame": 0, "end_frame": 0, "sort": False}
    current_frame = 0
    previous_detection = None

    segments = []

    for detection_result in results_list:
        if not detection_result.detections or \
                (previous_detection and are_detections_different(previous_detection, detection_result)) \
                or not all(det.categories[0].score > score_threshold for det in detection_result.detections):

            if (res["end_frame"] - res["start_frame"] + 1) >= least_frames:
                segments.append(copy.deepcopy(res))
            res = {"detections": [], "start_frame": current_frame, "end_frame": current_frame, "sort": False}
        else:
            # print(detection_result.detections[0].categories[0].score)

            res["end_frame"] = current_frame
            res["detections"].extend(
                [{"bounding_box": det.bounding_box, "frame": current_frame} for det in detection_result.detections])
            # if len(detection_result.detections) > 1:
            #     res["sort"] = True
            # else:
            #     res["sort"] = False

        previous_detection = detection_result
        current_frame += 1

    # 处理最后一个res
    if (res["end_frame"] - res["start_frame"] + 1) >= least_frames:
        segments.append(res)

    for segment in segments:
        for detection in segment["detections"]:
            # 假设每个detection包含一个bounding_box对象
            bbox = detection["bounding_box"]
            detection["bounding_box"] = {
                "origin_x": bbox.origin_x,
                "origin_y": bbox.origin_y,
                "width": bbox.width,
                "height": bbox.height,
            }

    result = {
        "video_file": video_file,
        "segments": segments,
    }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=4))

    return result

# 定义比较两个检测结果是否相同的函数
def are_detections_different(det1, det2):
    return len(det1.detections) != len(det2.detections)

def split_bboxes(bboxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """bbox变化大时，切分位多个列表

    Args:
        bboxes (list[tuple[int, int, int, int]]): _description_
    """
    result: list[list[tuple[int, int, int, int]]] = []
    cache: list[tuple[int, int, int, int]] = []
    def calc_distance(x, y):
        return abs((x - y) / x)
    pre = bboxes[0]
    for bbox in bboxes:
        distance = max(
            calc_distance(bbox[0], pre[0]),
            calc_distance(bbox[1], pre[1]),
            calc_distance(bbox[2], pre[2]),
            calc_distance(bbox[3], pre[3]),
        )
        if distance >= 0.5:
            if len(cache) > 0:
                result.append(cache)
            cache = [bbox]
        else:
            cache.append(bbox)
    if len(cache) > 0:
        result.append(cache)

    return result
