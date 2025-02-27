import cv2

def get_video_info(video_file: str):
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened()
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    return (frame_count, fps, frame_count / fps)
