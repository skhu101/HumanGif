import cv2
import numpy as np


def get_frame_info(video_file: str):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps

def get_moved_area_mask(frames, move_th=40, th=-1):
    ref_frame = frames[0]
    # Convert the reference frame to gray
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = ref_gray
    # Initialize the total accumulated motion mask
    total_mask = np.zeros_like(ref_gray)

    # Iterate through the video frames
    for i in range(1, len(frames)):
        frame = frames[i]
        # Convert the frame to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the reference frame and the current frame
        diff = cv2.absdiff(ref_gray, gray)
        # diff += cv2.absdiff(prev_gray, gray)

        # Apply a threshold to obtain a binary image
        ret, mask = cv2.threshold(diff, move_th, 255, cv2.THRESH_BINARY)

        # Accumulate the mask
        total_mask = cv2.bitwise_or(total_mask, mask)

        # Update the reference frame
        prev_gray = gray

    contours, _ = cv2.findContours(total_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    ref_mask = np.zeros_like(ref_gray)
    ref_mask = cv2.drawContours(ref_mask, contours, -1, (255, 255, 255), -1)
    for cnt in contours:
        cur_rec = cv2.boundingRect(cnt)
        rects.append(cur_rec)

    #rects = merge_overlapping_rectangles(rects)
    mask = np.zeros_like(ref_gray)
    if th < 0:
        h, w = mask.shape
        th = int(h*w*0.1)
    for rect in rects:
        x, y, w, h = rect
        if w*h < th:
            continue
        #ref_frame = cv2.rectangle(ref_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        mask[y:y+h, x:x+w] = 255
    return mask
