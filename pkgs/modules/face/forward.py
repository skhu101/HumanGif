import cv2
import mediapipe as mp


def detect_landmarks(video_file: str):
    cap = cv2.VideoCapture(video_file)
    mp_face_mesh = mp.solutions.face_mesh

    cap.release()

VIDEO_FILE = "/home/leeway/workspace/data_processors/.cache/output/stage12/O-yYdBTW-Jk_13_0-d480p-no_overlap-0-334-scene-0-334.mp4"

OUTPUT_FILE = ".cache/landmarks.mp4"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_estimation = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_FILE)

# 获取视频的总帧数和帧的FPS（每秒帧数）
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 创建VideoWriter对象
fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # 定义编解码器和文件格式
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (frame_width, frame_height))

assert cap.isOpened()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    pose_results = pose_estimation.process(frame_rgb)

    if pose_results.pose_landmarks:
        # You can access pose landmarks and orientation information here
        landmarks = pose_results.pose_landmarks
        # Extract relevant pose landmarks and calculate face orientation

        # Draw landmarks and orientation lines
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # if results.multi_face_landmarks:
        # for face_landmarks in results.multi_face_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image=frame,
        #         landmark_list=face_landmarks,
        #         connections=mp_face_mesh.FACEMESH_TESSELATION,
        #         landmark_drawing_spec=None,
        #         connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_tesselation_style())
        #     mp_drawing.draw_landmarks(
        #         image=frame,
        #         landmark_list=face_landmarks,
        #         connections=mp_face_mesh.FACEMESH_CONTOURS,
        #         landmark_drawing_spec=None,
        #         connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_contours_style())
        #     mp_drawing.draw_landmarks(
        #         image=frame,
        #         landmark_list=face_landmarks,
        #         connections=mp_face_mesh.FACEMESH_IRISES,
        #         landmark_drawing_spec=None,
        #         connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_iris_connections_style())
            
    out.write(frame)

out.release()
cap.release()
