import numpy as np
import cv2
import pandas as pd
import mediapipe as mp

# 영상 불러오기
cap = cv2.VideoCapture('02659_H_A_SY_C8.mp4')

# 특징점 잡을 때 쓸 설정값
feature_params = dict(
    maxCorners=10,
    qualityLevel=0.05,
    minDistance=40,
    blockSize=7
)

# 옵티컬 플로우 계산할 때 쓸 설정값
lk_params = dict(
    winSize=(15, 15),
    maxLevel=0,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 미디어파이프 포즈 추정기 세팅
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 첫 프레임 읽고, 포즈 추정
ret, old_frame = cap.read()
old_rgb = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
results = pose.process(old_rgb)

# 프레임 크기 저장
height, width = old_frame.shape[:2]

# 코 주변 영역에서 특징점 추출
p0 = None
if results.pose_landmarks:
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    x, y = int(nose.x * width), int(nose.y * height)
    roi_size = 40
    x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
    x2, y2 = min(width, x + roi_size), min(height, y + roi_size)
    head_roi = cv2.cvtColor(old_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(head_roi, mask=None, **feature_params)
    if p0 is not None:
        p0[:, 0, 0] += x1
        p0[:, 0, 1] += y1

# 이전 프레임 gray로 저장
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
vectors = []
frame_id = 0

# 프레임 루프 시작
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임 전처리
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # 이전 프레임 기준으로 Optical Flow 계산
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                x0, y0 = old.ravel()
                x1, y1 = new.ravel()
                dx = x1 - x0
                dy = y1 - y0
                mag = np.sqrt(dx ** 2 + dy ** 2)
                angle = np.arctan2(dy, dx)
                vectors.append({
                    'frame_id': frame_id,
                    'point_id': i,
                    'x0': x0, 'y0': y0,
                    'x1': x1, 'y1': y1,
                    'dx': dx, 'dy': dy,
                    'magnitude': mag,
                    'angle': angle
                })

    # 다음 프레임 기준으로 다시 머리 영역에서 특징점 추출
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        x, y = int(nose.x * width), int(nose.y * height)
        roi_size = 40
        x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
        x2, y2 = min(width, x + roi_size), min(height, y + roi_size)
        head_roi = frame_gray[y1:y2, x1:x2]
        p0 = cv2.goodFeaturesToTrack(head_roi, mask=None, **feature_params)
        if p0 is not None:
            p0[:, 0, 0] += x1
            p0[:, 0, 1] += y1

    # 다음 프레임 처리를 위해 현재 프레임 저장
    old_gray = frame_gray.copy()
    frame_id += 1

# 리소스 정리
cap.release()
pose.close()

# 벡터 결과 저장
df = pd.DataFrame(vectors)
df.to_csv('optical_flow_vectors.csv', index=False)
print("저장 완료")
