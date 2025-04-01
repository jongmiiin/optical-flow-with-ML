import numpy as np
import cv2
import pandas as pd
import mediapipe as mp

# 비디오 열기 (이 파일 이름으로 저장된 영상 있어야 함)
cap = cv2.VideoCapture('02659_H_A_SY_C8.mp4')

# Shi-Tomasi 설정값
feature_params = dict(
    maxCorners=10,
    qualityLevel=0.05,
    minDistance=40,
    blockSize=7
)

# LK Optical Flow 설정
lk_params = dict(
    winSize=(15, 15),
    maxLevel=0,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 미디어파이프 포즈 인식기
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 첫 프레임 읽고 초기 처리
ret, old_frame = cap.read()
old_rgb = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
results = pose.process(old_rgb)

h, w = old_frame.shape[:2]
p0 = None
if results.pose_landmarks:
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    x, y = int(nose.x * w), int(nose.y * h)
    roi = old_frame[max(0, y-40):min(h, y+40), max(0, x-40):min(w, x+40)]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray_roi, mask=None, **feature_params)
    if p0 is not None:
        p0[:, 0, 0] += max(0, x-40)
        p0[:, 0, 1] += max(0, y-40)

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
vectors = []
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        if p1 is not None and st is not None:
            new_pts = p1[st == 1]
            old_pts = p0[st == 1]

            for i, (new, old) in enumerate(zip(new_pts, old_pts)):
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

    # 다음 프레임에서 코 다시 잡아서 roi로 특징점 새로 추출
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        x, y = int(nose.x * w), int(nose.y * h)
        roi = gray[max(0, y-40):min(h, y+40), max(0, x-40):min(w, x+40)]
        p0 = cv2.goodFeaturesToTrack(roi, mask=None, **feature_params)
        if p0 is not None:
            p0[:, 0, 0] += max(0, x-40)
            p0[:, 0, 1] += max(0, y-40)

    old_gray = gray.copy()
    frame_id += 1

cap.release()
pose.close()

# 결과 저장
pd.DataFrame(vectors).to_csv('optical_flow_vectors.csv', index=False)
print("csv 저장됨")
