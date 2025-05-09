import cv2 as cv
import mediapipe as mp
import numpy as np
import os

# 저장 폴더 만들기 (없으면 생성)
os.makedirs("detected_falls", exist_ok=True)

# 영상 로드
cap = cv.VideoCapture(cv.samples.findFile("test.mp4"))

# Mediapipe pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

prev_gray = None
prev_points = None
prev_dy_ratio = None
consecutive_movement = 0  # 연속 큰 움직임 카운트

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number = int(cap.get(cv.CAP_PROP_POS_FRAMES))  # 현재 프레임 번호
    frame = cv.resize(frame, (640, 360))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w = frame.shape[:2]
        lm = results.pose_landmarks.landmark

        # 코, 엉덩이 중심 좌표 계산
        nose = lm[mp_pose.PoseLandmark.NOSE]
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        nose_xy = np.array([[int(nose.x * w), int(nose.y * h)]], dtype=np.float32)
        hip_xy = np.array([[int((l_hip.x + r_hip.x) / 2 * w), int((l_hip.y + r_hip.y) / 2 * h)]], dtype=np.float32)
        curr_points = np.array([nose_xy[0], hip_xy[0]], dtype=np.float32).reshape(-1, 1, 2)

        if prev_gray is not None and prev_points is not None:
            # Optical Flow 계산 (이전 → 현재 좌표)
            next_points, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)

            if status.sum() == 2:
                prev_nose, prev_hip = prev_points.reshape(-1, 2)
                next_nose, next_hip = next_points.reshape(-1, 2)

                body_height = np.linalg.norm(next_nose - next_hip)
                if body_height == 0:
                    continue

                # 비율 기반 지표 계산
                dy_ratio = (next_nose[1] - next_hip[1]) / body_height
                nose_move_ratio = np.linalg.norm(next_nose - prev_nose) / body_height
                hip_move_ratio = np.linalg.norm(next_hip - prev_hip) / body_height
                total_move = nose_move_ratio + hip_move_ratio

                print(f"[Frame {frame_number}] dy_ratio={dy_ratio:.2f}, move(H/N)={hip_move_ratio:.2f}/{nose_move_ratio:.2f}")

                # ✅ 기준 1: 자세가 낮고 움직임이 클 때 (기존 기준)
                if dy_ratio > -0.5 and (nose_move_ratio > 0.08 or hip_move_ratio > 0.08):
                    print(f"⚠️ [기준1] 낙상 감지 at frame {frame_number}!")
                    save_path = f"detected_falls/fall_frame_{frame_number}_rule1.jpg"
                    cv.imwrite(save_path, frame)
                    print(f"📸 저장 완료: {save_path}")

                # ✅ 기준 2: 급격한 자세 변화 + 연속 큰 움직임 (변화 기반)
                if prev_dy_ratio is not None:
                    delta_dy = abs(dy_ratio - prev_dy_ratio)

                    if delta_dy > 0.6 and total_move > 0.12:
                        consecutive_movement += 1
                    else:
                        consecutive_movement = 0

                    if consecutive_movement >= 2:
                        print(f"⚠️ [기준2] 낙상 감지 at frame {frame_number}!")
                        save_path = f"detected_falls/fall_frame_{frame_number}_rule2.jpg"
                        cv.imwrite(save_path, frame)
                        print(f"📸 저장 완료: {save_path}")
                
                prev_dy_ratio = dy_ratio

        prev_gray = gray.copy()
        prev_points = curr_points.copy()

    if cv.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv.destroyAllWindows()
