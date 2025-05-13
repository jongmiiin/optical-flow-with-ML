import numpy as np
import cv2 as cv

cap = cv.VideoCapture('data\ice.mp4')

if not cap.isOpened():
    print("동영상 파일을 열 수 없습니다.")
    exit()

# 비디오 속성 가져오기 (너비, 높이, 프레임 속도)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)


# Lucas-Kanade Optical Flow 파라미터 설정
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 격자 기반 추적 포인트 생성 함수
def generate_grid_points(w, h, spacing):
    points = []
    for y in range(spacing, h - spacing, spacing):
        for x in range(spacing, w - spacing, spacing):
            points.append([[x, y]])
    return np.array(points, dtype=np.float32)

GRID_SPACING = 20


ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)


# 최초 추적 포인트: 격자 생성
p0 = generate_grid_points(frame_width, frame_height, GRID_SPACING)

frame_idx = 0  # 프레임 인덱스 초기화

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Lucas-Kanade 방식으로 Optical Flow 계산
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None:
        break

    # 움직임이 있는 특징점만 선택
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    # Optical Flow 벡터 크기 계산
    motion_vectors = good_new - good_old
    magnitudes = np.linalg.norm(motion_vectors, axis=1)  # 벡터 크기

    
    dx = motion_vectors[:, 0]
    dy = motion_vectors[:, 1]
    angles = np.degrees(np.arctan2(dy, dx))

    # 낙상 판단 기준
    angle_condition = (angles > -130) & (angles < -30)
    speed_condition = magnitudes > 3.0
    fall_candidates = angle_condition & speed_condition
    fall_ratio = np.mean(fall_candidates)

    # 낙상 탐지 시 타임스탬프(초)와 낙상 벡터 출력
    if fall_ratio > 0.4:
        timestamp = frame_idx / fps  # 현재 프레임의 초 단위 시간 계산
        fall_vectors = motion_vectors[fall_candidates]
        fall_positions = good_old[fall_candidates]
        for vec, pos in zip(fall_vectors, fall_positions):
            x, y = pos
            dx, dy = vec
            print(f"[낙상 탐지]\n\t 타임스탬프: {timestamp:.5f}초 \n\t x: {x:.1f}, y: {y:.1f} \n\tdx: {dx:.2f}, dy: {dy:.2f}")
    

    old_gray = frame_gray.copy()
    p0 = generate_grid_points(frame_width, frame_height, GRID_SPACING)  # 매 프레임 리셋
    frame_idx += 1  # 프레임 인덱스 증가

cap.release()
