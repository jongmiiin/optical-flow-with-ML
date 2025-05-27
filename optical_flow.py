import numpy as np
import cv2 as cv
import json

cap = cv.VideoCapture('data/ice.mp4')

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
fall_data_list = []  # 낙상 정보 저장 리스트

frame_num = 1 # JSON에 넣을 ID

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
    speed_condition = magnitudes > 5.0
    fall_candidates = angle_condition & speed_condition

    # 낙상 탐지 시 타임스탬프(초)와 낙상 벡터 저장
    timestamp = frame_idx / fps  # 현재 프레임의 초 단위 시간 계산
    fall_vectors = motion_vectors[fall_candidates]
    fall_positions = good_old[fall_candidates]
    fall_angles = angles[fall_candidates]
    
    
    # 방향 유사성 필터링
    # 평균 방향에서 ±40도 이상 벗어난 벡터는 잡음으로 제거
    mean_angle = np.mean(fall_angles)
    angle_diff = np.abs(fall_angles - mean_angle)
    angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)  # 각도 wrap-around
    direction_filter = angle_diff < 40  # ±40도 이내 유지
    
    fall_vectors = fall_vectors[direction_filter]
    fall_positions = fall_positions[direction_filter]
    
    count = len(fall_vectors) # 낙상 벡터 개수 
    
    # 벡터 성분 분리
    vx_f = fall_vectors[:,0] if count>0 else np.array([])
    vy_f = fall_vectors[:,1] if count>0 else np.array([])
    sp_f = np.linalg.norm(fall_vectors, axis=1) if count>0 else np.array([])
    ang_f= np.degrees(np.arctan2(vy_f, vx_f)) if count>0 else np.array([])

    # 통계 계산
    mean_vx    = float(vx_f.mean()) if count>0 else 0.0
    std_vx     = float(vx_f.std(ddof=0)) if count>0 else 0.0
    mean_vy    = float(vy_f.mean()) if count>0 else 0.0
    std_vy     = float(vy_f.std(ddof=0)) if count>0 else 0.0
    mean_speed = float(sp_f.mean()) if count>0 else 0.0
    std_speed  = float(sp_f.std(ddof=0)) if count>0 else 0.0
    mean_ang   = float(ang_f.mean()) if count>0 else 0.0
    std_ang    = float(ang_f.std(ddof=0)) if count>0 else 0.0
    
    # 낙상 시간과 해당 값들을 저장
    fall_data_list.append({
        "frame_num": frame_num,
        "count": count,
        "mean_vx":      round(mean_vx, 2),
        "std_vx":       round(std_vx, 2),
        "mean_vy":      round(mean_vy, 2),
        "std_vy":       round(std_vy, 2),
        "mean_speed":   round(mean_speed, 2),
        "std_speed":    round(std_speed, 2),
        "mean_angle":   round(mean_ang, 2),
        "std_angle":    round(std_ang, 2),
    })

    old_gray = frame_gray.copy()
    p0 = generate_grid_points(frame_width, frame_height, GRID_SPACING)  # 매 프레임 리셋
    frame_num += 1  # 프레임 인덱스 증가

cap.release()


# 결과를 JSON 파일로 저장
with open("fall_detection_results.json", "w", encoding="utf-8") as f:
    json.dump(fall_data_list, f, indent=2, ensure_ascii=False)

print("fall_detection_results.json 파일에 저장완료")
