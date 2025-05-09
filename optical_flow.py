import numpy as np
import cv2 as cv

cap = cv.VideoCapture('data\hospital.mp4')

if not cap.isOpened():
    print("동영상 파일을 열 수 없습니다.")
    exit()

# 비디오 속성 가져오기 (너비, 높이, 프레임 속도)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# 비디오 저장 설정 (코덱, 저장할 파일명 설정)
fourcc = cv.VideoWriter_fourcc(*'XVID')  # 또는 'XVID', mp4v'MJPG' 가능
out = cv.VideoWriter('.\output.avi', fourcc, fps, (frame_width, frame_height))


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

    # 낙상 감지 기준: 평균 속도가 특정 값 이상 + 대부분 아래 방향
    downward_movement = motion_vectors[:, 1] > 1.0  # y방향으로 크게 이동
    
    if magnitudes.mean() > 2.0 and np.mean(downward_movement) > 0.5:
        cv.putText(frame, "Fall Detected!", (50, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        print("낙상 탐지!")

  
    # 궤적 잔상 없이 바로 그리기
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        cv.line(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
        cv.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)
    
    # 비디오 저장
    out.write(frame)

    cv.imshow('Grid-based Lucas-Kanade Optical Flow', frame)
    if cv.waitKey(30) & 0xFF == 27:  # ESC 키 종료
        break

    old_gray = frame_gray.copy()
    p0 = generate_grid_points(frame_width, frame_height, GRID_SPACING)  # 매 프레임 리셋

cap.release()
out.release()  # VideoWriter 객체 해제
cv.destroyAllWindows()
