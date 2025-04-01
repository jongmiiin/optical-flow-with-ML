import numpy as np
import cv2 as cv

cap = cv.VideoCapture('.\skate board.mp4')

if not cap.isOpened():
    print("동영상 파일을 열 수 없습니다.")
    exit()

# 비디오 속성 가져오기 (너비, 높이, 프레임 속도)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

# 비디오 저장 설정 (코덱, 저장할 파일명 설정)
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 또는 'XVID', 'MJPG' 가능
out = cv.VideoWriter('.\output.mp4', fourcc, fps, (frame_width, frame_height))


# Shi-Tomasi 특징점 검출을 위한 파라미터 설정
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Lucas-Kanade Optical Flow 파라미터 설정
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 특징점의 움직임을 나타낼 무작위 색상 생성
color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# 특징점 초기 추출 (Shi-Tomasi 코너 검출)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)

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

    # Optical Flow의 움직임 벡터 시각화
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    img = cv.add(frame, mask)
    
    # 비디오 저장
    out.write(img)

    cv.imshow('Lucas-Kanade Optical Flow', img)
    if cv.waitKey(30) & 0xFF == 27:  # ESC 키 종료
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
out.release()  # VideoWriter 객체 해제
cv.destroyAllWindows()
