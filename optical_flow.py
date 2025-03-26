#라이브러리 import
import math
import numpy as np
import cv2 as cv
import argparse
#명령줄에서 영상 파일을 입력받도록 설정 예) python optical_flow.py video.mp4
parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args(args=["C:\\Users\\master\\Desktop\\capstone\\02659_H_A_SY_C8.mp4"])
cap = cv.VideoCapture(args.image)
#Shi-Tomasi 코너 검출기 설정
MAXCORNERS = 100
feature_params = dict( maxCorners = MAXCORNERS,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
#Lucas-Kanade Optical Flow 설정
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
#랜덤 색상 생성
color = np.random.randint(0, 255, (MAXCORNERS, 3))
#첫 번째(t=0) 프레임에서 특징점 검출
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#궤적을 그리기 위한 빈 이미지 생성
mask = np.zeros_like(old_frame)
#창 축소
height, width = old_frame.shape[:2]
scale = 0.3  
new_width = int(width * scale)
new_height = int(height * scale)
cv.namedWindow('frame', cv.WINDOW_NORMAL)  # 창 크기 조절 가능하도록 설정
cv.resizeWindow('frame', new_width, new_height)

#Optical Flow 계산 및 추적
frame_number=1
while(1):
    fall_point_count=0
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #특징점 이동을 계산
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #좋은 특징점 선택 및 궤적 그리기
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    #추적이 성공한(st==1) 특징점만 저장.
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        angle = math.atan2(b - d, a - c)
        speed = np.linalg.norm(new - old)
        position = (a, b)
        if (d-b<0 and speed>20):
            fall_point_count+=1
            print(f"Point {i} - Angle: {angle:.2f}, Speed: {speed:.2f}, Position({a:.2f}, {b:.2f})")
            frame = cv.putText(frame, f"Point {i} - Angle: {angle:.2f}, Speed: {speed:.2f}, Position({a:.2f}, {b:.2f})", (int(a), int(b) - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color[i].tolist(), 1, cv.LINE_AA)
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    print(f"{frame_number}번째 frame 점 개수:{fall_point_count}")
    frame_number+=1
cv.destroyAllWindows()