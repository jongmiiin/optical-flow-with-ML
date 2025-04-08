import cv2 as cv
import numpy as np
import time

def grid(img, step=16):
  h,w = img.shape[:2]
  idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(int)
  indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
  return indices

def drawFlow(img,flow,indices):
  for x,y in indices:   # 인덱스 순회
    # 각 그리드 인덱스 위치에 점 그리기 ---③
    cv.circle(img, (x,y), 1, (0,255,0), -1)
    # 각 그리드 인덱스에 해당하는 플로우 결과 값 (이동 거리)  ---④
    dx,dy = flow[y, x].astype(int)
    # 각 그리드 인덱스 위치에서 이동한 거리 만큼 선 그리기 ---⑤
    cv.line(img, (x,y), (x+dx, y+dy), (0,255, 0),2, cv.LINE_AA )


prev = None # 이전 프레임 저장 변수


cap = cv.VideoCapture(cv.samples.findFile("test.mp4"))
fps = cap.get(cv.CAP_PROP_FPS)
delay = int(1000 / fps)
frame_time = 1 / fps  # 예: 25fps → 0.04초

cv.namedWindow('OpticalFlow-Farneback', cv.WINDOW_NORMAL)

while(1):
  start = time.time()
  ret,frame = cap.read()
  if not ret:
    print('No frames grabbed!')
    break
  gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
  if prev is None: 
    prev = gray
    indices = grid(prev)
  else:
    # 이전, 이후 프레임으로 옵티컬 플로우 계산 ---⑦
    flow = cv.calcOpticalFlowFarneback(prev,gray,None,\
                0.5,3,15,3,5,1.1,cv.OPTFLOW_FARNEBACK_GAUSSIAN) 
    drawFlow(frame,flow,indices)
    prev = gray
  cv.imshow('OpticalFlow-Farneback', frame)
  elapsed = time.time() - start
  delay = max(1, int((frame_time - elapsed) * 1000))  # ms 단위
  if cv.waitKey(delay) == 27:    # ESC 체크는 빠르게
    break
cap.release()
cv.destroyAllWindows()