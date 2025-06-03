# optical_flow.py (수정본)

import cv2 as cv
import numpy as np
import pickle
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────
# 1) Lucas-Kanade 옵티컬 플로우 파라미터 (원본 그대로)
lk_params = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 2) 그리드 간격 (Untitled1과 동일)
GRID_SPACING = 20

def generate_grid_points(width: int, height: int, spacing: int):
    """
    영상 전체를 spacing 간격의 격자로 나눠
    특징점을 생성합니다. (center of each grid cell)
    Returns: np.ndarray of shape (N,2) dtype=float32
    """
    xs = np.arange(spacing // 2, width, spacing, dtype=np.int32)
    ys = np.arange(spacing // 2, height, spacing, dtype=np.int32)
    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return pts.astype(np.float32)

# ───────────────────────────────────────────────────────────────────────────
# 3) Untitled1의 특징점 추출 함수(compute_of_stats_from_array) 그대로 복사
def compute_of_stats_from_array(gray_frames: np.ndarray):
    """
    gray_frames: NumPy array shape (N, H, W)
    returns: 18차원 리스트 [mean_cnt, std_cnt, mean_vx, std_vx, ..., std_angle]
    """
    # 1) 첫 프레임과 그리드 생성
    h, w = gray_frames.shape[1:]
    old_gray = gray_frames[0]
    p0 = generate_grid_points(w, h, GRID_SPACING)

    stats_list = []
    for i in range(1, gray_frames.shape[0]):
        frame_gray = gray_frames[i]
        # LK 옵티컬 플로우 계산
        p1, st, _ = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is None:
            stats_list.append([0]*9)
        else:
            # 유효 매칭만 골라냄
            good = (st.flatten() == 1)
            good_old = p0[good].reshape(-1, 2)
            good_new = p1[good].reshape(-1, 2)
            mv = good_new - good_old    # (M, 2)
            if mv.shape[0] > 0:
                vx, vy = mv[:, 0], mv[:, 1]           # 속도 성분 x, y
                sp = np.linalg.norm(mv, axis=1)       # 속도 크기
                ang3 = np.degrees(np.arctan2(vy, vx)) # 방향(각도)
                stats_list.append([
                    mv.shape[0],              # ① 벡터 개수 (count)
                    float(vx.mean()),         # ② 평균 vx
                    float(vy.mean()),         # ③ 평균 vy
                    float(vx.std()),          # ④ vx 표준편차
                    float(vy.std()),          # ⑤ vy 표준편차
                    float(sp.mean()),         # ⑥ 속도 크기 평균
                    float(sp.std()),          # ⑦ 속도 크기 표준편차
                    float(ang3.mean()),       # ⑧ 각도 평균
                    float(ang3.std())         # ⑨ 각도 표준편차
                ])
            else:
                stats_list.append([0]*9)

        # 다음 비교를 위해
        old_gray = frame_gray
        p0 = generate_grid_points(w, h, GRID_SPACING)

    # stats_list: shape ((N-1), 9)
    arr = np.array(stats_list)  # shape ((N-1), 9)
    summary = []
    for c in range(arr.shape[1]):
        summary += [float(arr[:, c].mean()), float(arr[:, c].std())]
    # summary: 길이 18
    return summary

# ───────────────────────────────────────────────────────────────────────────
# 4) 글로벌 변수 선언
WINDOW_SIZE = 45   # Untitled1에서 사용한 프레임 수. 필요에 따라 조정 가능.
gray_buffer = []   # 그레이스케일 프레임을 WINDOW_SIZE개까지 버퍼에 저장
xgb_model = None   # xgboost 모델을 한 번만 로딩

def load_xgb_model():
    """
    xgb 모델을 한 번만 로딩하고 글로벌 변수에 저장
    모델 파일을 프로젝트 내 'models/xgb_fall_detector.pkl' 위치에 두어야 함.
    """
    global xgb_model
    if xgb_model is None:
        # 상대 경로: 이 파일(optical_flow.py) 기준으로 부모 경로/models/ 아래에 .pkl 파일을 둔다
        model_path = Path(__file__).parent / "models" / "xgb_fall_detector.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"[ERROR] xgb 모델 파일을 찾을 수 없습니다: {model_path}")
        with open(model_path, 'rb') as f:
            xgb_model = pickle.load(f)

def detect_fall(frame: np.ndarray):
    """
    1) 프레임을 받아 그레이스케일 + 블러 → gray
    2) gray_buffer에 추가
    3) gray_buffer 길이가 WINDOW_SIZE에 미달하면 일단 TRUE/FALSE 판정 불가 → False 반환
    4) WINDOW_SIZE만큼 쌓이면 compute_of_stats_from_array() 호출 → xgb_model.predict()로 판정
    5) 슬라이딩 윈도우 적용(버퍼에서 맨 앞 프레임 제거)
    6) 결과 시각화 후 반환
    """
    global gray_buffer, xgb_model

    # 1) 그레이스케일 + 블러
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # 2) 그레이스케일 프레임을 버퍼에 추가 (copy한 이미지)
    gray_buffer.append(gray.copy())

    # 3) 버퍼에 충분한 프레임이 쌓였는지 확인
    if len(gray_buffer) < WINDOW_SIZE:
        # 아직 WINDOW_SIZE만큼 쌓이지 않음 → 충분한 데이터 없으므로 낙상 판정 보류
        return False, frame

    # 4) 이제 버퍼에 WINDOW_SIZE 프레임 만큼 존재하므로, 
    #    그 통계 특징을 뽑아내어 모델에 넣을 준비
    #    (a) 모델이 로드되어 있지 않으면 한 번 로드
    load_xgb_model()

    #    (b) 특징 벡터 생성 (길이 18 리스트)
    gray_np = np.stack(gray_buffer, axis=0)  # shape: (WINDOW_SIZE, H, W)
    features = compute_of_stats_from_array(gray_np)  # [f1, f2, ..., f18]

    #    (c) xgboost 모델 예측 (리스트 → 2D 배열로 변환)
    X_test = np.array([features])  # shape (1, 18)
    y_pred = xgb_model.predict(X_test)  # 예: [0] 또는 [1]

    detected = False
    if y_pred[0] == 1:
        detected = True

    # 5) 슬라이딩 윈도우: 버퍼에서 맨 앞(oldest) 프레임 제거
    gray_buffer.pop(0)

    # 6) 결과 시각화
    vis = frame.copy()
    if detected:
        cv.putText(vis, "FALL DETECTED", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return detected, vis
