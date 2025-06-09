# optical_flow.py (실시간 구조로 전면 수정본)

import cv2 as cv
import numpy as np
import pickle
from pathlib import Path

debug = False

# ───────────────────────────────────────────────────────────────────────────
# 1) Lucas-Kanade 옵티컬 플로우 파라미터 (원본 그대로)
lk_params = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 2) 그리드 간격 (원본 그대로)
GRID_SPACING = 20

# ───────────────────────────────────────────────────────────────────────────
# 실시간 처리를 위한 전역 변수
prev_gray = None            # 이전 그레이스케일 프레임
stats_buffer = []           # 매 프레임쌍당 9개 통계 → 최대 길이 = WINDOW_SIZE-1

# 윈도우 크기 정의 (기본 45)
WINDOW_SIZE = 45

# xgboost 모델 로딩용 (한 번만)
xgb_model = None

# ───────────────────────────────────────────────────────────────────────────
def load_xgb_model():
    """
    xgb 모델을 한 번만 로딩하여 글로벌 변수에 저장.
    파일 경로: 이 파일(optical_flow.py) 기준으로 ../models/xgb_fall_detector.pkl
    """
    global xgb_model
    if xgb_model is None:
        model_path = Path(__file__).parent / "models" / "xgb_fall_detector.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"[ERROR] xgb 모델 파일을 찾을 수 없습니다: {model_path}")
        with open(model_path, 'rb') as f:
            xgb_model = pickle.load(f)

# ───────────────────────────────────────────────────────────────────────────
def generate_grid_points(width: int, height: int, spacing: int):
    """
    영상 전체를 spacing 간격의 격자로 나눠서
    특징점을 생성합니다. (center of each grid cell)
    Returns: np.ndarray of shape (N,2) dtype=float32
    """
    xs = np.arange(spacing // 2, width, spacing, dtype=np.int32)
    ys = np.arange(spacing // 2, height, spacing, dtype=np.int32)
    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return pts.astype(np.float32)

# ───────────────────────────────────────────────────────────────────────────
def compute_single_of_stats(old_gray: np.ndarray, new_gray: np.ndarray):
    """
    두 그레이스케일 프레임(old_gray, new_gray) 사이에서 Optical Flow를 계산하고,
    9개 통계값을 리스트로 반환합니다:
      1) 벡터 개수(count)
      2) vx 평균
      3) vy 평균
      4) vx 표준편차
      5) vy 표준편차
      6) 속도 크기 평균
      7) 속도 크기 표준편차
      8) 각도 평균
      9) 각도 표준편차
    """
    h, w = old_gray.shape
    # 1) 그리드 포인트 생성
    p0 = generate_grid_points(w, h, GRID_SPACING)

    # 2) Optical Flow 계산 (LK)
    p1, st, _ = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
    if p1 is None:
        return [0] * 9

    # 3) 유효 매칭만 골라내기
    good = (st.flatten() == 1)
    good_old = p0[good].reshape(-1, 2)
    good_new = p1[good].reshape(-1, 2)
    mv = good_new - good_old  # shape: (M, 2)
    if debug:
        print(f"[SinglePair] 전체 벡터 수: {mv.shape[0]}")

    # 4) 1차 필터: 방향과 속도 기준
    dx, dy = mv[:, 0], mv[:, 1]
    angs = np.degrees(np.arctan2(dy, dx))
    mask1 = (angs > -130) & (angs < -30) & (np.linalg.norm(mv, axis=1) > 7)
    mv = mv[mask1]
    if debug:
        print(f"[SinglePair] 1차 필터 통과 벡터 수: {mv.shape[0]}")

    # 5) 2차 필터: 각도 분산 기준
    if mv.size:
        ang2 = np.degrees(np.arctan2(mv[:, 1], mv[:, 0]))
        mean_ang = ang2.mean()
        diff = np.abs(ang2 - mean_ang)
        diff = np.where(diff > 180, 360 - diff, diff)
        mv = mv[diff < 50]
        if debug:
            print(f"[SinglePair] 2차 필터 통과 벡터 수: {mv.shape[0]}")

    # 6) 통계 계산
    cnt = mv.shape[0]
    if cnt > 0:
        vx = mv[:, 0]
        vy = mv[:, 1]
        speeds = np.linalg.norm(mv, axis=1)
        ang3 = np.degrees(np.arctan2(vy, vx))
        return [
            cnt,                   # ① 벡터 개수
            float(vx.mean()),      # ② vx 평균
            float(vy.mean()),      # ③ vy 평균
            float(vx.std()),       # ④ vx 표준편차
            float(vy.std()),       # ⑤ vy 표준편차
            float(speeds.mean()),  # ⑥ 속도 크기 평균
            float(speeds.std()),   # ⑦ 속도 크기 표준편차
            float(ang3.mean()),    # ⑧ 각도 평균
            float(ang3.std())      # ⑨ 각도 표준편차
        ]
    else:
        return [0] * 9

# ───────────────────────────────────────────────────────────────────────────
def detect_fall(frame: np.ndarray):
    """
    1) 프레임을 그레이+블러 처리 → gray
    2) prev_gray가 None이면 prev_gray=gray, return False
    3) prev_gray와 현재 gray로 compute_single_of_stats → 9개 stats
    4) stats_buffer에 append, 길이 유지(WINDOW_SIZE-1)
    5) prev_gray=gray 갱신
    6) stats_buffer 길이가 WINDOW_SIZE-1이면 → (9×44) 배열에 대해 컬럼별 mean/std → 18차원 feature 생성 → xgb 예측
    7) detected=True이면 시각화, 반환
    """
    global prev_gray, stats_buffer, xgb_model

    # 1) 그레이스케일 + 블러
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # 2) prev_gray 없으면 최초 세팅
    if prev_gray is None:
        prev_gray = gray.copy()
        return False, frame

    # 3) 한 프레임쌍 통계 계산
    single_stats = compute_single_of_stats(prev_gray, gray)
    stats_buffer.append(single_stats)
    # 4) prev_gray 갱신
    prev_gray = gray.copy()

    # 5) 버퍼 길이 유지: 최대 WINDOW_SIZE-1개(=44)
    if len(stats_buffer) > WINDOW_SIZE - 1:
        stats_buffer.pop(0)

    detected = False
    vis = frame.copy()

    # 6) 버퍼가 충분히 차면(=44개 이상) → XGBoost 예측
    if len(stats_buffer) == WINDOW_SIZE - 1:
        load_xgb_model()
        arr = np.array(stats_buffer)  # shape: (44, 9)
        features = []
        for c in range(arr.shape[1]):
            col = arr[:, c]
            features.append(float(col.mean()))
            features.append(float(col.std()))
        X_test = np.array([features])  # shape: (1, 18)
        y_pred = xgb_model.predict(X_test)
        if y_pred[0] == 1:
            detected = True
            cv.putText(vis, "FALL DETECTED", (50, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return detected, vis
