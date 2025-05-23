# optical_flow.py

import cv2 as cv
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. Lucas-Kanade 옵티컬 플로우 파라미터
lk_params = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 2. 그리드 포인트 생성 간격(px)
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

# ─────────────────────────────────────────────────────────────────────────────
prev_gray = None

def detect_fall(frame: np.ndarray):
    """
    한 프레임 단위로 낙상 여부를 검사합니다.
    Args:
      frame: BGR 컬러 이미지 (np.ndarray)
    Returns:
      detected: bool      → 낙상 감지 여부
      vis: np.ndarray     → 결과를 그려넣은 BGR 이미지
    """
    global prev_gray

    # 1) 그레이스케일 + 블러
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # 2) 초기화
    if prev_gray is None:
        prev_gray = gray.copy()
        return False, frame

    # 3) 특징점(그리드) 생성
    h, w = gray.shape
    p0 = generate_grid_points(w, h, GRID_SPACING)     # shape (N,2)

    # 4) LK 옵티컬 플로우는 3D 좌표계 (N,1,2)를 요구
    p0 = p0.reshape(-1,1,2)
    p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
    if p1 is None:
        prev_gray = gray.copy()
        return False, frame

    # 5) 유효한 매칭만 골라서 (M,1,2) → (M,2) 로 변환
    mask = st.flatten() == 1
    good_old = p0[mask].reshape(-1,2)
    good_new = p1[mask].reshape(-1,2)

    # 6) 움직임 벡터 계산
    mv = good_new - good_old   # now shape (M,2)
    dx, dy = mv[:,0], mv[:,1]
    mag = np.linalg.norm(mv, axis=1)
    ang = np.degrees(np.arctan2(dy, dx))  # -180 ~ 180

    # 7) 1차 필터: 낙상 각도·크기 조건
    cond1 = (ang > -130) & (ang < -30) & (mag > 5.0)
    ang1 = ang[cond1]

    # 8) 2차 필터: 방향 일관성 (클러스터링처럼)
    detected = False
    if ang1.size > 0:
        mean_ang = ang1.mean()
        diff = np.abs(ang1 - mean_ang)
        diff = np.where(diff > 180, 360 - diff, diff)
        if np.count_nonzero(diff < 40) > 5:
            detected = True

    # 9) 시각화
    vis = frame.copy()
    if detected:
        cv.putText(vis, "FALL DETECTED", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    prev_gray = gray.copy()
    return detected, vis
