import os
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
from pytorch_tcn import TemporalConv1d
from joblib import load


# Constants: video dimensions and frame settings
WIDTH = 1080
HEIGHT = 720
FPS = 30
WINDOW_SIZE = 45  # frames per window (1.5s)
MIN_SPEED = 5   
MAX_SPEED = 300
old_gray = None
fall_sequence = []

# Optical Flow parameters
grid_spacing = 20
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# TCN model definition
class TCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn1 = TemporalConv1d(7,  64, kernel_size=3, dilation=1, padding=0)
        self.tcn2 = TemporalConv1d(64, 64, kernel_size=3, dilation=2, padding=0)
        self.tcn3 = TemporalConv1d(64, 64, kernel_size=3, dilation=4, padding=0)
        self.tcn4 = TemporalConv1d(64, 64, kernel_size=3, dilation=8, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        x = self.relu(self.tcn1(x)); x = self.dropout(x)
        x = self.relu(self.tcn2(x)); x = self.dropout(x)
        x = self.relu(self.tcn3(x)); x = self.dropout(x)
        x = self.relu(self.tcn4(x)); x = self.dropout(x)
        out = x[:, :, -1]
        return self.fc(out).squeeze(-1)
    
def load_model_and_scaler(model_filename: str, scaler_filename: str, device=None):
    # Resolve absolute paths in models/ folder
    base_dir = os.path.dirname(__file__)
    model_path  = os.path.join(base_dir, 'models', model_filename)
    scaler_path = os.path.join(base_dir, 'models', scaler_filename)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    scaler = load(scaler_path)
    return model, scaler, device    

MODEL_FILENAME  = 'tcn_model_state.pth'
SCALER_FILENAME = 'scaler.pkl'
MODEL, SCALER, DEVICE = load_model_and_scaler(MODEL_FILENAME, SCALER_FILENAME)

def generate_grid_points(w, h, spacing):
    """
    Generate a fixed grid of points for LK optical flow.
    """
    offset = spacing // 2
    pts = []
    for y in range(offset, h, spacing):
        for x in range(offset, w, spacing):
            pts.append([[x, y]])
    return np.array(pts, dtype=np.float32)

def push_new_frame(idx, gray, width: int = WIDTH, height: int = HEIGHT):
    global old_gray

        # 2) idx == 0 이면 old_gray 초기화, 아니면 optical flow 계산
    if idx == 0:
        old_gray = gray.copy()
        raise IndexError("First Index")  # 첫 프레임은 비교 대상이 없으니 함수 종료
    else:
        new_gray = gray.copy()

    p0 = generate_grid_points(width, height, grid_spacing)
    p1, st, _ = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    if p1 is None:
       of_features = [0.0]*7
    else:
        st_flat = st.flatten().astype(bool)
        p0_flat = p0.reshape(-1, 2)
        p1_flat = p1.reshape(-1, 2)
        mv_all = p1_flat[st_flat] - p0_flat[st_flat]
        if mv_all.size == 0:
            of_features = [0.0]*7
        else:
            dx, dy = mv_all[:,0], mv_all[:,1]
            sp = np.linalg.norm(mv_all, axis=1)
            mask = (sp > MIN_SPEED) & (sp < MAX_SPEED)
            mv = mv_all[mask]
            if mv.size == 0:
                of_features = [0.0]*7
            else:
                dx_f, dy_f = mv[:,0], mv[:,1]
                sp_f = np.linalg.norm(mv, axis=1)
                ang_f = np.degrees(np.arctan2(dy_f, dx_f))
                cnt = float(len(sp_f))
                mean_sp = float(np.mean(sp_f))
                std_sp  = float(np.std(sp_f))
                ang_mean = float(np.mean(ang_f))
                diff = np.abs(ang_f - ang_mean)
                diff = np.where(diff > 180, 360 - diff, diff)
                ang_std = float(np.sqrt(np.mean(diff**2)))
                sum_vert = float(np.sum(dy_f))
                sum_horiz = float(np.sum(dx_f))
                of_features = ([cnt, mean_sp, std_sp, ang_mean, ang_std, sum_vert, sum_horiz])
    old_gray = new_gray

    return sliding_window(idx, of_features)
    
def sliding_window(idx, of_features):
    global fall_sequence
    fall_sequence.append(of_features)
    if(idx >= WINDOW_SIZE-1):
        fall_sequence=fall_sequence[1:]
        return detect_fall(fall_sequence)
    raise IndexError("There are not 44 frames") 


def detect_fall(fall_sequence):
    # stats.shape == (44, 7)
    # 표준화: 각 프레임별 7개 feature 스케일링
    stats_scaled = SCALER.transform(fall_sequence)  # (44,7)
    # 텐서 변환: (batch=1, channels=7, seq_len=44)
    t = torch.tensor(stats_scaled.T[np.newaxis, ...], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(t)        # shape (1)
        prob = torch.sigmoid(logits).item()
        pred_label = int(prob > 0.5)          # 바로 int 변환
    return pred_label




def load_grayscale_raw(raw_path: str, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """
    Load rawvideo gray file into numpy array of shape (n_frames, height, width).
    """
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}")
    frame_size = width * height
    file_size = os.path.getsize(raw_path)
    n_frames = file_size // frame_size
    data = np.fromfile(raw_path, dtype=np.uint8)
    frames = data.reshape((n_frames, height, width))
    return frames


def compute_of_sequence(gray_frames_45: np.ndarray) -> np.ndarray:
    """
    Compute optical flow statistics for a 45-frame window.
    Returns array shape (44, 7): [cnt, mean_sp, std_sp, ang_mean, ang_std, sum_vert, sum_horiz]
    """
    H, W = gray_frames_45.shape[1], gray_frames_45.shape[2]
    old_gray = gray_frames_45[0]
    p0 = generate_grid_points(W, H, grid_spacing)
    stats = []
    for i in range(1, gray_frames_45.shape[0]):
        frame_gray = gray_frames_45[i]
        p1, st, _ = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is None:
            stats.append([0.0]*7)
        else:
            st_flat = st.flatten().astype(bool)
            p0_flat = p0.reshape(-1, 2)
            p1_flat = p1.reshape(-1, 2)
            mv_all = p1_flat[st_flat] - p0_flat[st_flat]
            if mv_all.size == 0:
                stats.append([0.0]*7)
            else:
                dx, dy = mv_all[:,0], mv_all[:,1]
                sp = np.linalg.norm(mv_all, axis=1)
                mask = (sp > MIN_SPEED) & (sp < MAX_SPEED)
                mv = mv_all[mask]
                if mv.size == 0:
                    stats.append([0.0]*7)
                else:
                    dx_f, dy_f = mv[:,0], mv[:,1]
                    sp_f = np.linalg.norm(mv, axis=1)
                    ang_f = np.degrees(np.arctan2(dy_f, dx_f))
                    cnt = float(len(sp_f))
                    mean_sp = float(np.mean(sp_f))
                    std_sp  = float(np.std(sp_f))
                    ang_mean = float(np.mean(ang_f))
                    diff = np.abs(ang_f - ang_mean)
                    diff = np.where(diff > 180, 360 - diff, diff)
                    ang_std = float(np.sqrt(np.mean(diff**2)))
                    sum_vert = float(np.sum(dy_f))
                    sum_horiz = float(np.sum(dx_f))
                    stats.append([cnt, mean_sp, std_sp, ang_mean, ang_std, sum_vert, sum_horiz])
        old_gray = frame_gray
        p0 = generate_grid_points(W, H, grid_spacing)

    stats_arr = np.array(stats, dtype=np.float32)
    if False:
        # 프레임별 통계 출력
        for i, row in enumerate(stats_arr):
            print(f"Frame {i+1:02d}: cnt={row[0]:.1f}, mean_sp={row[1]:.2f}, … sum_horiz={row[6]:.2f}")
        print("전체 stats shape:", stats_arr.shape)
    return stats_arr


def chunk_sequence_stats(frames: np.ndarray, window_size: int = WINDOW_SIZE) -> list:
    """
    Divide entire grayscale sequence into non-overlapping windows of given size
    and compute optical flow stats for each.
    Returns list of arrays, each shape (44,7).
    """
    offset = 1
    n_frames = frames.shape[0]
    stats_windows = []
    for start in range(0, n_frames-WINDOW_SIZE+2, offset):
            window = frames[start:start+WINDOW_SIZE-1]
            stats_windows.append(window)
    return stats_windows






def detect_falls_in_sequence(frames: np.ndarray, model, scaler, device) -> list:
    """
    Given full frames array, returns predictions for each 45-frame window.
    Predictions: list of float probabilities for each window.
    """
    stats_windows = chunk_sequence_stats(frames)
    preds = []
    for stats in stats_windows:
        # stats.shape == (44, 7)
        # 표준화: 각 프레임별 7개 feature 스케일링
        stats_scaled = scaler.transform(stats)  # (44,7)
        # 텐서 변환: (batch=1, channels=7, seq_len=44)
        t = torch.tensor(stats_scaled.T[np.newaxis, ...], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(t)        # shape (1)
            prob = torch.sigmoid(logits).item()
            pred_label = int(prob > 0.5)          # 바로 int 변환
        preds.append(pred_label)
    return preds