import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -----------------------
# ✨ 1. Helper Functions
# -----------------------
def extract_optical_flow_features(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"[ERROR] Can't read first frame from {video_path}")
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    features = []
    frame_id = 0

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # 1) 특징점 추출 (이전 프레임 기준)
        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        if p0 is None:
            prev_gray = next_gray
            frame_id += 1
            continue

        # 2) 루카스 카나데 Optical Flow 적용
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None)

        if p1 is None:
            prev_gray = next_gray
            frame_id += 1
            continue

        # 3) 벡터 계산
        good_old = p0[st == 1]
        good_new = p1[st == 1]

        magnitudes = []
        angles = []

        for (old, new) in zip(good_old, good_new):
            x0, y0 = old.ravel()
            x1, y1 = new.ravel()
            dx = x1 - x0
            dy = y1 - y0
            mag = np.sqrt(dx ** 2 + dy ** 2)
            ang = np.arctan2(dy, dx)
            magnitudes.append(mag)
            angles.append(ang)

        if magnitudes:
            avg_mag = np.mean(magnitudes)
            std_mag = np.std(magnitudes)
            avg_ang = np.mean(angles)
            features.append({
                "video_id": video_id,
                "frame_id": frame_id,
                "avg_mag": avg_mag,
                "std_mag": std_mag,
                "avg_ang": avg_ang
            })

        prev_gray = next_gray
        frame_id += 1

    cap.release()
    return features


# -----------------------
# 🌐 2. Main Optical Flow 추출
# -----------------------
VIDEO_DIR = "videos"  # 여기에 여러 개의 .mp4 영상 넣기
OUTPUT_CSV = "optical_flow_features.csv"

all_features = []

for filename in os.listdir(VIDEO_DIR):
    if filename.endswith(".mp4"):
        video_path = os.path.join(VIDEO_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        print(f"\u2705 Processing: {filename}")
        features = extract_optical_flow_features(video_path, video_id)
        all_features.extend(features)

# DataFrame 저장
if all_features:
    df = pd.DataFrame(all_features)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📄 Saved to {OUTPUT_CSV} (total rows: {len(df)})")

    # -----------------------
    # 🔮 3. KMeans Clustering 기반 낙상 분류 시도
    # -----------------------
    print("\n🔄 Running KMeans clustering...")
    X = df[['avg_mag', 'std_mag', 'avg_ang']].values
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
    df['cluster'] = kmeans.labels_

    # 클러스터 기준 재정렬 (낙상 cluster가 평균 속도 큰 쪽)
    cluster_stats = df.groupby('cluster')['avg_mag'].mean()
    fall_cluster = cluster_stats.idxmax()
    df['predicted_label'] = df['cluster'].apply(lambda x: 1 if x == fall_cluster else 0)

    # 저장
    df.to_csv("optical_flow_with_clusters.csv", index=False)
    print("\ud83d\udcc4 Clustering result saved to optical_flow_with_clusters.csv")

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(df['avg_mag'], df['std_mag'], c=df['predicted_label'], cmap='coolwarm', alpha=0.6)
    plt.xlabel("Average Magnitude")
    plt.ylabel("Standard Deviation of Magnitude")
    plt.title("Clustering Result: Fall vs Normal")
    plt.grid(True)
    plt.show()

else:
    print("\n[WARNING] No features extracted. Check your video files.")
