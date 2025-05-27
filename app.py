# app.py
import os
import uuid
import datetime
import subprocess
import shutil
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import cv2
import numpy as np
import optical_flow
from optical_flow import detect_fall

app = Flask(__name__, static_folder='static', template_folder='templates')

# 루트 페이지
@app.route('/')
def index():
    return render_template('index.html')

# LIVE 모드 낙상 감지
@app.route('/api/live-detect', methods=['POST'])
def live_detect():
    # Optical Flow 상태 초기화
    optical_flow.prev_gray = None

    # 프레임 수신 및 디코딩
    data = request.data
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify([]), 400

    detected, annotated = detect_fall(frame)
    if detected:
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        fname = f"{uuid.uuid4().hex}.jpg"
        snap_dir = os.path.join(app.root_path, 'static', 'snapshots')
        os.makedirs(snap_dir, exist_ok=True)
        path = os.path.join(snap_dir, fname)
        # imencode + Python I/O로 저장
        ret, buf = cv2.imencode('.jpg', annotated)
        if ret:
            try:
                with open(path, 'wb') as f:
                    f.write(buf.tobytes())
            except Exception as e:
                app.logger.error(f"Snapshot write failed: {path}, error: {e}")
        else:
            app.logger.error(f"Snapshot encode failed: {path}")
        return jsonify([{
            'timestamp': ts,
            'info': '낙상 감지',
            'imageUrl': url_for('static', filename=f'snapshots/{fname}')
        }])
    return jsonify([])

# VIDEO 모드 낙상 감지
@app.route('/api/video-detect', methods=['POST'])
def video_detect():
    # Optical Flow 상태 초기화
    optical_flow.prev_gray = None

    vid = request.files.get('video')
    if not vid:
        return jsonify({'error': 'no file'}), 400

    # 디렉터리 준비
    uploads_root = os.path.join(app.static_folder, 'uploads')
    tmp_dir = os.path.join(uploads_root, 'tmp')
    proc_dir = os.path.join(uploads_root, 'processed')
    snap_dir = os.path.join(app.root_path, 'static', 'snapshots')
    for d in (tmp_dir, proc_dir, snap_dir):
        os.makedirs(d, exist_ok=True)

    # 원본 저장
    orig_ext = os.path.splitext(vid.filename)[1].lower() or '.mp4'
    base = uuid.uuid4().hex
    tmp_path = os.path.join(tmp_dir, f"{base}{orig_ext}")
    vid.save(tmp_path)

    logs = []
    frame_idx = 0
    annotated_input = tmp_path

    # 프레임별 검사 및 어노테이션
    cap = cv2.VideoCapture(tmp_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_mp4 = os.path.join(proc_dir, f"{base}_raw.mp4")
        writer = cv2.VideoWriter(raw_mp4,
                                 cv2.VideoWriter_fourcc(*'avc1'),
                                 fps, (w, h))
        if writer.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detected, vis = detect_fall(frame)
                if detected:
                    ts = str(datetime.timedelta(seconds=frame_idx / fps))[:8]
                    snap_fname = f"{uuid.uuid4().hex}.jpg"
                    snap_path = os.path.join(snap_dir, snap_fname)
                    ret2, buf2 = cv2.imencode('.jpg', vis)
                    if ret2:
                        try:
                            with open(snap_path, 'wb') as f:
                                f.write(buf2.tobytes())
                        except Exception as e:
                            app.logger.error(f"Snapshot write failed: {snap_path}, error: {e}")
                    else:
                        app.logger.error(f"Snapshot encode failed: {snap_path}")
                    logs.append({
                        'timestamp': ts,
                        'info': '낙상 감지',
                        'imageUrl': url_for('static', filename=f'snapshots/{snap_fname}')
                    })
                writer.write(vis)
                frame_idx += 1
            writer.release()
            annotated_input = raw_mp4
        cap.release()

    # H.264 Baseline + faststart 인코딩
    encoded_mp4 = os.path.join(proc_dir, f"{base}.mp4")
    final_filename = os.path.basename(annotated_input)
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', annotated_input,
            '-c:v', 'libx264', '-profile:v', 'baseline', '-level', '3.0',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-movflags', '+faststart',
            encoded_mp4
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        final_filename = os.path.basename(encoded_mp4)
        if annotated_input != tmp_path:
            os.remove(annotated_input)
    except Exception as e:
        app.logger.warning(f"FFmpeg encoding failed: {e}")

    # 임시 파일 정리
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return jsonify({
        'logs': logs,
        'processedVideoUrl': url_for('serve_uploads', filename=f'processed/{final_filename}')
    })

# uploads 폴더 서빙
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(os.path.join(app.static_folder, 'uploads'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
