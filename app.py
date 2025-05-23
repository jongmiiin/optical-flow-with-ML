import os
import uuid
import datetime
import subprocess
import shutil
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import cv2
import numpy as np
from optical_flow import detect_fall

app = Flask(__name__, static_folder='static', template_folder='templates')

# 루트 페이지
@app.route('/')
def index():
    return render_template('index.html')

# LIVE 모드 낙상 감지
@app.route('/api/live-detect', methods=['POST'])
def live_detect():
    data = request.data
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify([]), 400

    detected, annotated = detect_fall(frame)
    if detected:
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        fname = f"{uuid.uuid4().hex}.jpg"
        snap_dir = os.path.join(app.static_folder, 'snapshots')
        os.makedirs(snap_dir, exist_ok=True)
        path = os.path.join(snap_dir, fname)
        cv2.imwrite(path, annotated)
        return jsonify([{
            'timestamp': ts,
            'info': '낙상 감지',
            'imageUrl': url_for('static', filename=f'snapshots/{fname}')
        }])
    return jsonify([])

# VIDEO 모드 낙상 감지
@app.route('/api/video-detect', methods=['POST'])
def video_detect():
    vid = request.files.get('video')
    if not vid:
        return jsonify({'error': 'no file'}), 400

    # 디렉터리 준비
    uploads_dir = os.path.join(app.static_folder, 'uploads')
    tmp_dir = os.path.join(uploads_dir, 'tmp')
    proc_dir = os.path.join(uploads_dir, 'processed')
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    # 원본 저장
    orig_ext = os.path.splitext(vid.filename)[1].lower() or '.mp4'
    base = uuid.uuid4().hex
    tmp_path = os.path.join(tmp_dir, f"{base}{orig_ext}")
    vid.save(tmp_path)

    # 프레임별 어노테이션
    raw_mp4 = os.path.join(proc_dir, f"{base}_raw.mp4")
    cap = cv2.VideoCapture(tmp_path)
    annotated_input = tmp_path
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
        if writer.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detected, vis = detect_fall(frame)
                writer.write(vis)
            writer.release()
            annotated_input = raw_mp4
        cap.release()
    else:
        cap.release()

    # H.264 Baseline 인코딩
    encoded_mp4 = os.path.join(proc_dir, f"{base}.mp4")
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', annotated_input,
            '-c:v', 'libx264',
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            encoded_mp4
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(encoded_mp4):
            final_filename = os.path.basename(encoded_mp4)
        else:
            raise RuntimeError("Encoded file not found")
    except Exception as e:
        app.logger.error(f"FFmpeg encoding failed: {e}")
        # 원본 복사
        fallback_path = os.path.join(proc_dir, f"{base}{orig_ext}")
        shutil.copy(tmp_path, fallback_path)
        final_filename = os.path.basename(fallback_path)

    # 임시 파일 정리
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    if os.path.exists(raw_mp4):
        try:
            os.remove(raw_mp4)
        except OSError:
            pass

    # WebM 변환 (선택)
    webm_path = os.path.join(proc_dir, f"{base}.webm")
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', os.path.join(proc_dir, final_filename),
            '-c:v', 'libvpx',
            '-b:v', '1M',
            '-c:a', 'libvorbis',
            webm_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        final_filename = os.path.basename(webm_path)
    except Exception as e:
        app.logger.warning(f"WebM conversion failed: {e}")

    return jsonify({
        'logs': [],
        'processedVideoUrl': url_for('serve_uploads', filename=f'processed/{final_filename}')
    })

# uploads 서빙
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(os.path.join(app.static_folder, 'uploads'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
