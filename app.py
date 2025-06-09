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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/live-reset', methods=['POST'])
def live_reset():
    optical_flow.prev_gray = None
    optical_flow.stats_buffer = []
    return jsonify({'status': 'ok', 'message': 'session reset'}), 200

@app.route('/api/live-detect', methods=['POST'])
def live_detect():
    data = request.data
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify([]), 400

    detected, annotated = detect_fall(frame)
    if detected:
        # 클라이언트가 보낸 videoTime을 가져온다
        video_time = request.args.get('videoTime', default=None, type=str)
        if video_time is not None:
            ts = video_time  # “비디오 내 몇 초”를 해당 timestamp로 사용
        else:
            ts = datetime.datetime.now().strftime('%H:%M:%S')
        fname = f"{uuid.uuid4().hex}.jpg"
        snap_dir = os.path.join(app.root_path, 'static', 'snapshots')
        os.makedirs(snap_dir, exist_ok=True)
        path = os.path.join(snap_dir, fname)
        ret, buf = cv2.imencode('.jpg', annotated)
        if ret:
            with open(path, 'wb') as f:
                f.write(buf.tobytes())
        return jsonify([{
            'timestamp': ts,
            'info': '낙상 감지',
            'imageUrl': url_for('static', filename=f'snapshots/{fname}')
        }])
    return jsonify([])

# ※ /api/video-detect 부분이 있으면 모두 삭제하거나 주석 처리하세요.

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(os.path.join(app.static_folder, 'uploads'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
