# app.py
import os
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory, Response, stream_with_context
import numpy as np
import ffmpeg
from fall_detect import (
    push_new_frame,
    reset_global
)


app = Flask(__name__, static_folder='static', template_folder='templates')

ORIGIN_DIR    = os.path.join(app.root_path, 'static', 'uploads', 'origin')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(os.path.join(app.static_folder, 'uploads'), filename)

@app.route('/upload/video', methods=['POST'])
def upload_video():
    file = request.files.get('videoFile')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # 기존 origin 폴더 비우기
    for fname in os.listdir(ORIGIN_DIR):
        os.remove(os.path.join(ORIGIN_DIR, fname))

    # 새 파일 저장
    filename = file.filename
    file.save(os.path.join(ORIGIN_DIR, filename))
    return jsonify({'status': 'uploaded', 'filename': filename}), 200
    
@app.route('/detect/video', methods=['GET', 'POST'])
def detect_video():
    # 1) origin 폴더에 있는 유일한 파일 찾기
    files = os.listdir(ORIGIN_DIR)
    if not files:
        return jsonify({'error': 'No video found'}), 400
    origin_fp = os.path.join(ORIGIN_DIR, files[0])

    # 2) FFmpeg 파이프 방식으로 시작
    process = (
        ffmpeg.input(origin_fp)
            .filter('scale', 1080, 720)
            .filter('format', 'gray')
            .filter('fps', fps=30)
            .output('pipe:1', format='rawvideo', pix_fmt='gray')
            .global_args('-hide_banner','-loglevel','error','-nostats')
            .run_async(pipe_stdout=True)
    )

    def generate():
        idx = 0
        while True:
            in_bytes = process.stdout.read(1080 * 720)
            if len(in_bytes) < 1080 * 720:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape((720, 1080))
            try:
                pred = push_new_frame(idx, frame)
                timestamp = round(idx / 30, 2)
                yield f'data: {{"pred":{pred}, "frame": {idx}, "time": {timestamp}}}\n\n'
            except IndexError:
                pass
            idx += 1
        process.wait()
        reset_global()
        yield 'event: done\ndata: {}\n\n'

    resp = Response(stream_with_context(generate()), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'


    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
