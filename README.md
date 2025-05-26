# optical-flow-with-ML
Optical Flow와 머신러닝을 활용한 실시간 낙상 판독 시스템

🐍 파이썬 라이브러리
모듈	역할	Conda 설치	Pip 설치
flask	웹 서버 및 API 엔드포인트	conda install -c conda-forge flask	pip install flask
numpy	배열 연산, optical_flow 내부 계산	conda install -c conda-forge numpy	pip install numpy
opencv-python<br/>(cv2)	영상 입출력·처리·Optical Flow	conda install -c conda-forge opencv	pip install opencv-python
xgboost	학습된 XGB 모델 로딩 및 예측	conda install -c conda-forge xgboost	pip install xgboost

Tip: pickle 과 logging , os, uuid, datetime, subprocess, shutil 등은 파이썬 표준 라이브러리로 별도 설치 불필요합니다.

🛠️ 시스템 의존성
툴	역할	설치 (Conda-forge)
FFmpeg	OpenCV 출력 MP4 → H.264/AAC 재인코딩<br/>WebM 변환용	conda install -c conda-forge ffmpeg
