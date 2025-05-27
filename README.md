# Fall Detection Web Application

이 프로젝트는 실시간 웹캠 및 업로드된 비디오 파일에서 Lucas–Kanade 광류(optical flow) 및 학습된 XGBoost 모델을 이용해 낙상을 감지하는 웹 애플리케이션입니다.

## 📦 환경 설정

### Conda 환경 (추천)

프로젝트 루트에 `environment.yml` 파일을 두고, 아래 명령어로 환경을 생성합니다.

```yaml
# environment.yml
name: fall-detect
channels:
  - conda-forge
dependencies:
  - python=3.8
  - flask
  - numpy
  - opencv
  - xgboost
  - ffmpeg
```

```bash
# Conda 환경 생성 및 활성화
conda env create -f environment.yml
conda activate fall-detect
```

### Pip 사용

Conda 대신 `pip`를 사용하려면, 아래 명령으로 직접 설치할 수 있습니다.

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install flask numpy opencv-python xgboost
# FFmpeg는 시스템 패키지로 설치하거나, conda로 설치하세요:
# conda install -c conda-forge ffmpeg
```

## 🛠️ 실행 방법

```bash
# 서버 실행
python app.py
```

서버가 `http://0.0.0.0:5000` 에서 동작하며, 브라우저에서 접속하여 Live 모드 또는 Video 모드를 테스트할 수 있습니다.

## 📂 디렉터리 구조

```
project-root/
├─ app.py                    # Flask 웹 서버
├─ optical_flow.py           # LK optical flow + XGB 모델 감지 로직
├─ models/
│   └─ xgb_fall_detector.pkl  # 사전 학습된 XGBoost 모델
├─ static/
│   ├─ uploads/              # 업로드 및 처리된 비디오 저장
│   └─ snapshots/            # 감지된 프레임 이미지 저장
├─ templates/
│   └─ index.html            # HTML 템플릿
├─ environment.yml           # Conda 환경 정의
└─ README.md                 # 이 문서
```

## 🚀 주요 종속성

* **Flask**: REST API 및 웹 UI 제공
* **NumPy**, **OpenCV**: 비디오 프레임 처리 및 optical flow 계산
* **XGBoost**: 학습된 모델로 낙상 예측
* **FFmpeg**: 처리된 비디오 재인코딩(H.264 baseline + faststart, WebM 변환)

## 🛠️ 커스터마이징 포인트

* `optical_flow.py` 내 **Shi–Tomasi** 및 **LK 파라미터** 조정
* XGBoost 모델 재학습 또는 새로운 피처 추가
* CSS (`static/style.css`) 에서 UI 레이아웃/스크롤 등 스타일 변경

---

Any issues or contributions welcome! Feel free to open an issue or pull request.
