# 🚑 Optical Flow와 머신러닝 기반 실시간 낙상 판독 시스템

## 1. 프로젝트 개요 (Overview)

* **목적**: 노인의 낙상 사고를 실시간으로 감지, 조기 경고 및 이후 대응 강화
* **동기**: 고령자 낙상사고는 매년 전체 고령자 안전사고의 60% 이상을 차지하며, 낙상사고 비율도 지속적으로 증가하고 있음

![image](https://github.com/user-attachments/assets/9cedab5e-dcbc-4b2b-a7da-306451c862a0)

* **접근**: 영상 → gray → Optical Flow 계산 → Optical Flow 크기 필터링 → Optical Flow 특징 CSV 추출 → ML 모델(XGBoost/TCN) 분류 → 실시간 웹 시스템(`web_system/app.py`)으로 결과 제공


---

## 2. 🏆 시스템 성능 요약 및 주요 기능 (System Performance & Features)

* **CSV 요약**: 영상 한 개를 단 몇 KB의 CSV 한 줄로 변환
* **실시간 처리 속도**: 3840*2160 60fps mp4 10초 영상을 8초 내 처리 (gray 변환 → Optical Flow → 낙상 판독)
* **모델 정확도**:

  * XGBoost: 81.8%
  * TCN: 95%

---

## 3. 📂 디렉터리 구조 (Project Structure)

```
.
├── web_system/
│   └── app.py              # 웹 기반 실시간 추론
├── _CSV.ipynb              # 영상 → CSV 변환
├── _features.csv           # 변환 결과 피처 데이터
├── _MODEL.ipynb            # CSV 기반 ML 모델 학습
└── zip.ipynb               # AI‑Hub 데이터 압축 해제
```

---

## 4. 🛠 설치 및 실행 (Installation & Usage)

1. `requirements.txt` 생성 가이드:

```txt
flask
opencv-python
numpy
torch
torchvision
xgboost
joblib
scikit-learn
matplotlib
ffmpeg-python
```

2. 설치 및 실행:

```bash
cd web_system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

* 이후 웹 접속 → 영상 업로드 → 결과(CSV/시각화) 확인 가능

---

## 📊 5. 데이터 및 전처리 (Datasets & Preprocessing)

* **데이터 출처**: AI‑Hub ‘낙상사고 위험동작 영상-센서 쌍 데이터’ ([데이터 보기](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EB%82%99%EC%83%81%EC%82%AC%EA%B3%A0%20%EC%9C%84%ED%97%98%EB%8F%99%EC%9E%91%20%EC%98%81%EC%83%81-%EC%84%BC%EC%84%9C%20%EC%8C%8D%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=71641))
* **샘플링 방식**:

  * 낙상: json 기준 1.5초 구간
  * 비낙상:

    * XGBoost: 랜덤 3구간
    * TCN: 움직임이 가장 많은 구간 + 랜덤 2구간
* **Feature 구성**:

  * **XGBoost**: 44개 Optical Flow 시퀀스의 점 개수, 속력, 각도, x/y, 평균/표준편차 요약  → 18개
  * **TCN**: 44개 Optical Flow 시퀀스 × 7개 특징 → 총 308개 (점 개수, 속력, 각도, x/y, 평균/표준편차)

---

## 📈 6. 성능 및 결과 (Results & Metrics)


* **출력 예시**:
* 낙상:
![image](https://github.com/user-attachments/assets/ba82d1f7-1607-45ab-8770-51030c6d12fc)

* 비낙상:
![image](https://github.com/user-attachments/assets/f43e1371-5a16-4618-9196-e8294fe6cfc1)


---

## 🔧 7. 향후 계획 (Future Work)

* **라이브 웹캠 버전**: `web_system/app.py`에서 웹캠 연결만으로 작동하도록 변경
* **조명 변화 대응**: 밝기 보정

---

## 8. 참고자료 & 라이선스 (References & License)

* **데이터**:

  * [AI‑Hub 고령자 이상행동 영상 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EB%82%99%EC%83%81%EC%82%AC%EA%B3%A0%20%EC%9C%84%ED%97%98%EB%8F%99%EC%9E%91%20%EC%98%81%EC%83%81-%EC%84%BC%EC%84%9C%20%EC%8C%8D%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=71641)

* **자료**:

  * [노인 낙상사고 통계 자료 (소비자원)](https://www.kca.go.kr/smartconsumer/sub.do?menukey=7301&mode=view&no=1003725851&searchKeyword=%EB%82%99%EC%83%81)
