# 🚑 Optical Flow와 머신러닝 기반 실시간 낙상 판독 시스템

## 1. 프로젝트 개요 (Overview)

* **목적**: 노인의 낙상 사고를 실시간으로 감지, 조기 경고 및 이후 대응을 강화
* **동기**: 고령자 낙상사고는 매년 전체 고령자 안전사고의 60% 이상을 차지하였으며, 낙상사고 비율 또한 계속해서 증가한 것으로 나타남.
* **접근**: 영상 → gray → Optical Flow 계산 → Optical Flow 크기 필터링 → Optical Flow 특징 CSV 추출 → ML 모델(XGBoost/TCN) 분류 → 실시간 웹 시스템(`web_system/app.py`)으로 결과 제공

---

## 2. 🏆 주요 기능 요약 (Highlights)

* **CSV 요약**: 영상 한 개를 몇 KB짜리 CSV 한 줄로 요약해 효율적 저장·전송·분석 가능
* **실시간 속도**: 10초 영상 ≒ 8초 내 종단 처리 (gray 변환 → Optical Flow → 낙상 판독)
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

1. `requirements.txt` 생성 요청
2. 설치 및 실행:

```bash
cd web_system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

* 이후 웹 → 영상 업로드 → 결과(CSV/시각화) 바로 확인 가능

---

## 📊 5. 데이터 및 전처리 (Datasets & Preprocessing)

* **데이터 출처**: AI‑Hub ‘낙상사고 위험동작 영상-센서 쌍 데이터’ (3840×2160, 60fps, 10초 mp4, 낙상 구간 json) (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EB%82%99%EC%83%81%EC%82%AC%EA%B3%A0%20%EC%9C%84%ED%97%98%EB%8F%99%EC%9E%91%20%EC%98%81%EC%83%81-%EC%84%BC%EC%84%9C%20%EC%8C%8D%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=71641)
* **샘플링 방식**: 낙상(1.5초 json 기준), 비낙상(XGBoost: 랜덤 3구간 / 움직임이 가장 많은 구간 + 랜덤 2구간)
* **Feature 구성**:

  * **XGBoost**: 점 개수, 속력,각도,x,y, 평균·표준편차 → 18개
  * **TCN**: 점 개수, 속력,각도,x,y 평균·표준편차 * 44개의 OF 시퀀스 총 308개

---

## 📈 6. 성능 및 결과 (Results & Metrics)

| 모델      | 정확도   | 처리 시간 |
| ------- | ----- | ----- |
| XGBoost | 81.8% | ≒ 8초  |
| TCN     | 95%   | ≒ 8초  |

* **출력 예시**:

  ```
  video1.mp4,fall,4.2
  ```

---

## 🔎 7. 시각화 예시 (Visual Examples)

* 웹 UI:

  * 낙상 vs 비낙상 Optical Flow 이미지 또는 영상
  * CSV 요약 결과 (ex. `"video1.mp4,fall,4.2"`) 출력 스크린샷

---

## 🔧  8. 향후 계획 (Future Work)

* **조명 변화 대응**: dynamic OF 등 기술 추가&#x20;
* **라이브 웹캠 버전**: web\_system → `app.py`에서 웹캠 연결만으로 작동하도록 변경
* **엣지 배포 & 경량화**: Mobile/Edge 디바이스 대응 및 모델 경량화

---

## 9. 참고자료 & 라이선스 (References & License)

* **데이터**: AI‑Hub 고령자 이상행동 영상 ([openaccess.thecvf.com][3])

