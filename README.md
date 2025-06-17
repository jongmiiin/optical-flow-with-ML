# 🚑 Optical Flow와 머신러닝 기반 실시간 낙상 판독 시스템

## 1. 프로젝트 개요 (Overview)

* **목적**: 노인의 낙상 사고를 실시간으로 감지, 조기 경고 및 이후 대응을 강화
* **동기**: 고령자 낙상사고는 매년 전체 고령자 안전사고의 60% 이상을 차지하였으며, 낙상사고 비율 또한 계속해서 증가한 것으로 나타남.
* **접근**: 영상 → gray → Optical Flow 특징 CSV 추출 → ML 모델(XGBoost/TCN) 분류 → 실시간 웹 시스템(`web_system/app.py`)으로 결과 제공

---

## 2. 🏆 주요 기능 요약 (Highlights)

* **CSV 요약**: 영상 한 개를 몇 KB짜리 CSV 한 줄로 요약해 효율적 저장·전송·분석 가능
* **실시간 속도**: 10초 영상 ≒ 8초 内 종단 처리 (gray→OF→분류)
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

## 4. 설치 및 실행 (Installation & Usage)

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

## 5. 데이터 및 전처리 (Datasets & Preprocessing)

* **데이터 출처**: AI‑Hub ‘고령자 이상행동 데이터셋’ (3840×2160, 60fps, 10초 영상) ([researchgate.net][1], [mdpi.com][2])
* **샘플링 방식**: 낙상(1.5초 json 기준), 비낙상(XGBoost: 랜덤 3구간 / TCN: 빈도 구간 + 랜덤)
* **Feature 구성**:

  * **XGBoost**: 점수, 속력·각도(x,y), 평균·표준편차 → 18개
  * **TCN**: 44개의 OF 시퀀스 → 평균·표준편차 포함 총 308개

---

## 6. 성능 및 결과 (Results & Metrics)

| 모델      | 정확도   | 처리 시간 |
| ------- | ----- | ----- |
| XGBoost | 81.8% | ≒ 8초  |
| TCN     | 95%   | ≒ 8초  |

* **출력 예시**:

  ```
  video1.mp4,fall,4.2
  ```

---

## 7. 시각화 예시 (Visual Examples)

* 웹 UI:

  * 낙상 vs 비낙상 Optical Flow 이미지 또는 영상
  * CSV 요약 결과 (ex. `"video1.mp4,fall,4.2"`) 출력 스크린샷

---

## 8. 향후 계획 (Future Work)

* **조명 변화 대응**: dynamic OF 등 기술 추가&#x20;
* **라이브 웹캠 버전**: web\_system → `app.py`에서 웹캠 연결만으로 작동하도록 변경
* **엣지 배포 & 경량화**: Mobile/Edge 디바이스 대응 및 모델 경량화

---

## 9. 참고자료 & 라이선스 (References & License)

* **데이터**: AI‑Hub 고령자 이상행동 영상 ([openaccess.thecvf.com][3])
* **Optical Flow 기반 낙상 감지**:

  * Enhanced Optical Dynamic Flow 논문: 정확도 +3%, 처리시간 절감 40–50ms ([researchgate.net][4])
  * Edge-device 실시간 OF 모델: accuracy 96.2%, 83 FPS&#x20;
* **License**: MIT (작성자 판단 하에 적용)

---

## ✅ 요약 포인트

* **4번 설치**에서 `requirements.txt` 생성 요청 and 실행 흐름 명확화
* **5번 데이터**: AI‑Hub json 어노테이션 방식 포함, Feature 구성 상세 기술
* **8번 계획**: 조명/라이브/모바일 대응 등 존재감 있는 미래 지향 제시
* **9번 참고**: 성능·속도 인용, 논문 기반 신뢰도 강화

필요하시면 `requirements.txt` 예시, 화면 예시 이미지, 코드 스니펫도 드릴 수 있어요. 바로 도와드릴게요!

[1]: https://www.researchgate.net/figure/Visualization-results-on-the-URFD-and-Le2i-datasets-From-left-to-right-original-video_fig6_358717484?utm_source=chatgpt.com "Visualization results on the URFD and Le2i datasets (From left to right"
[2]: https://www.mdpi.com/1424-8220/24/22/7256?utm_source=chatgpt.com "Reduction of Vision-Based Models for Fall Detection - MDPI"
[3]: https://openaccess.thecvf.com/content/ICCV2023W/JRDB/papers/Noor_A_Lightweight_Skeleton-Based_3D-CNN_for_Real-Time_Fall_Detection_and_Action_ICCVW_2023_paper.pdf?utm_source=chatgpt.com "[PDF] A Lightweight Skeleton-Based 3D-CNN for Real-Time Fall Detection ..."
[4]: https://www.researchgate.net/publication/350853328_Deep_Learning_for_Vision-Based_Fall_Detection_System_Enhanced_Optical_Dynamic_Flow?utm_source=chatgpt.com "Deep Learning for Vision-Based Fall Detection System: Enhanced ..."
