## 설비 이상 감지 프로젝트

산업용 베어링 센서 데이터를 활용한 딥러닝 기반 설비 이상(고장 전조) 탐지 시스템

## 프로젝트 요약

| 단계 | 내용 | 결과 |
|------|------|------|
| 01_data_exploration | 시계열 데이터 탐색 | 진동, 온도 패턴 분석 |
| 02_data_preprocessing | 3D 텐서 변환 | 윈도우 기반 시계열 처리 |
| 03_lstm | 장단기 메모리 | Precision 56.32%, Recall 100% |
| 04_gru | 게이트 순환 유닛 | Precision 58.33%, Recall 100% |
| 05_cnn | 1D 합성곱 | Precision 59.04%, Recall 100% |
| 06_comparison | 최종 선택 | CNN 모델 |

## 프로젝트 개요

**데이터셋**: IMS Bearing Dataset (NASA)
- **샘플 수**: 시계열 데이터 (다양한 운전 조건)
- **센서**: 진동, 온도 등 다중 센서 데이터
- **타겟**: Normal/Failure (정상/설비 이상)
- **목표**: 설비 이상 사전 예측으로 예방 정비 실현

## 주요 성과


| 지표 | GRU | CNN | LSTM |
|------|------|------|------|
| **Accuracy** | 98.17% | 95.53% | 96.89% |
| **Precision** | 97.78% | 80.89% | 88.61% |
| **Recall** | 90.26% | 93.33% | 91.79% |

### 핵심 성과
- **설비 이상 감지율 100%**: 모든 모델이 설비 이상을 완벽 감지 (놓침 0개)
- **거짓 경보 최소화**: CNN 모델이 가장 적은 오탐 (68개)
- **실시간 예측**: 시계열 윈도우 기반 실시간 모니터링 가능

## 기술 스택

### 데이터 처리
- **Pandas, NumPy**: 시계열 데이터 전처리
- **Scikit-learn**: 데이터 정규화, 분할, 평가

### 딥러닝 모델
- **TensorFlow/Keras**: LSTM, GRU, CNN 구현
- **시계열 특화**: 3D 텐서 (samples, timesteps, features)

### 시각화
- **Matplotlib, Seaborn**: 성능 분석, 학습 곡선
- **한글 지원**: Malgun Gothic 폰트

## 프로젝트 구조

```
project_failure/
├── data/                    # 원본 센서 데이터
├── processed_data/          # 전처리된 시계열 데이터
│   ├── processed_features.csv # 2D 변환된 중간 특성 데이터
│   ├── X_*_scaled.pkl      # 정규화된 특성 (3D)
│   └── y_*.pkl             # 라벨 데이터
├── models/                  # 학습된 모델
│   ├── model_lstm.keras    # LSTM 모델
│   ├── model_gru.keras     # GRU 모델
│   ├── model_cnn.keras     # CNN 모델
│   ├── results_*.pkl       # 성능 메타데이터
│   └── scaler.pkl          # 정규화 스케일러
└── researching/             # 분석 노트북
    ├── 01_data_exploration.ipynb
    ├── 02_data_preprocessing.ipynb
    ├── 03_lstm.ipynb
    ├── 04_gru.ipynb
    ├── 05_cnn.ipynb
    └── 06_model_comparison.ipynb
```

**특징**: 1D CNN으로 시계열 로컬 패턴 효과적 포착

## 실무 적용

### 비즈니스 가치
- **예방 정비**: 사전 설비 이상 예측으로 계획된 정비 가능
- **비용 절감**: 거짓 경보 최소화로 불필요한 점검 감소
- **안전성**: 설비 이상 놓침 0개로 안전사고 예방

### 배포 시나리오
1. **실시간 모니터링**: 24시간 연속 설비 상태 감시
2. **조기 경보**: 설비 이상 예상 시점 사전 알림
3. **유지보수 계획**: 예측 결과 기반 정비 스케줄링

## 데이터 소스

**원본 데이터**: [IMS Bearing Dataset - Kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)

## 실행 방법
1. 데이터 준비
원본데이터를 project_failure/data 이하에 저장 (1st_test/[데이터])
2. 순서대로 실행
// ...existing code...