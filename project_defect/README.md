## 제품 불량 예측 프로젝트

반도체 제조 공정에서 센서 데이터를 활용한 불량품 조기 탐지 시스템

## 프로젝트 요약

| 단계 | 내용 | 결과 |
|------|------|------|
| 01_data_exploration | 데이터 탐색 | 불균형, 약한 상관관계 발견 |
| 02_data_preprocessing | 전처리 + 시간 Feature | 594개 특성 |
| 03_logistic | 베이스라인 | Recall 19.05% |
| 04_random_forest | 비선형 모델 | Recall 57.14% |
| 05_xgboost | 비교 실험 | Recall 38.10% |
| 06_comparison | 최종 선택 | Random Forest |

### 프로젝트 개요
데이터셋: SECOM (반도체 제조 센서 데이터)
샘플 수: 1,567개
특성 수: 594개 (센서 590개 + 시간 4개)
타겟: Pass/Fail (정상 93.4%, 불량 6.6%)
목표: 불량품 조기 탐지로 고객 출하 전 차단

### 주요 성과
## 프로젝트 요약

| 지표 | 베이스라인 | 최종 모델 | 개선 |
|------|------|------|
| Recall | 19.05% | 57.14% | +38.09%p |
| 불량 탐지 | 4개/21개 | 12개/21개 | +8개 |

**최종 선택 모델**: Random Forest (임계값 0.10)

### 기술 스택
언어: Python 3.x
라이브러리: scikit-learn, pandas, numpy, matplotlib, seaborn
모델: Logistic Regression, Random Forest, XGBoost

### 프로젝트 구조
project2_defect_prediction/
├── researching/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_random_forest.ipynb
│   ├── 05_xgboost.ipynb
│   └── 06_model_comparison.ipynb
├── processed_data/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   └── *.pkl (모델 결과)
└── README.md

### 실행 방법
1. 환경 설정
bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
2. 데이터 준비
Kaggle SECOM Dataset 다운로드
secom.csv를 researching/ 폴더에 저장
3. 순서대로 실행
01_data_exploration.ipynb → 02_data_preprocessing.ipynb → 
03~05 (모델 학습) → 06_model_comparison.ipynb

### 주요 분석 내용
1. 데이터 탐색 (01번)
클래스 불균형 확인 (93.4% vs 6.6%)
결측치 4.52%
상관관계 약함 (최고 0.156)
비선형 모델 필요성 파악
2. 전처리 (02번)
결측치 중앙값 대체
상수 특성 116개 제거
시간 Feature 추가 (hour, dayofweek, time_gap)
StandardScaler 적용
최종 특성: 478개
3. 모델 실험 (03~05번)
Logistic Regression: Recall 19.05% (베이스라인)
Random Forest: Recall 57.14% (최적)
XGBoost: Recall 38.10% (과적합)
4. 최종 선택 (06번)
Random Forest 선택
임계값 최적화 (F1 차이 0.05 이내면 Recall 우선)
불량 21개 중 12개 탐지


### 핵심 발견
시간 Feature의 효과
시간 정보 없음: Recall 38.10%
시간 정보 추가: Recall 57.14%
개선: +19.04%p
교대 근무, 설비 가동 패턴 등이 불량에 영향

### 임계값 전략
기본 임계값 (0.5): TP = 0 (불량 하나도 못 찾음)
최적화 (0.10): TP = 12 (57% 탐지)
불균형 데이터에서 임계값 조정 필수

### 모델 선택
XGBoost > RF? → 데이터에 따라 다름
이 데이터: RF가 XGB보다 우수
이유: 샘플 적고 특성 많아 RF의 병렬 학습이 유리

### 한계 및 개선 방향
현재 한계
Recall 57% (목표 60%에 근접하나 미달)
9개 불량품 여전히 놓침
Precision 26% (거짓 경보 많음)

### 개선 방향
- Feature Engineering
- 센서 간 상호작용 변수
- 센서 변화율
- 추가 데이터
- 공정 파라미터
- 원자재 정보

### 실무 적용
AI 1차 스크리닝 + 육안 2차 검사
최종 탐지율 70%+ 예상

### 비즈니스 가치
불량품 조기 발견: 47% 개선 (17개 → 9개 출하)
품질 비용 절감: 리콜, A/S 비용 감소
브랜드 이미지: 고객 불만 감소

### 참고 자료
SECOM Dataset
scikit-learn Documentation
