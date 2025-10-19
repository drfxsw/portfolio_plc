## 제조 공정 품질 이상 감지 프로젝트

반도체 제조 공정에서 센서 데이터를 활용한 전통적 머신러닝 기반 품질 이상(불량) 조기 탐지 시스템

## 프로젝트 요약

| 단계 | 내용 | 결과 |
|------|------|------|
| 01_data_exploration | 데이터 탐색 | 불균형, 약한 상관관계 발견 |
| 02_data_preprocessing | 전처리 + 시간 Feature | 594개 특성 |
| 03_logistic | 베이스라인 | Recall 19.05% |
| 04_random_forest | 비선형 모델 | Recall 57.14% |
| 05_xgboost | 비교 실험 | Recall 38.10% |
| 06_comparison | 최종 선택 | Random Forest |

## 프로젝트 개요

**데이터셋**: SECOM (반도체 제조 센서 데이터)
- **샘플 수**: 1,567개
- **특성 수**: 594개 (센서 590개 + 시간 4개)
- **타겟**: Pass/Fail (정상 93.4%, 품질 이상 6.6%)
- **목표**: 품질 이상(불량) 조기 탐지로 고객 출하 전 차단

## 주요 성과

| 지표 | Logistic | Random Forest | XGBoost |
|------|----------|---------------|---------|
| **Accuracy** | 84.39% | **93.04%** | 92.41% |
| **Precision** | 11.11% | **42.86%** | 33.33% |
| **Recall** | 19.05% | **57.14%** | 38.10% |
| **품질 이상 탐지** | 4개/21개 | **12개/21개** | 8개/21개 |

### 핵심 성과
- **품질 이상(불량) 탐지율 3배 개선**: 19.05% → 57.14% (38.09%p 향상)
- **놓침 최소화**: 21개 중 12개 탐지 (9개만 놓침)
- **실용적 성능**: 임계값 조정으로 Recall 최적화

## 기술 스택

### 데이터 처리
- **Pandas, NumPy**: 센서 데이터 전처리, 결측값 처리
- **Scikit-learn**: 데이터 정규화, 분할, 평가

### 머신러닝 모델
- **Scikit-learn**: Logistic Regression, Random Forest, XGBoost
- **불균형 데이터**: SMOTE, 임계값 조정, 계층 분할

### 시각화
- **Matplotlib, Seaborn**: 성능 분석, 특성 중요도
- **한글 지원**: Malgun Gothic 폰트

## 프로젝트 구조

```
project_defect/
├── data/                    # 원본 센서 데이터
├── processed_data/          # 전처리된 데이터
│   ├── X_train.csv         # 훈련 특성 데이터
│   ├── X_test.csv          # 테스트 특성 데이터
│   ├── y_train.csv         # 훈련 라벨
│   ├── y_test.csv          # 테스트 라벨
│   └── scaler.pkl          # 정규화 스케일러
├── models/                  # 학습된 모델
│   ├── model_logistic.pkl  # 로지스틱 회귀
│   ├── model_rf.pkl        # 랜덤 포레스트
│   ├── model_xgb.pkl       # XGBoost
│   └── results_*.pkl       # 성능 메타데이터
└── researching/             # 분석 노트북
    ├── 01_data_exploration.ipynb
    ├── 02_data_preprocessing.ipynb
    ├── 03_logistic_regression.ipynb
    ├── 04_random_forest.ipynb
    ├── 05_xgboost.ipynb
    └── 06_model_comparison.ipynb
```

**특징**: 앙상블 방법으로 높은 일반화 성능과 안정성 확보
## 실무 적용

### 비즈니스 가치
- **품질 보증**: 품질 이상 사전 감지로 고객 신뢰도 향상
- **비용 절감**: 리콜 비용 및 클레임 처리 비용 최소화
- **효율성**: 자동화된 품질 관리 시스템 구축

### 배포 시나리오
1. **인라인 검사**: 제조 라인에서 실시간 품질 이상 감지
2. **품질 관리**: 출하 전 최종 검증 단계 적용
3. **공정 개선**: 품질 이상 패턴 분석을 통한 제조 공정 최적화

## 데이터 소스

**원본 데이터**: [SECOM Dataset - Kaggle](https://www.kaggle.com/datasets/paresh2047/uci-semcom)

## 핵심 발견

### 시간 Feature의 효과
- **시간 정보 없음**: Recall 38.10%
- **시간 정보 추가**: Recall 57.14%
- **개선도**: +19.04%p

### 모델별 특성
- **Logistic Regression**: 선형 관계 한계로 낮은 성능
- **Random Forest**: 비선형 패턴 포착으로 최고 성능
- **XGBoost**: 과적합 경향, 소규모 데이터셋 한계

### 한계 및 개선 방향
현재 한계
Recall 57% (목표 60%에 근접하나 미달)
9개 품질 이상(불량) 여전히 놓침
Precision 26% (거짓 경보 많음)

### 개선 방향
- Feature Engineering
- 센서 간 상호작용 변수
- 센서 변화율
- 추가 데이터
- 공정 파라미터
- 원자재 정보

## 실행 방법
1. 데이터 준비
원본데이터를 project_defect/data 이하에 저장 (secom.csv로 파일명 변경)
2. 순서대로 실행
분석 노트북을 순서대로 실행하여 데이터 탐색, 전처리, 모델 학습 및 평가 수행
3. 결과 확인