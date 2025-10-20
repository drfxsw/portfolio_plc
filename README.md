# Manufacturing AI Portfolio

**제조 현장 AI 솔루션 포트폴리오** — 센서 데이터 기반 품질 이상 감지 및 설비 고장 예측 시스템

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://portfolio-plc-dasol.streamlit.app/)

---

## 프로젝트 개요

제조업 현장의 센서 데이터를 활용하여 **품질 이상(불량)을 사전 탐지**하고 **설비 고장을 예측**하는 AI 시스템입니다.  
전통적인 머신러닝(ML)과 딥러닝(DL) 모델을 비교 검증하여 현장 적용 가능한 최적 모델을 선정했습니다.

### 핵심 역량 및 기술 스택
- **데이터 엔지니어링**: Pandas 기반 전처리, 결측치/이상치 처리, 시계열 특성 생성
- **불균형 데이터 처리**: 샘플링(SMOTE), 클래스 가중치, 임계값 최적화
- **모델링**: Random Forest, XGBoost, LSTM, GRU 비교 실험
- **재현성**: Jupyter Notebook 기반 실험 관리, 모델 아티팩트(pickle/json) 제공
- **배포**: Streamlit 기반 웹 데모

---

## 배포 정보

- **Live Demo**: [https://portfolio-plc-dasol.streamlit.app/](https://portfolio-plc-dasol.streamlit.app/)
- **기술**: Streamlit Cloud (무료 호스팅)
- **접근**: 웹 브라우저에서 즉시 확인 가능 (별도 설치 불필요)

---

## 프로젝트 구조

```
portfolio_plc/
├── app/                          # Streamlit 웹 애플리케이션
│   ├── Home.py                   # 메인 페이지 (프로젝트 카드)
│   ├── pages/                    # 개별 프로젝트 상세 페이지
│   │   ├── 1_제조_공정_품질_이상_감지.py
│   │   └── 2_설비_이상_감지.py
│   └── utils/                    # 공통 스타일·유틸리티
│       └── styles.py
├── project_defect/               # 프로젝트 1: 품질 이상 감지 (ML)
│   ├── data/                     # 원본 센서 데이터
│   ├── processed_data/           # 전처리된 데이터 (train/test split)
│   ├── models/                   # 학습된 모델 (.pkl)
│   ├── researching/              # Jupyter 실험 노트북 (01~06)
│   └── README.md                 # 프로젝트별 상세 문서
├── project_failure/              # 프로젝트 2: 설비 이상 감지 (DL)
│   ├── data/                     # 원본 진동 센서 데이터
│   ├── processed_data/           # 전처리된 시계열 데이터
│   ├── models/                   # 학습된 모델 (.h5, .keras)
│   ├── researching/              # Jupyter 실험 노트북 (01~06)
│   └── README.md                 # 프로젝트별 상세 문서
├── tools/                        # 자동화 스크립트
│   ├── run_all_notebooks.py      # 노트북 일괄 실행
│   └── sync_result_to_docs.py    # 결과→문서 자동 동기화
├── requirements.txt              # Python 의존성
└── README.md                     # 본 파일 (전체 프로젝트 소개)
```

---

## 주요 프로젝트

### 1. 제조 공정 품질 이상 감지 (Machine Learning)

**목표**: 반도체 제조 공정에서 센서 데이터 기반으로 불량(품질 이상)을 조기 탐지하여 출하 전 차단

**데이터셋**: SECOM (반도체 센서 1,567개 샘플, 594개 특성)

**핵심 성과**
- **Recall 개선**: 19.05% → 57.14% (+38.09%p) (Random Forest)
- **Test Accuracy**: 86.30%
- **불량 탐지**: 21개 중 12개 성공 탐지 (기존 대비 +8건)

**기술 스택**
- Python, Pandas, Scikit-learn, XGBoost
- Feature Engineering (시간 기반 특성, 파생 변수)
- 불균형 데이터 처리 (SMOTE, 클래스 가중치, 임계값 최적화)

**재현성**
- 실험 노트북 6개 (데이터 탐색 → 전처리 → 모델 비교 → 최종 선택)
- 모델 아티팩트 (pickle), 결과 요약 (JSON)
- Streamlit 데모로 예측 시연

---

### 2. 설비 이상 감지 (Deep Learning)

**목표**: 베어링 진동 센서 데이터 분석을 통한 설비 고장 사전 예측 및 예방 정비 최적화

**데이터셋**: NASA Bearing Dataset (진동 센서 시계열)

**핵심 성과**
- **Test Accuracy**: 98.17%
- **Test Recall**: 90.26% (고장 195개 중 176개 탐지)
- **거짓 경보 최소화**: FP 4개 (LSTM 23개, CNN 43개 대비 최소) → 유지보수 비용 절감

**기술 스택**
- Python, TensorFlow/Keras, LSTM, GRU, CNN
- 시계열 데이터 전처리 (윈도우 슬라이싱, 정규화)
- 모델 경량화 (GRU 파라미터 23% 감소)

**재현성**
- 실험 노트북 6개 (Baseline → LSTM/GRU/CNN 비교 → 최종 선택)
- 학습 스크립트 및 모델 checkpoint 제공
- Streamlit 데모로 예측 시연

---

## 로컬 실행 방법

### 1. 환경 설정
```bash
# 리포지토리 클론
git clone <repository_url>
cd portfolio_plc

# Python 가상환경 생성 (권장)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Streamlit 앱 실행
```bash
streamlit run app/Home.py
```
브라우저에서 http://localhost:8501 접속

---

## 성능 요약

| 프로젝트 | 모델 | Accuracy | Precision | Recall | 비고 |
|---------|------|----------|-----------|--------|------|
| 품질 이상 감지 | Random Forest | 86.30% | 26.09% | 57.14% | 불균형 데이터 처리 최적화 |
| 설비 이상 감지 | GRU | 98.17% | 97.78% | 90.26% | 거짓경보 최소화(FP 4개) |

---

## 향후 계획

### 단기 (1~3개월)
- 추가 실험: 앙상블 모델(Voting/Stacking), AutoML(TPOT) 적용
- 데이터 증강: 센서 시뮬레이션, Synthetic Minority Oversampling
- 모델 경량화: ONNX/TFLite 변환, 엣지 디바이스 배포 검증

### 중기 (3~6개월)
- 프론트엔드 고도화: React/Vue 기반 대시보드, 모니터링 UI
- 백엔드 분리: FastAPI/Flask 기반 REST API 구축, 모델 서빙 최적화
- CI/CD 파이프라인: GitHub Actions 기반 자동 테스트·배포
- 데이터베이스 연동: PostgreSQL/MongoDB 기반 실험 이력 관리

### 장기 (6개월~)
- MLOps 적용: MLflow/Kubeflow 기반 모델 버전 관리 및 A/B 테스트
- 클라우드 배포: AWS/GCP/Azure 기반 Auto-scaling 환경 구축
- 스트림 처리: Kafka/Spark Streaming 기반 센서 데이터 파이프라인
- 설명 가능한 AI: SHAP/LIME 기반 모델 해석성 강화


## References

- **데이터셋**:
  - [SECOM Dataset - Kaggle](https://www.kaggle.com/datasets/paresh2047/uci-semcom)
  - [IMS Bearing Dataset - Kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)
- **프레임워크**: Streamlit, TensorFlow, Scikit-learn, XGBoost
- **배포**: Streamlit Cloud

---

**마지막 업데이트**: 2025년 10월  
**버전**: 1.0.0