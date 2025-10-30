# PCB 외관검사 자동화 프로젝트

딥러닝 기반 인쇄회로기판(PCB) 결함 자동 검출 시스템 (AOI)

## 프로젝트 배경

**산업 현황**
- 반도체/전자 제조 공정에서 PCB 외관검사는 필수
- 기존 육안검사: 검사자 피로도 증가, 일관성 부족
- **AOI(Automated Optical Inspection) 필요성 증대**

**프로젝트 목표**
- 딥러닝 기반 실시간 결함 검출 시스템 구축
- 생산 수율 향상 및 불량률 최소화
- 생산라인 적용 가능한 경량 모델 개발

## 프로젝트 요약

| 단계 | 내용 | 산업 적용 |
|------|------|----------|
| 데이터 분석 | 6가지 결함 유형 정의 | 불량 유형 분류 체계 |
| 전처리 | 이미지 정규화, 라벨링 | 검사 데이터 표준화 |
| 모델 개발 | YOLOv8 객체검출 | 실시간 결함 위치 특정 |
| 성능 최적화 | Small → Medium | 정확도/속도 Trade-off |
| 검증 | mAP50 90.3% 달성 | 양산 적용 가능 수준 |

## 검출 성능 (KPI)

### 최종 모델: YOLOv8m

| 지표 | 목표 | 달성 | 비고 |
|------|------|------|------|
| **정밀도 (Precision)** | 90% 이상 | **93.9%** | 오검출 최소화 |
| **재현율 (Recall)** | 80% 이상 | **82.5%** | 불량 미검출 방지 |
| **mAP50** | 85% 이상 | **90.3%** | 종합 검출 정확도 |
| **추론 속도** | 실시간 | **4.9ms/이미지** | 생산라인 적용 가능 |

### 결함 유형별 검출율

| 결함 유형 | 정의 | 검출율 (mAP50) | 산업 중요도 |
|---------|------|----------------|------------|
| **Missing_hole** | 홀 누락 | 96.6% | 치명적 결함 |
| **Short** | 회로 단락 | 95.6% | 치명적 결함 |
| **Open_circuit** | 회로 단선 | 93.3% | 치명적 결함 |
| **Mouse_bite** | 모서리 결함 | 88.8% | 주요 결함 |
| **Spurious_copper** | 불필요 동박 | 88.3% | 주요 결함 |
| **Spur** | 돌기 | 79.2% | 경미한 결함 |

## 기술 스택

### 비전 AI
- **YOLOv8**: 실시간 객체검출 SOTA 모델
- **Transfer Learning**: COCO 사전학습 → PCB 도메인 전이
- **PyTorch**: 딥러닝 프레임워크

### 데이터 처리
- **OpenCV**: 이미지 전처리, 증강
- **Albumentations**: 데이터 증강 (Blur, CLAHE)
- **XML→YOLO 변환**: 라벨링 포맷 표준화

### 학습 환경
- **Google Colab**: Tesla T4 GPU
- **Mixed Precision (AMP)**: 학습 속도 최적화

### 성능 평가
- **mAP (mean Average Precision)**: 업계 표준 지표
- **Confusion Matrix**: 오분류 패턴 분석
- **PR Curve**: 임계값 최적화

## 산업 적용 방안

### 생산라인 통합 시나리오
```
[카메라] → [Edge Device] → [YOLOv8m] → [불량 판정] → [NG 배출]
   ↓            ↓              ↓            ↓            ↓
 PCB 촬영   전처리 (416px)  결함검출    품질판정     통계집계
```

### 비즈니스 임팩트

1. **수율 향상**
   - 재현율 82.5%: 불량품 유출 최소화
   - 조기 발견: 후공정 불량 비용 절감

2. **검사 효율**
   - 4.9ms/장: 실시간 전수검사 가능
   - 24시간 무인 운영: 인건비 절감

3. **품질 일관성**
   - 정밀도 93.9%: 과검출 최소화
   - 객관적 기준: 검사자 편차 제거

### 현장 배포 고려사항

| 항목 | 요구사항 | 현재 수준 | 비고 |
|------|---------|----------|------|
| 정확도 | mAP50 >85% | **90.3%** |  충족 |
| 속도 | <10ms | **4.9ms** |  충족 |
| 모델 크기 | <100MB | **52MB** |  Edge 배포 가능 |
| 오검출 | <10% | **6.1%** |  충족 |

## 프로젝트 구조
```
project_vision/
├── data/                    # 원본 검사 이미지
│   └── PCB_DATASET/
│       ├── images/          # 693장 (6가지 결함)
│       └── Annotations/     # 바운딩박스 좌표
├── processed_data/          # 전처리 데이터
│   ├── train/ (552장)
│   └── test/ (141장)
├── models/                  # 배포 모델
│   ├── yolov8s_best.pt     # 경량 모델 (22.5MB)
│   └── yolov8m_best.pt     # 정확도 우선 (52MB)
└── researching/             # 연구 개발
    ├── 01_data_exploration.ipynb
    ├── 02_preprocessing.ipynb
    ├── 03_yolov8s.ipynb
    ├── 04_yolov8m.ipynb
    ├── 05_model_comparison.ipynb
    └── results/             # 성능 리포트
```

## 학습 파라미터

| 파라미터 | 값 | 산업 기준 |
|---------|-----|----------|
| Image Size | 416×416 | AOI 표준 해상도 |
| Batch Size | 16 | GPU 메모리 최적화 |
| Epochs | 50 | Early Stopping 적용 |
| Learning Rate | 0.001 (AdamW) | 안정적 수렴 |
| Data Augmentation | Blur, CLAHE | 실환경 변동 대응 |

## 데이터 소스

**원본**: [PCB Defects - Kaggle](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- 693장 PCB 이미지 (3034×1586)
- 6가지 결함 유형, 바운딩박스 라벨링

## 실행 방법

### 1. 환경 설정
```bash
pip install ultralytics opencv-python pandas
```

### 2. 데이터 준비
```bash
# Kaggle 다운로드 → data/PCB_DATASET/
# 02_preprocessing.ipynb 실행 → processed_data/ 생성
```

### 3. 모델 학습 (Colab)
```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
model.train(data='data.yaml', epochs=50, imgsz=416)
```

### 4. 추론 테스트
```python
results = model.predict('test_image.jpg')
results[0].show()  # 결함 시각화
```

## 핵심 성과

**AOI 적용 가능 수준 달성**: mAP50 90.3%  
**실시간 검사 가능**: 4.9ms/이미지  
**치명적 결함 고검출**: Missing_hole 96.6%, Short 95.6%  
**생산라인 배포 준비**: 52MB 경량 모델  

## 관련 기술

- 컴퓨터 비전 (Computer Vision)
- 객체 검출 (Object Detection)
- 품질 관리 (Quality Control)
- AOI (Automated Optical Inspection)
- 딥러닝 (Deep Learning)
- 전이 학습 (Transfer Learning)

## 라이선스

연구 및 포트폴리오 목적