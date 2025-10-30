# Home.py - 메인 페이지

import streamlit as st
import pickle
import os
from utils.styles import load_common_styles, create_home_header

# 결과 파일 로드 함수
@st.cache_data
def load_model_results():
    """모델 성능 결과를 로드합니다."""
    results = {}
    
    # 제조 공정 품질 이상 감지 (Random Forest 최고 성능)
    defect_path = os.path.join('..', 'project_defect', 'models', 'results_rf.pkl')
    if os.path.exists(defect_path):
        with open(defect_path, 'rb') as f:
            results['defect'] = pickle.load(f)
    else:
        # 백업 경로
        defect_path = os.path.join('project_defect', 'models', 'results_rf.pkl')
        if os.path.exists(defect_path):
            with open(defect_path, 'rb') as f:
                results['defect'] = pickle.load(f)
    
    # 설비 이상 감지 (GRU 최고 성능) 
    failure_path = os.path.join('..', 'project_failure', 'models', 'results_gru.pkl')
    if os.path.exists(failure_path):
        with open(failure_path, 'rb') as f:
            results['failure'] = pickle.load(f)
    else:
        # 백업 경로
        failure_path = os.path.join('project_failure', 'models', 'results_gru.pkl')
        if os.path.exists(failure_path):
            with open(failure_path, 'rb') as f:
                results['failure'] = pickle.load(f)
    
    return results

# 페이지 설정
st.set_page_config(
    page_title="Manufacturing AI Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# 공통 스타일 로드
load_common_styles()

# 메인 헤더
create_home_header()

# 모델 결과 로드
model_results = load_model_results()

# 성능 지표 추출 (기본값 설정)
if 'defect' in model_results:
    defect_accuracy = model_results['defect'].get('test_accuracy', 0.8630) * 100
    defect_recall = model_results['defect'].get('test_recall', 0.5714) * 100
else:
    defect_accuracy, defect_recall = 86.30, 57.14

if 'failure' in model_results:
    failure_accuracy = model_results['failure'].get('accuracy', 0.9817) * 100
    failure_recall = model_results['failure'].get('recall', 0.9026) * 100
else:
    failure_accuracy, failure_recall = 98.17, 90.26

# 프로젝트 카드들 (수평 정렬)
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown(f"""
    <div class="industrial-card card-ml">
        <div class="card-header">제조 공정 품질 이상 감지</div>
        <div class="card-type type-ml">Machine Learning</div>
        <div class="card-description">
            <strong>목표</strong>: 반도체 센서 데이터로 불량(품질 이상)을 조기 탐지하여 출하 차단<br>
            <strong>핵심 역량</strong>
            <ul>
              <li>데이터 엔지니어링: Pandas 기반 전처리, 결측/이상치 처리, 시계열 특성 추출</li>
              <li>불균형 대응: 샘플링·가중치·임계값 최적화 경험</li>
              <li>모델링: Random Forest / XGBoost 활용한 피처 중요도 분석 및 튜닝</li>
              <li>재현성: 실험 노트북 + 모델 아티팩트(pickle/json) 제공</li>
            </ul>
            <strong>주요 성과</strong>
            <ul>
              <li>Test Accuracy: {defect_accuracy:.2f}% · Recall: {defect_recall:.2f}% (Random Forest)</li>
              <li>불량 탐지 개선: 기존 대비 탐지 +8건 (47% 감소)</li>
            </ul>
            <strong>재현성·배포</strong>: 노트북(.ipynb)로 재현 가능, Streamlit 데모·모델 pickle 제공
        </div>
        <div class="performance-badge">{defect_accuracy:.2f}% Accuracy · Recall {defect_recall:.2f}%</div>
        <div class="tech-stack">
            <span class="tech-badge">Python</span>
            <span class="tech-badge">Pandas</span>
            <span class="tech-badge">Scikit-learn</span>
            <span class="tech-badge">XGBoost</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="industrial-card card-dl">
        <div class="card-header">설비 이상 감지</div>
        <div class="card-type type-dl">Deep Learning</div>
        <div class="card-description">
            <strong>목표</strong>: 베어링 진동 시계열에서 이상 징후 예측, 유지보수 효율화<br>
            <strong>핵심 역량</strong>
            <ul>
              <li>시계열 딥러닝: LSTM / GRU 모델 설계 및 하이퍼파라미터 튜닝</li>
              <li>경량화·서빙: 모델 경량화(파라미터 축소) 및 실시간 추론 고려</li>
              <li>성능분석: Confusion Matrix 기반 FP/FN 밸런스 최적화</li>
            </ul>
            <strong>주요 성과</strong>
            <ul>
              <li>Test Accuracy: {failure_accuracy:.2f}% · Recall: {failure_recall:.2f}% (GRU)</li>
              <li>거짓 경보 최소화(운영 비용 저감) 및 실시간 적용 가능성 확인</li>
            </ul>
            <strong>재현성·배포</strong>: 학습 스크립트 및 실행 노트북 제공, Streamlit 데모로 시연 가능
        </div>
        <div class="performance-badge">{failure_accuracy:.2f}% Accuracy · Recall {failure_recall:.2f}%</div>
        <div class="tech-stack">
            <span class="tech-badge">TensorFlow</span>
            <span class="tech-badge">LSTM/GRU</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Docker</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
col3, col4 = st.columns([1, 1], gap="medium")
with col3:
    st.markdown(f"""
    <div class="industrial-card card-vision">
        <div class="card-header">PCB 외관검사 자동화</div>
        <div class="card-type type-vision">Computer Vision</div>
        <div class="card-description">
            <strong>목표</strong>: 딥러닝 기반 PCB 결함 실시간 검출 (AOI 시스템)<br>
            <strong>핵심 역량</strong>
            <ul>
              <li>객체 검출: YOLOv8 활용한 실시간 결함 위치 특정</li>
              <li>데이터 처리: XML→YOLO 변환, 이미지 정규화, 증강</li>
              <li>모델 최적화: Small→Medium 비교, 정확도/속도 Trade-off</li>
              <li>산업 적용: mAP 기반 성능 평가, 생산라인 배포 고려</li>
            </ul>
            <strong>주요 성과</strong>
            <ul>
              <li>mAP50: 90.3% · Precision: 93.9% · Recall: 82.5%</li>
              <li>추론 속도: 4.9ms/이미지 (실시간 검사 가능)</li>
              <li>치명적 결함 고검출: Missing_hole 96.6%, Short 95.6%</li>
            </ul>
            <strong>재현성·배포</strong>: Google Colab 학습 코드, 모델 파일(.pt), Streamlit 데모
        </div>
        <div class="performance-badge">mAP50 90.3% · 4.9ms 추론</div>
        <div class="tech-stack">
            <span class="tech-badge">YOLOv8</span>
            <span class="tech-badge">PyTorch</span>
            <span class="tech-badge">OpenCV</span>
            <span class="tech-badge">Colab</span>
        </div>
    </div>
    """, unsafe_allow_html=True)