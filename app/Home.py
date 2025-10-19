# Home.py - 메인 페이지

import streamlit as st
from utils.styles import load_common_styles, create_home_header

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
              <li>Test Accuracy: 86.30% · Recall: 57.14% (Random Forest)</li>
              <li>불량 탐지 개선: 기존 대비 탐지 +8건 (47% 감소)</li>
            </ul>
            <strong>재현성·배포</strong>: 노트북(.ipynb)로 재현 가능, Streamlit 데모·모델 pickle 제공
        </div>
        <div class="performance-badge">86.30% Accuracy · Recall 57.14%</div>
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
              <li>Test Accuracy: 98.17% · Recall: 90.26% (GRU)</li>
              <li>거짓 경보 최소화(운영 비용 저감) 및 실시간 적용 가능성 확인</li>
            </ul>
            <strong>재현성·배포</strong>: 학습 스크립트 및 실행 노트북 제공, Streamlit 데모로 시연 가능
        </div>
        <div class="performance-badge">98.17% Accuracy · Recall 90.26%</div>
        <div class="tech-stack">
            <span class="tech-badge">TensorFlow</span>
            <span class="tech-badge">LSTM/GRU</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Docker</span>
        </div>
    </div>
    """, unsafe_allow_html=True)