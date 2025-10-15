# Home.py - 메인 페이지

import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="제조 설비 AI 포트폴리오",
    layout="wide"
)

# 제목
st.title("제조 설비 AI 예측 시스템")

# 부제목
st.markdown("### 머신러닝/딥러닝 기반 불량 및 고장 예측")

# 구분선
st.divider()

# 소개
st.markdown("""
### 프로젝트 구성:
""")

# 2개 컬럼
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 1. 불량 예측 (ML)
    - **데이터**: SECOM 반도체 센서
    - **모델**: Logistic, Random Forest, XGBoost
    - **목표**: 제품 불량 조기 탐지
    """)

with col2:
    st.markdown("""
    #### 2. 고장 예측 (DL)
    - **데이터**: NASA 베어링 진동
    - **모델**: LSTM, GRU, 1D CNN
    - **목표**: 설비 고장 사전 예측
    """)