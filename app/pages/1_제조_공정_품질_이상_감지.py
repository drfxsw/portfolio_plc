# 1_Defect_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import joblib
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import io
import base64
import os
import sys

# 스타일 유틸리티 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.styles import load_common_styles, create_page_header, create_metric_cards, COLORS, CHART_COLORS

# 페이지 설정
st.set_page_config(
    page_title="Defect Prediction System",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# 공통 스타일 로드
load_common_styles()

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정 (상대 경로)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
model_path = os.path.join(project_root, "project_defect", "models")
data_path = os.path.join(project_root, "project_defect", "processed_data")

# 페이지 헤더
create_page_header(
    "Product Defect Detection",
    "반도체 제조 공정 센서 데이터 기반 불량품 조기 탐지 (Machine Learning)"
)

# 탭 생성
tab1, tab2, tab3 = st.tabs(["프로젝트 정보", "성능 분석", "End-to-End 시스템"])

# ========================= TAB 1: 프로젝트 정보 =========================
with tab1:
    st.header("프로젝트 개요")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("프로젝트 목표")
        st.markdown("""
        - **목적**: 반도체 제조 공정에서 불량품 조기 탐지
        - **데이터**: SECOM 센서 데이터 (UCI Repository)
        - **방법**: 머신러닝 모델 비교
        - **핵심**: Recall 최대화 (불량품 놓치지 않기)
        """)
        
        st.subheader("주요 성과")
        st.markdown("""
        - **Recall 개선**: 19.05% → 57.14% (+38.09%p)
        - **불량 탐지**: 21개 중 12개 성공 탐지
        - **최종 모델**: Random Forest
        """)
    
    with col2:
        st.subheader("기술 스택")
        st.markdown("""
        **데이터 처리**
        - Pandas, NumPy: 전처리
        - Scikit-learn: 모델링, 평가
        
        **모델**
        - Logistic Regression (베이스라인)
        - Random Forest (최종 선택)
        - XGBoost (비교 실험)
        
        **시각화**
        - Matplotlib, Seaborn
        - Streamlit (웹 대시보드)
        """)
    
    st.markdown("---")
    
    # 데이터 설명
    st.subheader("데이터 설명")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 샘플 수", "1,567개")
    with col2:
        st.metric("특성 수", "594개")
    with col3:
        st.metric("정상 비율", "93.4%")
    with col4:
        st.metric("불량 비율", "6.6%")
    
    st.markdown("---")
    
    # 전처리 과정
    st.subheader("전처리 과정")
    
    process_steps = {
        "1. 결측값 처리": "중앙값으로 대체 (4.52% 결측)",
        "2. 상수 특성 제거": "116개 상수 특성 삭제",
        "3. 시간 Feature 추가": "hour, dayofweek, time_gap",
        "4. 데이터 정규화": "StandardScaler 적용",
        "5. 최종 특성": "478개 특성으로 축소"
    }
    
    for step, desc in process_steps.items():
        st.markdown(f"**{step}**: {desc}")
    
    st.markdown("---")
    
    # 모델 소개
    st.subheader("모델 소개")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Logistic Regression**
        - 베이스라인 모델
        - 선형 관계 학습
        - 빠른 학습, 해석 용이
        - Recall: 19.05%
        """)
    
    with col2:
        st.markdown("""
        **Random Forest** (최종 선택)
        - 최종 선택 모델
        - 앙상블 방법
        - 비선형 패턴 포착
        - Recall: 57.14%
        """)
    
    with col3:
        st.markdown("""
        **XGBoost**
        - 그래디언트 부스팅
        - 높은 성능 잠재력
        - 소규모 데이터 과적합
        - Recall: 38.10%
        """)

# ========================= TAB 2: 성능 분석 =========================
with tab2:
    st.header("성능 분석")
    
    # 성능 결과 로드
    @st.cache_data
    def load_results():
        try:
            results_logistic = joblib.load(os.path.join(model_path, 'results_logistic.pkl'))
            results_rf = joblib.load(os.path.join(model_path, 'results_rf.pkl')) 
            results_xgb = joblib.load(os.path.join(model_path, 'results_xgb.pkl'))
            return results_logistic, results_rf, results_xgb
        except Exception as e:
            st.error(f"성능 결과 파일을 불러올 수 없습니다: {e}")
            return None, None, None
    
    results_logistic, results_rf, results_xgb = load_results()
    
    if results_logistic and results_rf and results_xgb:
        # 성능 지표 표
        st.subheader("성능 지표 비교")
        
        performance_data = {
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'Accuracy': [
                results_logistic['accuracy'],
                results_rf['accuracy'], 
                results_xgb['accuracy']
            ],
            'Precision': [
                results_logistic['precision'],
                results_rf['precision'],
                results_xgb['precision']
            ],
            'Recall': [
                results_logistic['recall'],
                results_rf['recall'],
                results_xgb['recall']
            ],
            'F1-Score': [
                results_logistic['f1_score'],
                results_rf['f1_score'],
                results_xgb['f1_score']
            ]
        }
        
        df_performance = pd.DataFrame(performance_data)
        
        # 최고 성능 하이라이트
        def highlight_max(s):
            if s.name == 'Model':
                return [''] * len(s)
            return ['background-color: lightgreen' if v == s.max() else '' for v in s]
        
        st.dataframe(
            df_performance.style.apply(highlight_max, axis=0).format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}', 
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # 시각화
        st.subheader("성능 시각화")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 성능 지표 막대 그래프 (Plotly)
            fig = go.Figure()
            
            models = df_performance['Model']
            x = np.arange(len(models))
            
            # 각 지표별 막대 추가
            fig.add_trace(go.Bar(
                name='Accuracy',
                x=models,
                y=df_performance['Accuracy'],
                hovertemplate='<b>Accuracy</b><br>모델: %{x}<br>값: %{y:.4f}<extra></extra>',
                marker_color='lightblue',
                opacity=0.8
            ))
            
            fig.add_trace(go.Bar(
                name='Precision',
                x=models,
                y=df_performance['Precision'],
                hovertemplate='<b>Precision</b><br>모델: %{x}<br>값: %{y:.4f}<extra></extra>',
                marker_color='orange',
                opacity=0.8
            ))
            
            fig.add_trace(go.Bar(
                name='Recall',
                x=models,
                y=df_performance['Recall'],
                hovertemplate='<b>Recall</b><br>모델: %{x}<br>값: %{y:.4f}<extra></extra>',
                marker_color='green',
                opacity=0.8
            ))
            
            fig.add_trace(go.Bar(
                name='F1-Score',
                x=models,
                y=df_performance['F1-Score'],
                hovertemplate='<b>F1-Score</b><br>모델: %{x}<br>값: %{y:.4f}<extra></extra>',
                marker_color='red',
                opacity=0.8
            ))
            
            fig.update_layout(
                title='모델별 성능 지표 비교',
                xaxis_title='모델',
                yaxis_title='Score',
                barmode='group',
                yaxis=dict(range=[0, 1]),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confusion Matrix 비교 (Plotly)
            results_list = [results_logistic, results_rf, results_xgb]
            model_names = ['Logistic', 'RF', 'XGBoost']
            
            # 서브플롯 생성
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=[f'{name} Confusion Matrix' for name in model_names],
                horizontal_spacing=0.1
            )
            
            for i, (result, name) in enumerate(zip(results_list, model_names)):
                cm = result['confusion_matrix']
                
                # Heatmap 추가
                fig.add_trace(
                    go.Heatmap(
                        z=cm,
                        x=['정상', '불량'],
                        y=['정상', '불량'],
                        colorscale='Blues',
                        showscale=(i == 2),  # 마지막에만 컬러바 표시
                        hovertemplate='실제: %{y}<br>예측: %{x}<br>개수: %{z}<extra></extra>',
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 14}
                    ),
                    row=1, col=i+1
                )
            
            # Y축 라벨 설정
            fig.update_yaxes(title_text="실제", row=1, col=1)
            
            # X축 라벨 설정 (가운데에만)
            fig.update_xaxes(title_text="예측", row=1, col=2)
            
            fig.update_layout(
                height=300,
                title_text="",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 분석 및 해석
        st.subheader("분석 및 해석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **핵심 발견**
            
            1. **Random Forest 최고 성능**
               - Recall: 57.14% (목표에 가장 근접)
               - 21개 불량 중 12개 탐지 성공
            
            2. **시간 Feature의 중요성**
               - 시간 정보 추가로 19%p 성능 향상
               - 제조 공정의 시간적 패턴 존재
            
            3. **클래스 불균형 영향**
               - 불량 비율 6.6%로 매우 낮음
               - Precision보다 Recall 우선 최적화
            """)
        
        with col2:
            st.markdown("""
            **한계점**
            
            1. **아직 부족한 Recall**
               - 현재 57.14% (목표 60%)
               - 9개 불량품 여전히 놓침
            
            2. **낮은 Precision**
               - Random Forest 42.86%
               - 많은 거짓 경보 발생
            
            **개선 방안**
            - Feature Engineering 강화
            - 앙상블 모델 적용
            - 추가 도메인 지식 활용
            """)

# ========================= TAB 3: End-to-End 시스템 =========================
with tab3:
    st.header("End-to-End 예측 시스템")
    st.markdown("테스트 데이터를 불러와 불량확률을 예측합니다.")
    
    # 모델 로드
    @st.cache_resource
    def load_models():
        try:
            model_logistic = joblib.load(os.path.join(model_path, 'model_logistic.pkl'))
            model_rf = joblib.load(os.path.join(model_path, 'model_rf.pkl')) 
            model_xgb = joblib.load(os.path.join(model_path, 'model_xgb.pkl'))
            scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
            return model_logistic, model_rf, model_xgb, scaler
        except Exception as e:
            st.error(f"모델을 불러올 수 없습니다: {e}")
            return None, None, None, None
    
    # 실제 데이터 로드
    @st.cache_data
    def load_secom_data():
        try:
            # 전처리된 데이터 로드 (CSV 형식)
            X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
            y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
            
            # numpy array로 변환
            X_test = X_test.values
            y_test = y_test.values.flatten()  # Series로 변환
            
            return X_test, y_test
        except Exception as e:
            st.error(f"전처리된 데이터를 불러올 수 없습니다: {e}")
            return None, None
    
    model_logistic, model_rf, model_xgb, scaler = load_models()
    X_test, y_test = load_secom_data()
    
    if all([model_logistic, model_rf, model_xgb, scaler]) and X_test is not None:
        st.subheader("불량 예측 시뮬레이션")
        # 세션 상태 초기화
        if 'current_sample_idx' not in st.session_state:
            st.session_state.current_sample_idx = None
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'prediction_done' not in st.session_state:
            st.session_state.prediction_done = False
        
        # 데이터 생성 버튼
        col1, col2 = st.columns([3, 1])
        with col1: 
            st.markdown("**Test 데이터에서 한 샘플씩 가져와서 3개 모델로 예측해보기**")
        with col2:
            if st.button("랜덤 센서 데이터 생성", use_container_width=True):
                # 랜덤 샘플 선택
                random_idx = np.random.choice(len(X_test))
                st.session_state.current_sample_idx = random_idx
                st.session_state.current_data = {
                    'X': X_test[random_idx],
                    'y_actual': y_test[random_idx]
                }
                st.session_state.prediction_done = False
                st.rerun()
        
        # 현재 데이터 표시
        if st.session_state.current_data is not None:
            st.markdown("---")
            st.subheader("현재 센서 데이터")
            
            # 데이터 미리보기
            current_X = st.session_state.current_data['X']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 특성 데이터를 표로 표시 (처음 20개만)
                preview_df = pd.DataFrame({
                    'Feature': [f'Feature_{i+1}' for i in range(20)],
                    'Value': current_X[:20]
                })
                st.dataframe(
                    preview_df.style.format({'Value': '{:.4f}'}),
                    use_container_width=True,
                    height=360
                )
                st.caption(f"전체 {len(current_X)}개 특성 중 처음 20개만 표시")
            
            with col2:
                # 센서 데이터 요약 통계 (작은 크기)
                st.markdown(f"""
                <div style="display: grid; gap: 8px;">
                    <div style="background: rgba(102, 126, 234, 0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(102, 126, 234, 0.1);">
                        <div style="font-size: 0.7rem; color: #666; margin-bottom: 2px;">총 특성 수</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #333;">{len(current_X)}개</div>
                    </div>
                    <div style="background: rgba(102, 126, 234, 0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(102, 126, 234, 0.1);">
                        <div style="font-size: 0.7rem; color: #666; margin-bottom: 2px;">평균값</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #333;">{current_X.mean():.4f}</div>
                    </div>
                    <div style="background: rgba(102, 126, 234, 0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(102, 126, 234, 0.1);">
                        <div style="font-size: 0.7rem; color: #666; margin-bottom: 2px;">표준편차</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #333;">{current_X.std():.4f}</div>
                    </div>
                    <div style="background: rgba(102, 126, 234, 0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(102, 126, 234, 0.1);">
                        <div style="font-size: 0.7rem; color: #666; margin-bottom: 2px;">최솟값</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #333;">{current_X.min():.4f}</div>
                    </div>
                    <div style="background: rgba(102, 126, 234, 0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(102, 126, 234, 0.1);">
                        <div style="font-size: 0.7rem; color: #666; margin-bottom: 2px;">최댓값</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #333;">{current_X.max():.4f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # 예측하기 버튼
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("3개 모델로 예측 실행", use_container_width=True):
                    st.session_state.prediction_done = True
                    st.rerun()
            
            # 예측 결과 표시
            if st.session_state.prediction_done:
                st.markdown("---")
                st.subheader("예측 결과")
                
                # 예측 실행
                X_single = current_X.reshape(1, -1)
                
                pred_logistic = model_logistic.predict(X_single)[0]
                pred_rf = model_rf.predict(X_single)[0]
                pred_xgb = model_xgb.predict(X_single)[0]
                
                # 예측 확신도 (해당 예측이 맞을 확률)
                def get_prediction_confidence(model, X, prediction, results):
                    if prediction == 1:  # 불량으로 예측한 경우
                        return results['precision']  # Precision: 불량 예측 중 실제 불량 비율
                    else:  # 정상으로 예측한 경우
                        # NPV (Negative Predictive Value): 정상 예측 중 실제 정상 비율
                        # NPV = TN / (TN + FN)
                        cm = np.array(results['confusion_matrix'])
                        tn, fp, fn, tp = cm.ravel()
                        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                        return npv
                
                # 각 모델의 예측 확신도 계산
                confidence_logistic = get_prediction_confidence(model_logistic, X_single, pred_logistic, results_logistic)
                confidence_rf = get_prediction_confidence(model_rf, X_single, pred_rf, results_rf)
                confidence_xgb = get_prediction_confidence(model_xgb, X_single, pred_xgb, results_xgb)
                
                y_actual = st.session_state.current_data['y_actual']
                
                # 결과 표시
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if y_actual == 1:
                        st.error("**실제 정답**")
                        st.error("불량")
                    else:
                        st.success("**실제 정답**")
                        st.success("정상")
                
                with col2:
                    st.info("**Logistic Regression**")
                    pred_text = "불량" if pred_logistic == 1 else "정상"
                    is_correct = pred_logistic == y_actual
                    if is_correct:
                        st.success(f"{pred_text}")
                        st.success(f"정답 확률: {confidence_logistic:.1%}")
                    else:
                        st.error(f"{pred_text}")
                        st.error(f"정답 확률: {confidence_logistic:.1%}")
                
                with col3:
                    st.info("**Random Forest**")
                    pred_text = "불량" if pred_rf == 1 else "정상"
                    is_correct = pred_rf == y_actual
                    if is_correct:
                        st.success(f"{pred_text}")
                        st.success(f"정답 확률: {confidence_rf:.1%}")
                    else:
                        st.error(f"{pred_text}")
                        st.error(f"정답 확률: {confidence_rf:.1%}")
                
                with col4:
                    st.info("**XGBoost**")
                    pred_text = "불량" if pred_xgb == 1 else "정상"
                    is_correct = pred_xgb == y_actual
                    if is_correct:
                        st.success(f"{pred_text}")
                        st.success(f"정답 확률: {confidence_xgb:.1%}")
                    else:
                        st.error(f"{pred_text}")
                        st.error(f"정답 확률: {confidence_xgb:.1%}")
                
                # 모델 정확도 요약
                st.markdown("---")
                
                correct_models = []
                if pred_logistic == y_actual:
                    correct_models.append("Logistic")
                if pred_rf == y_actual:
                    correct_models.append("Random Forest")
                if pred_xgb == y_actual:
                    correct_models.append("XGBoost")
                
                if len(correct_models) == 3:
                    st.success(f"**모든 모델이 정답!** ({', '.join(correct_models)})")
                elif len(correct_models) > 0:
                    st.warning(f"**{len(correct_models)}개 모델 정답:** {', '.join(correct_models)}")
                else:
                    st.error("**모든 모델이 틀렸습니다!**")
                
                # 확률 비교 차트
                st.subheader("모델별 예측 정답 확률 비교")
                
                models = ['Logistic', 'Random Forest', 'XGBoost']
                probs = [confidence_logistic, confidence_rf, confidence_xgb]
                predictions = [pred_logistic, pred_rf, pred_xgb]
                
                # 막대 색깔 (정답이면 초록, 틀리면 빨강)
                colors = []
                status_text = []
                for pred in predictions:
                    if pred == y_actual:
                        colors.append('lightgreen')
                        status_text.append('정답')
                    else:
                        colors.append('lightcoral')
                        status_text.append('오답')
                
                # Plotly 막대 차트
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=models,
                    y=probs,
                    marker_color=colors,
                    opacity=0.7,
                    text=[f'{prob:.1%}' for prob in probs],
                    textposition='outside',
                    textfont=dict(size=12, color='black'),
                    hovertemplate='<b>%{x}</b><br>정답 확률: %{y:.1%}<br>결과: %{customdata}<extra></extra>',
                    customdata=status_text
                ))
                
                # 우수 기준선 추가
                fig.add_hline(
                    y=0.8,
                    line_dash="dash",
                    line_color="orange",
                    opacity=0.7,
                    annotation_text="우수 기준 (80%)",
                    annotation_position="bottom right"
                )
                
                fig.update_layout(
                    title='모델별 예측 정답 확률 비교',
                    xaxis_title='모델',
                    yaxis_title='예측 정답 확률',
                    yaxis=dict(range=[0, 1], tickformat='.0%'),
                    height=650,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("**버튼을 클릭해서 테스트용 데이터 생성**")

    
    else:
        st.error("필요한 파일들을 확인해주세요:")
        st.code("""
        필요 파일:
        - project_defect/models/model_*.pkl
        - project_defect/processed_data/X_test.csv
        - project_defect/processed_data/y_test.csv
        """)