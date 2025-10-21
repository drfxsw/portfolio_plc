# 2_Failure_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 차트 백엔드 설정  
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
import base64
import os
import time
import sys

# 스타일 유틸리티 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.styles import load_common_styles, create_page_header, create_metric_cards, COLORS, CHART_COLORS

# 페이지 설정
st.set_page_config(
    page_title="설비 이상 감지 시스템",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# 공통 스타일 로드 및 페이지 헤더
load_common_styles()
create_page_header("설비 이상 감지",
                    "베어링 진동데이터 기반 이상 예측 및 예방 정비를 위한 딥러닝 활용 시계열 분석 시스템")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정 (app 기준으로)
model_path = "../project_failure/models/"

# 성능 결과 로드
@st.cache_data
def load_failure_results():
    try:
        results_lstm = joblib.load(os.path.join(model_path, 'results_lstm.pkl'))
        results_gru = joblib.load(os.path.join(model_path, 'results_gru.pkl')) 
        results_cnn = joblib.load(os.path.join(model_path, 'results_cnn.pkl'))
        return results_lstm, results_gru, results_cnn
    except Exception as e:
        st.error(f"성능 결과 파일을 불러올 수 없습니다: {e}")
        return None, None, None

# 결과 로드
results_lstm, results_gru, results_cnn = load_failure_results()

# 메인 타이틀
st.title("설비 이상 감지 시스템")
# 탭 생성
tab1, tab2, tab3 = st.tabs(["프로젝트 정보", "성능 분석", "End-to-End 시스템"])

# ========================= TAB 1: 프로젝트 정보 =========================
with tab1:
    st.header("프로젝트 개요")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("프로젝트 목표")
        st.markdown("""
        - **목적**: 산업 장비의 이상 예측 및 예방 정비
        - **데이터**: NASA의 Bearing 진동 센서 데이터
        - **방법**: 딥러닝 시계열 모델 비교
        - **핵심**: 설비 이상 전 조기 경고 시스템 구축
        """)
        
        st.subheader("주요 성과")
        # 동적 성능 지표 사용
        if results_gru:
            gru_accuracy = results_gru.get('accuracy', 0.9817) * 100
            gru_recall = results_gru.get('recall', 0.9026) * 100
            st.markdown(f"""
        - **최고 성능**: GRU 모델 ({gru_accuracy:.2f}% 정확도)
        - **Recall**: {gru_recall:.2f}% (대부분 설비 이상 사전 탐지)
        - **조기 경고**: 설비 이상 전 미리 감지 가능
        - **실시간 예측**: 연속 센서 데이터 처리
        """)
        else:
            st.markdown("""
        - **최고 성능**: GRU 모델 (98.17% 정확도)
        - **Recall**: 90.26% (대부분 설비 이상 사전 탐지)
        - **조기 경고**: 설비 이상 전 미리 감지 가능
        - **실시간 예측**: 연속 센서 데이터 처리
        """)
    
    with col2:
        st.subheader("기술 스택")
        st.markdown("""
        **딥러닝 프레임워크**
        - TensorFlow/Keras: 모델 개발
        - LSTM, GRU, CNN: 시계열 분석
        
        **데이터 처리**
        - Pandas, NumPy: 전처리
        - Scikit-learn: 정규화, 평가
        
        **시각화**
        - Matplotlib, Seaborn
        - Streamlit, Plotly (웹 대시보드)
        """)
    
    st.markdown("---")
    
    # 데이터 설명
    st.subheader("데이터 설명")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 샘플 수", "20,480개")
    with col2:
        st.metric("특성 수", "32개")
    with col3:
        st.metric("정상 비율", "84.13%")
    with col4:
        st.metric("설비 이상 비율", "15.87%")
    
    st.markdown("---")
    
    # 전처리 과정
    st.subheader("전처리 과정")
    
    process_steps = {
        "1. 시계열 윈도우": "10일 단위로 분할 (과거 10일로 오늘 예측)",
        "2. 센서 특성": "베어링 진동 센서의 다양한 통계적 특성",
        "3. 정규화": "StandardScaler로 평균 0, 분산 1 정규화",
        "4. 레이블 생성": "설비 이상 6일 이내면 이상(1), 아니면 정상(0)",
        "5. 3D 텐서": "(samples, 10, 32) 형태"
    }
    
    for step, desc in process_steps.items():
        st.markdown(f"**{step}**: {desc}")
    
    st.markdown("---")
    
    # 모델 소개
    st.subheader("모델 소개")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 동적 성능 지표 사용
        lstm_accuracy = results_lstm.get('accuracy', 0.9689) * 100 if results_lstm else 96.89
        st.markdown(f"""
        **LSTM (Long Short-Term Memory)**
        - 순환 신경망의 변형
        - 장기 의존성 학습
        - 게이트 메커니즘 활용
        - 정확도: {lstm_accuracy:.2f}%
        """)
    
    with col2:
        # 동적 성능 지표 사용 (이미 위에서 정의됨)
        st.markdown(f"""
        **GRU (Gated Recurrent Unit)** (최종 선택)
        - LSTM의 간소화 버전
        - 빠른 학습 속도
        - 적은 파라미터 수
        - 정확도: {gru_accuracy:.2f}%
        """)
    
    with col3:
        # 동적 성능 지표 사용
        cnn_accuracy = results_cnn.get('accuracy', 0.9553) * 100 if results_cnn else 95.53
        st.markdown(f"""
        **CNN (Convolutional Neural Network)**
        - 1D 컨볼루션 레이어
        - 시계열 지역 패턴 탐지
        - 8개 센서 특성 처리
        - 정확도: {cnn_accuracy:.2f}%
        """)

# ========================= TAB 2: 성능 분석 =========================
with tab2:
    st.header("성능 분석")
    
    if results_lstm and results_gru and results_cnn:
        # 성능 지표 표
        st.subheader("성능 지표 비교")
        
        # F1-Score 계산 (저장되지 않았으므로 계산)
        def calculate_f1(precision, recall):
            if precision + recall == 0:
                return 0
            return 2 * (precision * recall) / (precision + recall)
        
        performance_data = {
            'Model': ['LSTM', 'GRU', 'CNN'],
            'Accuracy': [
                results_lstm['accuracy'],
                results_gru['accuracy'], 
                results_cnn['accuracy']
            ],
            'Precision': [
                results_lstm['precision'],
                results_gru['precision'],
                results_cnn['precision']
            ],
            'Recall': [
                results_lstm['recall'],
                results_gru['recall'],
                results_cnn['recall']
            ],
            'F1-Score': [
                calculate_f1(results_lstm['precision'], results_lstm['recall']),
                calculate_f1(results_gru['precision'], results_gru['recall']),
                calculate_f1(results_cnn['precision'], results_cnn['recall'])
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
            # 성능 지표 막대 그래프
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(df_performance))
            width = 0.2
            
            ax.bar(x - width*1.5, df_performance['Accuracy'], width, label='Accuracy', alpha=0.8)
            ax.bar(x - width/2, df_performance['Precision'], width, label='Precision', alpha=0.8)
            ax.bar(x + width/2, df_performance['Recall'], width, label='Recall', alpha=0.8)
            ax.bar(x + width*1.5, df_performance['F1-Score'], width, label='F1-Score', alpha=0.8)
            
            ax.set_ylabel('Score')
            ax.set_title('모델별 성능 지표 비교', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['LSTM', 'GRU', 'CNN'])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            st.pyplot(fig)
        
        with col2:
            # Confusion Matrix 비교
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            results_list = [results_lstm, results_gru, results_cnn]
            model_names = ['LSTM', 'GRU', 'CNN']
            
            for i, (result, name) in enumerate(zip(results_list, model_names)):
                cm = np.array(result['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['정상', '설비 이상'],
                           yticklabels=['정상', '설비 이상'],
                           ax=axes[i])
                axes[i].set_title(f'{name} Confusion Matrix')
                axes[i].set_xlabel('예측')
                axes[i].set_ylabel('실제')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # 분석 및 해석
        st.subheader("분석 및 해석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **핵심 발견**
            
            1. **GRU 모델 최고 성능**
               - 정확도: {gru_accuracy:.2f}% (3개 모델 중 최고)
               - 정밀도: {results_gru.get('precision', 0.9778) * 100:.2f}% (거짓 경보 최소화)
               - 재현율: {gru_recall:.2f}% (대부분 설비 이상 사전 탐지)
            
            2. **효율적 학습**
               - 23% 적은 파라미터로 우수한 성능
               - 거짓 경보 4개로 운영 효율성 극대화
               - 실제 제조업 현장 적용 가능한 실용적 성능
            
            3. **실무 적용 가치**
               - 거짓 경보 4개로 운영 효율성 극대화
               - 90.26% 재현율로 대부분 설비 이상 사전 탐지
               - 실제 제조업 현장 적용 가능한 실용적 성능
            """)
        
        with col2:
            st.markdown("""
            **성능 비교**
            
            1. **정확도 순위**
               - GRU: {gru_accuracy:.2f}% (1위)
               - LSTM: {lstm_accuracy:.2f}% (2위)
               - CNN: {cnn_accuracy:.2f}% (3위)
            
            2. **정밀도 순위**
               - GRU: {results_gru.get('precision', 0.9778) * 100:.2f}% (1위)
               - LSTM: {results_lstm.get('precision', 0.8861) * 100:.2f}% (2위)
               - CNN: {results_cnn.get('precision', 0.8089) * 100:.2f}% (3위)
            
            3. **False Alarm 비교**
               - GRU: 4개 (최소)
               - LSTM: 23개
               - CNN: 43개
            
            **실제 적용 가치**
            - 조기 경고 시스템으로 활용 가능
            - 예방 정비 계획 수립 지원
            - 장비 가동률 향상 기대
            """)

   # ========================= TAB 3: End-to-End 시스템 =========================
with tab3:
    st.header("End-to-End 설비 이상 감지 시스템")
    st.markdown("가상 베어링 데이터를 생성하고 설비 이상 시점을 예측합니다.")
    
    # 모델 로드 함수
    @st.cache_resource
    def load_models():
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "..", "..")
            model_path = os.path.join(project_root, "project_failure", "models")
            
            model_lstm = keras.models.load_model(os.path.join(model_path, "model_lstm.keras"))
            model_gru = keras.models.load_model(os.path.join(model_path, "model_gru.keras"))
            model_cnn = keras.models.load_model(os.path.join(model_path, "model_cnn.keras"))
            scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
            
            return model_lstm, model_gru, model_cnn, scaler
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")
            return None, None, None, None
    
    model_lstm, model_gru, model_cnn, scaler = load_models()
    
    if all([model_lstm, model_gru, model_cnn, scaler]):
        st.markdown("---")
        
        # 사용자 입력
        st.subheader("시뮬레이션 데이터 설정")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            failure_day = st.slider(
                "설비 이상 발생 시점 (일)",
                min_value=5,
                max_value=30,
                value=25,
                step=1,
                help="이 시점에 설비 이상이 발생합니다"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("랜덤 데이터 생성", type="primary", use_container_width=True)
        
        # 데이터 생성
        if generate_btn:
            # 이전 예측 결과 삭제
            if 'prediction_results' in st.session_state:
                del st.session_state.prediction_results
            
            # 정상/설비 이상 데이터 특성 범위 (실제 데이터 기반)
            normal_ranges = {
                'rms': (0.062242, 0.237261, 0.148823, 0.010883),
                'peak': (0.312, 2.8, 0.673632, 0.125524),
                'std': (0.058207, 0.176177, 0.094817, 0.012779),
                'kurtosis': (-0.060628, 71.572022, 0.752199, 0.964164)
            }
            
            failure_ranges = {
                'rms': (0.143046, 0.59361, 0.173856, 0.026982),
                'peak': (0.562, 5.0, 1.302485, 0.732393),
                'std': (0.085083, 0.579418, 0.129529, 0.03184),
                'kurtosis': (0.277354, 71.579558, 4.618403, 7.498321)
            }
            
            # 가상 데이터 생성 (5분 간격, 설비 이상일까지만)
            np.random.seed(int(time.time()))
            
            measurements_per_hour = 12  # 5분 간격
            measurements_per_day = 288  # 24시간 × 12
            total_measurements = failure_day * measurements_per_day  # 설비 이상일까지만
            features = 8  # ch1-ch2의 4가지 통계 = 8개 특성
            
            synthetic_data = []
            labels = []
            
            # 알람 단계별 목표 설비 이상 비율 설정
            # 설비 이상일 기준 역산
            failure_warning_start_day = max(1, failure_day - 5)  # 위험 알람 시작 (5일간)
            transition_start_day = max(1, failure_warning_start_day - 3)  # 경고 시작 (3일 전)
            caution_start_day = max(1, transition_start_day - 1)  # 주의 시작 (1일 전)
            
            # 설비 이상일까지만 데이터 생성
            for i in range(total_measurements):
                current_day = (i // measurements_per_day) + 1  # 현재 일수 (1부터 시작)
                
                # 각 일별 목표 설비 이상 신호 비율
                if current_day < caution_start_day:
                    # 완전 정상
                    failure_ratio = 0.0
                elif current_day < transition_start_day:
                    # 주의 단계 (30% 설비 이상 신호)
                    failure_ratio = 0.30
                elif current_day < failure_warning_start_day:
                    # 경고 단계 (50% 설비 이상 신호, 3일간)
                    failure_ratio = 0.50
                elif current_day < failure_day:
                    # 위험 단계 (80% 설비 이상 신호, 설비 이상 전날까지)
                    failure_ratio = 0.80
                else:
                    # 설비 이상 당일 (100% 설비 이상 신호)
                    failure_ratio = 1.0
                
                # 확률적으로 정상/설비 이상 데이터 선택
                if np.random.random() < failure_ratio:
                    ranges = failure_ranges
                    label = 1
                else:
                    ranges = normal_ranges
                    label = 0
                
                # 1개 특성 세트 생성 (이 시점의 대표값)
                sample = []
                
                # ch1의 4가지 특성 생성
                for stat_name in ['rms', 'peak', 'std', 'kurtosis']:
                    min_val, max_val, mean, std = ranges[stat_name]
                    value = np.random.normal(mean, std)
                    value = np.clip(value, min_val, max_val)
                    sample.append(value)
                
                # ch2의 4가지 특성 독립적으로 생성
                for stat_name in ['rms', 'peak', 'std', 'kurtosis']:
                    min_val, max_val, mean, std = ranges[stat_name]
                    value = np.random.normal(mean, std)
                    value = np.clip(value, min_val, max_val)
                    sample.append(value)
                
                synthetic_data.append(sample)
                labels.append(label)
            
            synthetic_data = np.array(synthetic_data)  # shape: (failure_day * 288, 8)
            labels = np.array(labels)  # shape: (failure_day * 288,)
            
            # 시계열 윈도우 생성 (10개 타임스텝)
            # 각 시점마다 예측을 위해 슬라이딩 윈도우 생성
            window_size = 10
            
            if len(synthetic_data) >= window_size:
                X_sequences = []
                y_sequences = []
                pred_time_indices = []  # 각 예측이 어느 시점인지 기록
                
                for i in range(window_size, len(synthetic_data)):
                    # i번째 시점 예측 = (i-10)~(i-1) 시점 데이터 사용
                    X_sequences.append(synthetic_data[i-window_size:i])
                    y_sequences.append(labels[i])
                    pred_time_indices.append(i)  # 시점 인덱스
                
                X_sequences = np.array(X_sequences)  # shape: (N, 10, 8)
                y_sequences = np.array(y_sequences)  # shape: (N,)
                pred_time_indices = np.array(pred_time_indices)
            else:
                st.error("데이터가 부족합니다. 최소 10개 시점이 필요합니다.")
                X_sequences = np.array([])
                y_sequences = np.array([])
                pred_time_indices = np.array([])
            
            # 정규화
            if len(X_sequences) > 0:
                X_2d = X_sequences.reshape(-1, features)
                X_scaled_2d = scaler.transform(X_2d)
                X_scaled = X_scaled_2d.reshape(X_sequences.shape)
                
                # 세션에 저장 (예측은 아직 안함)
                st.session_state.simulation_data = {
                    'synthetic_data': synthetic_data,
                    'labels': labels,
                    'X_scaled': X_scaled,
                    'y_sequences': y_sequences,
                    'pred_time_indices': pred_time_indices,
                    'failure_day': failure_day,
                    'failure_warning_start_day': failure_warning_start_day,
                    'caution_start_day': caution_start_day,
                    'transition_start_day': transition_start_day,
                    'measurements_per_day': measurements_per_day
                }
                
                st.info(f"{failure_day}일에 설비 이상 발생 (장비 정지), {failure_warning_start_day}일부터 '위험' 알람 시작")
            else:
                st.error("시퀀스 생성 실패")
        
        # 예측하기 버튼 (데이터가 생성된 경우에만 표시)
        if 'simulation_data' in st.session_state:
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                predict_btn = st.button("3개 모델로 예측 실행", use_container_width=True)
            
            # 예측 실행
            if predict_btn:
                data = st.session_state.simulation_data
                
                with st.spinner("예측 중..."):
                    # 모델 예측
                    pred_lstm = (model_lstm.predict(data['X_scaled'], verbose=0) > 0.5).astype(int).flatten()
                    pred_gru = (model_gru.predict(data['X_scaled'], verbose=0) > 0.5).astype(int).flatten()
                    pred_cnn = (model_cnn.predict(data['X_scaled'], verbose=0) > 0.5).astype(int).flatten()
                    
                    # 예측 결과 저장
                    st.session_state.prediction_results = {
                        'pred_lstm': pred_lstm,
                        'pred_gru': pred_gru,
                        'pred_cnn': pred_cnn
                    }
                
        
        # 결과 표시 (예측이 완료된 경우에만)
        if 'simulation_data' in st.session_state and 'prediction_results' in st.session_state:
            data = st.session_state.simulation_data
            predictions = st.session_state.prediction_results
            
            st.markdown("---")
            st.subheader("2. 모델 예측 결과")
            
            # 일별 알람 레벨 계산 함수
            def calculate_daily_alarm_levels(predictions, time_indices, measurements_per_day, total_days):
                """일별 설비 이상 신호 비율로 알람 레벨 계산"""
                daily_levels = {}
                
                for day in range(total_days):
                    # 해당 일의 측정값 필터링
                    day_start = day * measurements_per_day
                    day_end = (day + 1) * measurements_per_day
                    
                    # 해당 일에 속하는 예측 찾기
                    day_mask = (time_indices >= day_start) & (time_indices < day_end)
                    day_predictions = predictions[day_mask]
                    
                    if len(day_predictions) > 0:
                        failure_ratio = np.mean(day_predictions)
                        
                        # 알람 레벨 판정
                        if failure_ratio >= 0.80:
                            level = "위험"
                            color = "red"
                        elif failure_ratio >= 0.50:
                            level = "경고"
                            color = "orange"
                        elif failure_ratio >= 0.30:
                            level = "주의"
                            color = "yellow"
                        else:
                            level = "정상"
                            color = "green"
                        
                        daily_levels[day + 1] = {
                            'level': level,
                            'ratio': failure_ratio,
                            'color': color
                        }
                
                return daily_levels
            
            # 각 모델의 일별 알람 레벨 계산
            total_days = 30  # 전체 30일
            levels_lstm = calculate_daily_alarm_levels(
                predictions['pred_lstm'], data['pred_time_indices'], 
                data['measurements_per_day'], total_days
            )
            levels_gru = calculate_daily_alarm_levels(
                predictions['pred_gru'], data['pred_time_indices'],
                data['measurements_per_day'], total_days
            )
            levels_cnn = calculate_daily_alarm_levels(
                predictions['pred_cnn'], data['pred_time_indices'],
                data['measurements_per_day'], total_days
            )
            
            # 레벨별 날짜 그룹화 함수
            def group_days_by_level(daily_levels):
                """레벨별로 날짜를 그룹화"""
                level_days = {
                    "주의": [],
                    "경고": [],
                    "위험": []
                }
                
                for day in sorted(daily_levels.keys()):
                    level = daily_levels[day]['level']
                    if level in level_days:
                        level_days[level].append(day)
                
                return level_days
            
            days_lstm = group_days_by_level(levels_lstm)
            days_gru = group_days_by_level(levels_gru)
            days_cnn = group_days_by_level(levels_cnn)
            
            # 결과 테이블
            col1, col2, col3 = st.columns(3)
            
            models = [
                ("LSTM", days_lstm),
                ("GRU", days_gru),
                ("CNN", days_cnn)
            ]
            
            for col, (model_name, level_days) in zip([col1, col2, col3], models):
                with col:
                    st.markdown(f"### {model_name}")
                    
                    st.markdown("**알람 발생 일자**")
                    
                    # 주의
                    if level_days["주의"]:
                        days_str = ", ".join(map(str, level_days["주의"]))
                        st.markdown(f"**주의**: {days_str}일")
                    else:
                        st.markdown(f"**주의**: 없음")
                    
                    # 경고
                    if level_days["경고"]:
                        days_str = ", ".join(map(str, level_days["경고"]))
                        st.markdown(f"**경고**: {days_str}일")
                    else:
                        st.markdown(f"**경고**: 없음")
                    
                    # 위험
                    if level_days["위험"]:
                        days_str = ", ".join(map(str, level_days["위험"]))
                        st.markdown(f"**위험**: {days_str}일")
                    else:
                        st.markdown(f"**위험**: 없음")
            
            st.markdown("---")
            
            # 시계열 예측 시각화
            st.subheader("3. 시계열 예측 비교")
            
            # 일별 평균으로 집계 (30개 포인트)
            total_days = 30
            measurements_per_day = data['measurements_per_day']
            
            models_pred = [
                ("LSTM", predictions['pred_lstm'], levels_lstm),
                ("GRU", predictions['pred_gru'], levels_gru),
                ("CNN", predictions['pred_cnn'], levels_cnn)
            ]
            
            # Plotly 서브플롯 생성
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=[f"{model[0]} 모델 - 일별 예측" for model in models_pred],
                vertical_spacing=0.08,
                shared_xaxes=True
            )
            
            # 범례용 invisible traces 추가 (첫 번째 서브플롯에만)
            if True:  # 항상 실행
                # 설비 이상 발생일 범례
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='lines',
                        name='설비 이상 발생일',
                        line=dict(color='red', dash='dash', width=2),
                        showlegend=True,
                        legendgroup='failure_day'
                    ),
                    row=1, col=1
                )
                
                # 설비 이상 징후 구간 범례
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='lines',
                        name='설비 이상 징후 구간',
                        line=dict(color='pink', width=10),
                        opacity=0.3,
                        showlegend=True,
                        legendgroup='failure_warning'
                    ),
                    row=1, col=1
                )
            
            for i, (model_name, predictions_model, daily_levels) in enumerate(models_pred):
                row = i + 1
                
                # X축: 1~30일
                days = list(range(1, total_days + 1))
                
                # 일별 평균 예측값 계산
                daily_predictions = []
                for day in days:
                    if day in daily_levels:
                        daily_predictions.append(daily_levels[day]['ratio'])
                    else:
                        daily_predictions.append(0)  # 데이터 없는 날은 0
                
                # 일별 실제 라벨 계산
                daily_actual = []
                for day in days:
                    if day < data['caution_start_day']:
                        daily_actual.append(0)  # 정상
                    elif day < data['transition_start_day']:
                        daily_actual.append(0.30)  # 주의 (30%)
                    elif day < data['failure_warning_start_day']:
                        daily_actual.append(0.60)  # 경고 (60%)
                    elif day < data['failure_day']:
                        daily_actual.append(0.80)  # 위험 (80%)
                    elif day == data['failure_day']:
                        daily_actual.append(1.0)  # 설비 이상 (100%)
                    else:
                        daily_actual.append(np.nan)  # 데이터 없음
                
                # 실제 라벨 (설비 이상일까지만)
                valid_days = [d for d in days if d <= data['failure_day']]
                valid_actual = [daily_actual[j] for j in range(len(days)) if days[j] <= data['failure_day']]
                
                # 실제 라벨 추가
                fig.add_trace(
                    go.Scatter(
                        x=valid_days,
                        y=valid_actual,
                        mode='lines+markers',
                        name='실제 라벨',
                        line=dict(color='black', dash='dash', width=2),
                        marker=dict(symbol='circle', size=6),
                        hovertemplate='<b>실제 라벨</b><br>일: %{x}<br>설비 이상 비율: %{y:.2%}<extra></extra>',
                        showlegend=(i == 0),
                        legendgroup='actual'
                    ),
                    row=row, col=1
                )
                
                # 예측값 추가
                fig.add_trace(
                    go.Scatter(
                        x=days,
                        y=daily_predictions,
                        mode='lines+markers',
                        name=f'{model_name} 예측',
                        line=dict(color='blue', width=2),
                        marker=dict(symbol='square', size=6),
                        hovertemplate=f'<b>{model_name} 예측</b><br>일: %{{x}}<br>설비 이상 비율: %{{y:.2%}}<extra></extra>',
                        showlegend=True,
                        legendgroup=f'pred_{model_name.lower()}'
                    ),
                    row=row, col=1
                )
                
                # 설비 이상 발생일 수직선 (각 서브플롯에 개별 추가)
                fig.add_shape(
                    type="line",
                    x0=data['failure_day'], x1=data['failure_day'],
                    y0=-0.1, y1=1.1,
                    line=dict(color="red", width=2, dash="dash"),
                    opacity=0.7,
                    row=row, col=1
                )
                
                # 설비 이상 징후 구간 배경
                fig.add_shape(
                    type="rect",
                    x0=data['failure_warning_start_day'], x1=data['failure_day'],
                    y0=-0.1, y1=1.1,
                    fillcolor="pink",
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                    row=row, col=1
                )
                
                # 알람 레벨 배경색
                for day in days:
                    if day in daily_levels:
                        level_color = daily_levels[day]['color']
                        if level_color == 'yellow':
                            fig.add_shape(
                                type="rect",
                                x0=day - 0.5, x1=day + 0.5,
                                y0=-0.1, y1=1.1,
                                fillcolor="yellow", opacity=0.1,
                                layer="below", line_width=0,
                                row=row, col=1
                            )
                        elif level_color == 'orange':
                            fig.add_shape(
                                type="rect",
                                x0=day - 0.5, x1=day + 0.5,
                                y0=-0.1, y1=1.1,
                                fillcolor="orange", opacity=0.15,
                                layer="below", line_width=0,
                                row=row, col=1
                            )
                        elif level_color == 'red':
                            fig.add_shape(
                                type="rect",
                                x0=day - 0.5, x1=day + 0.5,
                                y0=-0.1, y1=1.1,
                                fillcolor="red", opacity=0.2,
                                layer="below", line_width=0,
                                row=row, col=1
                            )
            
            # 레이아웃 설정
            fig.update_layout(
                height=800,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                ),
                hovermode='x unified',
                margin=dict(r=150)  # 오른쪽 마진 추가 (범례 공간)
            )
            
            # Y축 설정
            fig.update_yaxes(title_text="설비 이상 비율", range=[-0.1, 1.1])
            
            # X축 설정 (마지막 서브플롯에만)
            fig.update_xaxes(title_text="시간 (일)", row=3, col=1)
            fig.update_xaxes(range=[0, 31], dtick=5)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(
                   f"일별 288개 측정 중 설비 이상 신호 비율로 평균을 계산했습니다. "
                   f"배경색: 주의(노랑), 경고(주황), 위험(빨강). "
                   f"설비 이상 징후는 {data['failure_warning_start_day']}일부터 {data['failure_day']}일까지 발생합니다."
            )

            st.markdown("---")
            st.subheader("4. 종합 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **시뮬레이션 설정**
                - 총 기간: 30일
                - 설비 이상 시점: {data['failure_day']}일
                - 측정 간격: 5분
                - 일일 측정: 288회
                
                **데이터 특성**
                - 생성된 시점: {len(data['synthetic_data']):,}개 ({data['failure_day']}일분)
                - 시퀀스 수: {len(data['y_sequences']):,}개
                - 특성 수: 8개 (ch1, ch2 각 4개)
                """)
            
            with col2:
                st.markdown(f"""
                **알람 기준**
                - 주의: 일일 30% 이상 설비 이상 신호
                - 경고: 일일 50% 이상 설비 이상 신호
                - 위험: 일일 80% 이상 설비 이상 신호
                
                **실용성 평가**
                - 30일 장기 모니터링 시뮬레이션
                - 3단계 알람 시스템 (주의/경고/위험)
                - 일별 설비 이상 비율 기반 판단
                - 오경보 최소화 전략 적용
                """)
    
    else:
        st.error("모델 파일을 불러올 수 없습니다. 모델 경로를 확인해주세요.")