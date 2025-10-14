# 2_Failure_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import io
import base64
import os

# 페이지 설정
st.set_page_config(
    page_title="장비 고장 예측 시스템",
    layout="wide"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정 (app 기준으로)
model_path = "../project_failure/models/"

# 메인 타이틀
st.title("장비 고장 예측 시스템")
st.markdown("**시계열 센서 데이터 기반 장비 고장 예측 (Deep Learning)**")
st.markdown("---")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["프로젝트 정보", "성능 분석", "End-to-End 시스템"])

# ========================= TAB 1: 프로젝트 정보 =========================
with tab1:
    st.header("프로젝트 개요")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("프로젝트 목표")
        st.markdown("""
        - **목적**: 산업 장비의 고장 예측 및 예방 정비
        - **데이터**: NASA의 Bearing 진동 센서 데이터
        - **방법**: 딥러닝 시계열 모델 비교
        - **핵심**: 고장 전 조기 경고 시스템 구축
        """)
        
        st.subheader("주요 성과")
        st.markdown("""
        - **최고 성능**: CNN 모델 (78.41% 정확도)
        - **Recall**: 100% (모든 고장 상황 탐지)
        - **조기 경고**: 고장 전 미리 감지 가능
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
        - Streamlit (웹 대시보드)
        """)
    
    st.markdown("---")
    
    # 데이터 설명
    st.subheader("데이터 설명")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("시계열 길이", "10일")
    with col2:
        st.metric("특성 수", "32개")
    with col3:
        st.metric("정상 비율", "84.13%")
    with col4:
        st.metric("고장 비율", "15.87%")
    
    st.markdown("---")
    
    # 전처리 과정
    st.subheader("전처리 과정")
    
    process_steps = {
        "1. 시계열 윈도우": "10일 단위로 분할 (과거 10일로 오늘 예측)",
        "2. 센서 특성": "베어링 진동 센서의 다양한 통계적 특성",
        "3. 정규화": "StandardScaler로 평균 0, 분산 1 정규화",
        "4. 레이블 생성": "고장 6일 이내면 고장(1), 아니면 정상(0)",
        "5. 3D 텐서": "(samples, 10, 32) 형태"
    }
    
    for step, desc in process_steps.items():
        st.markdown(f"**{step}**: {desc}")
    
    st.markdown("---")
    
    # 모델 소개
    st.subheader("모델 소개")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **LSTM (Long Short-Term Memory)**
        - 순환 신경망의 변형
        - 장기 의존성 학습
        - 게이트 메커니즘 활용
        - 정확도: 71.43%
        """)
    
    with col2:
        st.markdown("""
        **GRU (Gated Recurrent Unit)**
        - LSTM의 간소화 버전
        - 빠른 학습 속도
        - 적은 파라미터 수
        - 정확도: 76.19%
        """)
    
    with col3:
        st.markdown("""
        **CNN (Convolutional Neural Network)** (최종 선택)
        - 1D 컨볼루션 레이어
        - 시계열 지역 패턴 탐지
        - 32개 센서 특성 처리
        - 정확도: 78.41%
        """)

# ========================= TAB 2: 성능 분석 =========================
with tab2:
    st.header("성능 분석")
    
    # 성능 결과 로드
    @st.cache_data
    def load_failure_results():
        try:
            results_lstm = joblib.load(model_path + 'results_lstm.pkl')
            results_gru = joblib.load(model_path + 'results_gru.pkl') 
            results_cnn = joblib.load(model_path + 'results_cnn.pkl')
            return results_lstm, results_gru, results_cnn
        except Exception as e:
            st.error(f"성능 결과 파일을 불러올 수 없습니다: {e}")
            return None, None, None
    
    results_lstm, results_gru, results_cnn = load_failure_results()
    
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
                           xticklabels=['정상', '고장'],
                           yticklabels=['정상', '고장'],
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
            
            1. **CNN 모델 최고 성능**
               - 정확도: 78.41% (3개 모델 중 최고)
               - Recall: 100% (모든 고장 상황 탐지)
            
            2. **모든 모델 완벽한 Recall**
               - 세 모델 모두 Recall 100% 달성
               - 고장 상황을 놓치지 않음 (안전성 확보)
            
            3. **CNN의 우수한 특성 추출**
               - 1D Conv 레이어로 지역 패턴 포착
               - 진동 신호의 주파수 특성 효과적 학습
            """)
        
        with col2:
            st.markdown("""
            **성능 비교**
            
            1. **정확도 순위**
               - CNN: 78.41% (1위)
               - GRU: 76.19% (2위)  
               - LSTM: 71.43% (3위)
            
            2. **False Alarm 비교**
               - CNN: 68개 (가장 적음)
               - GRU: 75개
               - LSTM: 90개
            
            **실제 적용 가치**
            - 조기 경고 시스템으로 활용 가능
            - 예방 정비 계획 수립 지원
            - 장비 가동률 향상 기대
            """)

# ========================= TAB 3: End-to-End 시스템 =========================
with tab3:
    st.header("End-to-End 예측 시스템")
    
    # 모델 로드
    @st.cache_resource
    def load_failure_models():
        try:
            model_lstm = keras.models.load_model(model_path + 'model_lstm.keras')
            model_gru = keras.models.load_model(model_path + 'model_gru.keras')
            model_cnn = keras.models.load_model(model_path + 'model_cnn.keras')
            scaler = joblib.load(model_path + 'scaler.pkl')
            return model_lstm, model_gru, model_cnn, scaler
        except Exception as e:
            st.error(f"모델을 불러올 수 없습니다: {e}")
            return None, None, None, None
    
    model_lstm, model_gru, model_cnn, scaler = load_failure_models()
    
    if all([model_lstm, model_gru, model_cnn, scaler]):
        # 설정
        st.subheader("예측 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            n_sequences = st.selectbox("시계열 시퀀스 개수", [5, 10, 20], index=0)
        with col2:
            random_seed = st.number_input("랜덤 시드", value=42, min_value=0, max_value=9999)
        
        if st.button("시계열 데이터 생성 및 예측 실행", use_container_width=True):
            # 랜덤 시계열 데이터 생성
            np.random.seed(random_seed)
            
            # 시계열 파라미터 (실제 모델과 맞춤)
            timesteps = 10  # window_size
            n_features = 32  # 실제 모델이 학습한 특성 수
            
            # 시계열 데이터 생성 (진동 센서 데이터를 모방)
            sequences = []
            labels = []
            
            for i in range(n_sequences):
                # 정상 또는 고장 패턴 결정
                is_failure = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% 고장 확률
                
                # 실제 시계열 센서 데이터 패턴 생성 (표준화된 데이터)
                if is_failure:
                    # 고장 패턴: 점진적으로 변화하는 패턴
                    sequence = np.random.normal(0, 1, (timesteps, n_features))
                    # 시간에 따른 변화 추가 (고장으로 갈수록 변화 증가)
                    trend = np.linspace(0, 2, timesteps)
                    for t in range(timesteps):
                        sequence[t, :] += trend[t] * np.random.normal(0, 0.5, n_features)
                else:
                    # 정상 패턴: 안정적인 노이즈
                    sequence = np.random.normal(0, 1, (timesteps, n_features))
                
                sequences.append(sequence)
                labels.append(is_failure)
            
            X_sequences = np.array(sequences)
            
            # 3개 모델로 예측
            pred_lstm = model_lstm.predict(X_sequences, verbose=0)
            pred_gru = model_gru.predict(X_sequences, verbose=0)
            pred_cnn = model_cnn.predict(X_sequences, verbose=0)
            
            # 확률을 클래스로 변환
            pred_lstm_class = (pred_lstm > 0.5).astype(int).flatten()
            pred_gru_class = (pred_gru > 0.5).astype(int).flatten()
            pred_cnn_class = (pred_cnn > 0.5).astype(int).flatten()
            
            # 결과 데이터프레임 생성
            results_df = pd.DataFrame({
                'Sequence_ID': [f'SEQ_{i+1:03d}' for i in range(n_sequences)],
                'Actual_Label': ['고장' if l == 1 else '정상' for l in labels],
                'LSTM_Prediction': ['고장' if p == 1 else '정상' for p in pred_lstm_class],
                'LSTM_Probability': pred_lstm.flatten(),
                'GRU_Prediction': ['고장' if p == 1 else '정상' for p in pred_gru_class],
                'GRU_Probability': pred_gru.flatten(),
                'CNN_Prediction': ['고장' if p == 1 else '정상' for p in pred_cnn_class],
                'CNN_Probability': pred_cnn.flatten()
            })
            
            # 결과 표시
            st.subheader("예측 결과")
            
            # 요약 통계
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_failure_lstm = sum(pred_lstm_class)
                st.metric("LSTM 고장 예측", f"{n_failure_lstm}개", 
                         f"{n_failure_lstm/n_sequences*100:.1f}%")
            
            with col2:
                n_failure_gru = sum(pred_gru_class)
                st.metric("GRU 고장 예측", f"{n_failure_gru}개",
                         f"{n_failure_gru/n_sequences*100:.1f}%")
            
            with col3:
                n_failure_cnn = sum(pred_cnn_class)
                st.metric("CNN 고장 예측", f"{n_failure_cnn}개",
                         f"{n_failure_cnn/n_sequences*100:.1f}%")
            
            # 상세 결과 테이블
            st.subheader("상세 예측 결과")
            
            # 고장으로 예측된 항목만 필터링 옵션
            show_all = st.checkbox("모든 시퀀스 표시", value=False)
            
            if not show_all:
                # 하나라도 고장으로 예측된 시퀀스만 표시
                mask = (pred_lstm_class == 1) | (pred_gru_class == 1) | (pred_cnn_class == 1)
                display_df = results_df[mask].copy()
                st.write(f"**고장 예측 시퀀스: {len(display_df)}개**")
            else:
                display_df = results_df.copy()
                st.write(f"**전체 시퀀스: {len(display_df)}개**")
            
            if len(display_df) > 0:
                # 확률 기준으로 정렬
                display_df = display_df.sort_values('CNN_Probability', ascending=False)
                
                # 스타일 적용
                def color_predictions(val):
                    if val == '고장':
                        return 'background-color: #ffcccc'  # 연한 빨강
                    else:
                        return 'background-color: #ccffcc'  # 연한 초록
                
                styled_df = display_df.style.map(
                    color_predictions, 
                    subset=['Actual_Label', 'LSTM_Prediction', 'GRU_Prediction', 'CNN_Prediction']
                ).format({
                    'LSTM_Probability': '{:.3f}',
                    'GRU_Probability': '{:.3f}',
                    'CNN_Probability': '{:.3f}'
                })
                
                st.dataframe(styled_df, use_container_width=True)
                
                # 시계열 시각화
                st.subheader("시계열 데이터 시각화")
                
                # 처음 3개 시퀀스의 일부 특성만 시각화 (32개는 너무 많음)
                n_show = min(3, len(display_df))
                n_features_show = 8  # 처음 8개 특성만 표시
                
                fig, axes = plt.subplots(n_show, n_features_show, figsize=(20, 3*n_show))
                if n_show == 1:
                    axes = axes.reshape(1, -1)
                
                for i in range(n_show):
                    idx = display_df.index[i]
                    seq_data = X_sequences[idx]  # (timesteps, features)
                    actual = labels[idx]
                    
                    for feature in range(n_features_show):
                        ax = axes[i, feature] if n_show > 1 else axes[feature]
                        
                        color = 'red' if actual == 1 else 'blue'
                        ax.plot(seq_data[:, feature], color=color, linewidth=1.5)
                        
                        if i == 0:
                            ax.set_title(f"Feature {feature+1}", fontsize=10)
                        if feature == 0:
                            ax.set_ylabel(f"SEQ_{idx+1:03d}\n({'고장' if actual == 1 else '정상'})", fontsize=10)
                        if i == n_show - 1:
                            ax.set_xlabel('Time', fontsize=9)
                        
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(labelsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info(f"💡 32개 특성 중 처음 8개만 표시했습니다. 실제로는 {n_features}개 특성이 모두 사용됩니다.")
                
                # CSV 다운로드
                st.subheader("결과 다운로드")
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="CSV 파일 다운로드",
                    data=csv,
                    file_name=f'failure_prediction_results_{n_sequences}sequences.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                
                # 모델 비교 차트
                st.subheader("모델 예측 비교")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # 고장 예측 개수 비교
                models = ['LSTM', 'GRU', 'CNN']
                failure_counts = [n_failure_lstm, n_failure_gru, n_failure_cnn]
                
                bars = ax1.bar(models, failure_counts, color=['skyblue', 'lightgreen', 'orange'])
                ax1.set_title('모델별 고장 예측 개수', fontweight='bold')
                ax1.set_ylabel('고장 예측 개수')
                
                # 막대 위에 값 표시
                for bar, count in zip(bars, failure_counts):
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                            f'{count}개', ha='center', va='bottom', fontweight='bold')
                
                # 확률 분포 히스토그램
                ax2.hist(pred_lstm.flatten(), alpha=0.5, label='LSTM', bins=20)
                ax2.hist(pred_gru.flatten(), alpha=0.5, label='GRU', bins=20)  
                ax2.hist(pred_cnn.flatten(), alpha=0.5, label='CNN', bins=20)
                ax2.set_title('고장 확률 분포', fontweight='bold')
                ax2.set_xlabel('고장 확률')
                ax2.set_ylabel('빈도')
                ax2.legend()
                ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='임계값')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            else:
                st.info("고장으로 예측된 시퀀스가 없습니다.")
    
    else:
        st.error("모델 파일들을 확인해주세요. project_failure/models/ 폴더에 다음 파일들이 필요합니다:")
        st.code("""
        - model_lstm.keras
        - model_gru.keras  
        - model_cnn.keras
        - scaler.pkl
        """)