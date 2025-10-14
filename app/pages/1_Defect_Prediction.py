# 1_Defect_Prediction.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import io
import base64
import os

# 페이지 설정
st.set_page_config(
    page_title="제품 불량 예측 시스템",
    layout="wide"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정 (app 기준으로)
model_path = "../project_defect/models/"

# 메인 타이틀
st.title("제품 불량 예측 시스템")
st.markdown("**반도체 제조 공정 센서 데이터 기반 불량품 조기 탐지**")
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
        - **목적**: 반도체 제조 공정에서 불량품 조기 탐지
        - **데이터**: SECOM 센서 데이터 (UCI Repository)
        - **방법**: 전통적 머신러닝 모델 비교
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
            results_logistic = joblib.load(model_path + 'results_logistic.pkl')
            results_rf = joblib.load(model_path + 'results_rf.pkl') 
            results_xgb = joblib.load(model_path + 'results_xgb.pkl')
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
            ax.set_xticklabels(['Logistic', 'Random Forest', 'XGBoost'])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            st.pyplot(fig)
        
        with col2:
            # Confusion Matrix 비교
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            results_list = [results_logistic, results_rf, results_xgb]
            model_names = ['Logistic', 'RF', 'XGBoost']
            
            for i, (result, name) in enumerate(zip(results_list, model_names)):
                cm = result['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['정상', '불량'],
                           yticklabels=['정상', '불량'],
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
    
    # 모델 로드
    @st.cache_resource
    def load_models():
        try:
            model_logistic = joblib.load(model_path + 'model_logistic.pkl')
            model_rf = joblib.load(model_path + 'model_rf.pkl') 
            model_xgb = joblib.load(model_path + 'model_xgb.pkl')
            scaler = joblib.load(model_path + 'scaler.pkl')
            return model_logistic, model_rf, model_xgb, scaler
        except Exception as e:
            st.error(f"모델을 불러올 수 없습니다: {e}")
            return None, None, None, None
    
    # 실제 데이터 로드
    @st.cache_data
    def load_secom_data():
        try:
            # 전처리된 데이터 로드 (CSV 형식)
            X_test = pd.read_csv('../project_defect/processed_data/X_test.csv')
            y_test = pd.read_csv('../project_defect/processed_data/y_test.csv')
            
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
        # 설정
        st.subheader("예측 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.selectbox("샘플 개수 선택", [10, 20, 50], index=0)
        with col2:
            random_seed = st.number_input("랜덤 시드", value=42, min_value=0, max_value=9999)
        
        if st.button("실제 데이터 샘플링 및 예측 실행", use_container_width=True):
            # 랜덤 샘플 선택
            np.random.seed(random_seed)
            sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
            
            X_samples = X_test[sample_indices]
            y_actual = y_test[sample_indices]
            
            st.success(f"Test 데이터에서 {n_samples}개 샘플 추출 완료!")
            
            # 입력 데이터 확인
            with st.expander("입력 데이터 확인 (처음 5개 샘플, 처음 10개 특성)"):
                # 데이터프레임으로 변환 (보기 편하게)
                preview_df = pd.DataFrame(
                    X_samples[:5, :10],  # 처음 5개 샘플, 처음 10개 특성
                    columns=[f'Feature_{i+1}' for i in range(10)],
                    index=[f'Sample_{i+1}' for i in range(5)]
                )
                st.dataframe(preview_df.style.format("{:.4f}"), use_container_width=True)
                st.info(f"실제로는 {X_samples.shape[1]}개 특성이 모두 사용됩니다.")
            
            # 3개 모델로 예측
            pred_logistic = model_logistic.predict(X_samples)
            pred_rf = model_rf.predict(X_samples)
            pred_xgb = model_xgb.predict(X_samples)
            
            # 예측 확률
            prob_logistic = model_logistic.predict_proba(X_samples)[:, 1]
            prob_rf = model_rf.predict_proba(X_samples)[:, 1] 
            prob_xgb = model_xgb.predict_proba(X_samples)[:, 1]
            
            # 결과 데이터프레임 생성
            results_df = pd.DataFrame({
                'Sample_ID': [f'SAMPLE_{sample_indices[i]+1:04d}' for i in range(n_samples)],
                'Actual_Label': ['불량' if y == 1 else '정상' for y in y_actual],
                'Logistic_Pred': ['불량' if p == 1 else '정상' for p in pred_logistic],
                'Logistic_Prob': prob_logistic,
                'RF_Pred': ['불량' if p == 1 else '정상' for p in pred_rf],
                'RF_Prob': prob_rf,
                'XGB_Pred': ['불량' if p == 1 else '정상' for p in pred_xgb],
                'XGB_Prob': prob_xgb
            })
            
            # 정답 맞춤 여부 추가
            results_df['Logistic_Correct'] = (pred_logistic == y_actual)
            results_df['RF_Correct'] = (pred_rf == y_actual)
            results_df['XGB_Correct'] = (pred_xgb == y_actual)
            
            # 결과 표시
            st.subheader("예측 결과")
            
            # 요약 통계
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                actual_defects = sum(y_actual)
                st.metric("실제 불량", f"{actual_defects}개", 
                         f"{actual_defects/n_samples*100:.1f}%")
            
            with col2:
                n_defect_logistic = sum(pred_logistic)
                accuracy_logistic = sum(pred_logistic == y_actual) / n_samples
                st.metric("Logistic 예측", f"{n_defect_logistic}개", 
                         f"정확도: {accuracy_logistic*100:.1f}%")
            
            with col3:
                n_defect_rf = sum(pred_rf)
                accuracy_rf = sum(pred_rf == y_actual) / n_samples
                st.metric("RF 예측", f"{n_defect_rf}개",
                         f"정확도: {accuracy_rf*100:.1f}%")
            
            with col4:
                n_defect_xgb = sum(pred_xgb)
                accuracy_xgb = sum(pred_xgb == y_actual) / n_samples
                st.metric("XGBoost 예측", f"{n_defect_xgb}개",
                         f"정확도: {accuracy_xgb*100:.1f}%")
            
            # 상세 결과 테이블
            st.subheader("상세 예측 결과")
            
            # 필터링 옵션
            filter_option = st.radio(
                "표시 옵션",
                ["전체", "불량만", "오답만"],
                horizontal=True
            )
            
            if filter_option == "불량만":
                mask = y_actual == 1
                display_df = results_df[mask].copy()
                st.write(f"**실제 불량 샘플: {len(display_df)}개**")
            elif filter_option == "오답만":
                mask = ~(results_df['Logistic_Correct'] | results_df['RF_Correct'] | results_df['XGB_Correct'])
                display_df = results_df[mask].copy()
                st.write(f"**3개 모델 모두 틀린 샘플: {len(display_df)}개**")
            else:
                display_df = results_df.copy()
                st.write(f"**전체 샘플: {len(display_df)}개**")
            
            if len(display_df) > 0:
                # 확률 기준으로 정렬
                display_df = display_df.sort_values('RF_Prob', ascending=False)
                
                # 스타일 적용
                def color_predictions(row):
                    colors = []
                    for col in row.index:
                        if col == 'Actual_Label':
                            if row[col] == '불량':
                                colors.append('background-color: #ffe6e6; font-weight: bold')
                            else:
                                colors.append('background-color: #e6ffe6')
                        elif col in ['Logistic_Pred', 'RF_Pred', 'XGB_Pred']:
                            # 정답과 비교
                            is_correct = row[col] == row['Actual_Label']
                            if is_correct:
                                colors.append('background-color: #ccffcc')  # 연한 초록
                            else:
                                colors.append('background-color: #ffcccc')  # 연한 빨강
                        else:
                            colors.append('')
                    return colors
                
                styled_df = display_df.drop(columns=['Logistic_Correct', 'RF_Correct', 'XGB_Correct']).style.apply(
                    color_predictions, axis=1
                ).format({
                    'Logistic_Prob': '{:.3f}',
                    'RF_Prob': '{:.3f}',
                    'XGB_Prob': '{:.3f}'
                })
                
                st.dataframe(styled_df, use_container_width=True)
                
                st.markdown("""
                **색상 설명:**
                - 초록색 배경: 정답
                - 빨간색 배경: 오답
                - Actual Label 진한 색: 실제 불량/정상
                """)
                
                # CSV 다운로드
                st.subheader("결과 다운로드")
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="CSV 파일 다운로드",
                    data=csv,
                    file_name=f'defect_prediction_results_{n_samples}samples.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                
                # 모델 비교 차트
                st.subheader("모델 성능 비교")
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # 1. 정확도 비교
                ax = axes[0, 0]
                models = ['Logistic', 'Random Forest', 'XGBoost']
                accuracies = [accuracy_logistic, accuracy_rf, accuracy_xgb]
                
                bars = ax.bar(models, accuracies, color=['skyblue', 'lightgreen', 'orange'])
                ax.set_title('모델별 정확도', fontweight='bold')
                ax.set_ylabel('Accuracy')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, acc in zip(bars, accuracies):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                           f'{acc*100:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                # 2. 불량 예측 개수
                ax = axes[0, 1]
                defect_counts = [actual_defects, n_defect_logistic, n_defect_rf, n_defect_xgb]
                labels = ['실제', 'Logistic', 'RF', 'XGBoost']
                colors_bar = ['red', 'skyblue', 'lightgreen', 'orange']
                
                bars = ax.bar(labels, defect_counts, color=colors_bar)
                ax.set_title('불량 예측 개수 비교', fontweight='bold')
                ax.set_ylabel('개수')
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, count in zip(bars, defect_counts):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                           f'{count}개', ha='center', va='bottom', fontweight='bold')
                
                # 3. 확률 분포
                ax = axes[1, 0]
                ax.hist(prob_logistic, alpha=0.5, label='Logistic', bins=20)
                ax.hist(prob_rf, alpha=0.5, label='Random Forest', bins=20)  
                ax.hist(prob_xgb, alpha=0.5, label='XGBoost', bins=20)
                ax.set_title('불량 확률 분포', fontweight='bold')
                ax.set_xlabel('불량 확률')
                ax.set_ylabel('빈도')
                ax.legend()
                ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='임계값')
                ax.grid(True, alpha=0.3)
                
                # 4. Confusion Matrix (RF만)
                ax = axes[1, 1]
                cm = confusion_matrix(y_actual, pred_rf)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['정상', '불량'],
                           yticklabels=['정상', '불량'],
                           ax=ax, cbar=True, annot_kws={'size': 14})
                ax.set_title('Random Forest Confusion Matrix', fontweight='bold')
                ax.set_xlabel('예측')
                ax.set_ylabel('실제')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            else:
                st.info("표시할 샘플이 없습니다.")
    
    else:
        st.error("필요한 파일들을 확인해주세요:")
        st.code("""
            필요 파일:
            - project_defect/models/model_*.pkl
            - project_defect/processed_data/X_test.pkl
            - project_defect/processed_data/y_test.pkl
                    """)