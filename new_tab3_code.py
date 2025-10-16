    # 새로운 시나리오: 진동 패턴 분석 시뮬레이터
    
    # 모델 로드
    @st.cache_resource
    def load_models_and_data():
        try:
            # 모델 로드
            model_lstm = keras.models.load_model(os.path.join(project_root, "project_failure", "models", "model_lstm.keras"))
            model_gru = keras.models.load_model(os.path.join(project_root, "project_failure", "models", "model_gru.keras"))
            model_cnn = keras.models.load_model(os.path.join(project_root, "project_failure", "models", "model_cnn.keras"))
            scaler = joblib.load(os.path.join(project_root, "project_failure", "models", "scaler.pkl"))
            
            # 테스트 데이터 로드
            X_test = joblib.load(os.path.join(project_root, "project_failure", "processed_data", "X_test_scaled.pkl"))
            y_test = joblib.load(os.path.join(project_root, "project_failure", "processed_data", "y_test.pkl"))
            
            return model_lstm, model_gru, model_cnn, scaler, X_test, y_test
        except Exception as e:
            st.error(f"모델을 불러올 수 없습니다: {e}")
            return None, None, None, None, None, None
    
    # 모델과 데이터 로드
    models_and_data = load_models_and_data()
    
    if all(x is not None for x in models_and_data):
        model_lstm, model_gru, model_cnn, scaler, X_test, y_test = models_and_data
        
        st.success("✅ AI 모델 및 테스트 데이터 로드 완료!")
        st.info(f"📊 사용 가능한 테스트 시퀀스: {len(X_test)}개")
        
        # Step 1: 베어링 데이터 생성
        st.subheader("Step 1: 베어링 진동 데이터 생성")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎲 새로운 베어링 데이터 생성", type="primary", use_container_width=True):
                # 랜덤으로 테스트 시퀀스 선택
                selected_idx = np.random.randint(0, len(X_test))
                selected_sequence = X_test[selected_idx]  # (10, 32) 
                selected_label = y_test[selected_idx] if selected_idx < len(y_test) else 0
                
                st.session_state.selected_sequence = selected_sequence
                st.session_state.selected_label = selected_label
                st.session_state.selected_idx = selected_idx
                st.session_state.show_prediction = False
                st.session_state.generated_data = True
                
                st.rerun()
        
        with col2:
            if st.button("🤖 AI 예측 분석", use_container_width=True, 
                        disabled='generated_data' not in st.session_state):
                st.session_state.show_prediction = True
                st.rerun()
        
        # Step 2: 생성된 데이터 시각화
        if 'generated_data' in st.session_state:
            st.subheader("Step 2: 실제 베어링 진동 패턴")
            st.markdown("**10분간 측정된 베어링 진동 데이터** (실제 기업에서 받는 것과 동일한 형태)")
            
            selected_sequence = st.session_state.selected_sequence
            
            # 시퀀스의 평균 통계값 계산 (시각화용)
            avg_stats = np.mean(selected_sequence, axis=0)  # (32,) - 각 특성의 평균
            
            # 4개 베어링의 RMS, Peak, Std, Kurtosis 추출
            bearings_stats = {
                'Bearing 1': {
                    'rms': avg_stats[0], 'peak': avg_stats[1], 'std': avg_stats[2], 'kurtosis': avg_stats[3]
                },
                'Bearing 2': {
                    'rms': avg_stats[8], 'peak': avg_stats[9], 'std': avg_stats[10], 'kurtosis': avg_stats[11]
                },
                'Bearing 3': {
                    'rms': avg_stats[16], 'peak': avg_stats[17], 'std': avg_stats[18], 'kurtosis': avg_stats[19]
                },
                'Bearing 4': {
                    'rms': avg_stats[24], 'peak': avg_stats[25], 'std': avg_stats[26], 'kurtosis': avg_stats[27]
                }
            }
            
            # 합성 진동 패턴 생성 및 시각화
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, (bearing_name, stats) in enumerate(bearings_stats.items()):
                ax = axes[i]
                
                # 합성 진동 신호 생성
                vibration_signal = generate_synthetic_vibration(
                    stats['rms'], stats['peak'], stats['std'], stats['kurtosis'], 
                    samples=1500
                )
                
                # 시간축 (1.5초, 1000Hz 샘플링)
                time_axis = np.linspace(0, 1.5, len(vibration_signal))
                
                # 진동 패턴 플롯
                ax.plot(time_axis, vibration_signal, color=colors[i], linewidth=1, alpha=0.8)
                
                ax.set_title(f'{bearing_name} 진동 패턴', fontweight='bold', fontsize=14)
                ax.set_xlabel('시간 (초)')
                ax.set_ylabel('진동 가속도 (g)')
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#f8f9fa')
                
                # 통계 정보 텍스트로 표시
                info_text = f'RMS: {stats["rms"]:.3f}g\nPeak: {stats["peak"]:.3f}g'
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('🔧 실제 베어링 진동 데이터 (10분 평균)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            
            # 데이터 정보
            with st.expander("📋 베어링 데이터 상세 정보"):
                st.markdown(f"""
                **측정 조건:**
                - 샘플링 주파수: 20kHz
                - 측정 시간: 10분간
                - 베어링 개수: 4개
                - 센서 타입: 가속도계 (진동 측정)
                
                **선택된 테스트 시퀀스:** #{st.session_state.selected_idx + 1}
                **실제 라벨:** {"고장" if st.session_state.selected_label == 1 else "정상"} (AI 예측과 비교용)
                """)
        
        # Step 3: AI 예측 결과
        if st.session_state.get('show_prediction', False):
            st.subheader("Step 3: AI 예측 결과")
            st.markdown("**3개의 딥러닝 모델이 진동 패턴을 분석한 결과:**")
            
            selected_sequence = st.session_state.selected_sequence
            
            # 실제 모델 예측
            X_input = selected_sequence.reshape(1, selected_sequence.shape[0], selected_sequence.shape[1])
            
            pred_lstm = float(model_lstm.predict(X_input, verbose=0)[0][0])
            pred_gru = float(model_gru.predict(X_input, verbose=0)[0][0])
            pred_cnn = float(model_cnn.predict(X_input, verbose=0)[0][0])
            
            # 예측 결과 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="🧠 LSTM 모델", 
                    value=f"{pred_lstm:.1%}",
                    delta="위험" if pred_lstm > 0.5 else "정상"
                )
            
            with col2:
                st.metric(
                    label="⚡ GRU 모델", 
                    value=f"{pred_gru:.1%}",
                    delta="위험" if pred_gru > 0.5 else "정상"
                )
            
            with col3:
                st.metric(
                    label="🔍 CNN 모델", 
                    value=f"{pred_cnn:.1%}",
                    delta="위험" if pred_cnn > 0.5 else "정상"
                )
            
            # 예측 결과 차트
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = ['LSTM', 'GRU', 'CNN']
            predictions = [pred_lstm, pred_gru, pred_cnn]
            colors_pred = ['#1f77b4', '#2ca02c', '#ff7f0e']
            
            bars = ax.bar(models, predictions, color=colors_pred, alpha=0.7, width=0.6)
            
            # 막대 위에 퍼센트 표시
            for bar, pred in zip(bars, predictions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{pred:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            # 기준선들
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='경고 기준 (50%)')
            ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.8, linewidth=2, label='위험 기준 (70%)')
            
            # 위험 구간 색칠
            ax.fill_between([-0.5, 2.5], 0.5, 0.7, alpha=0.1, color='orange', label='경고구간')
            ax.fill_between([-0.5, 2.5], 0.7, 1.0, alpha=0.1, color='red', label='위험구간')
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('고장 확률', fontsize=12, fontweight='bold')
            ax.set_title('🤖 AI 모델별 고장 예측 결과', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            
            # Y축 퍼센트 표시
            ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
            ax.set_yticklabels(['0%', '20%', '40%', '50%', '60%', '70%', '80%', '100%'])
            
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            
            # 결과 해석
            max_prob = max(predictions)
            dominant_model = models[predictions.index(max_prob)]
            actual_label = st.session_state.selected_label
            
            st.subheader("📊 예측 결과 분석")
            
            if max_prob > 0.7:
                st.error(f"🚨 **고장 위험 감지!** {dominant_model} 모델이 {max_prob:.1%} 확률로 고장을 예측했습니다.")
                recommendation = "즉시 베어링 점검 및 교체를 권고합니다."
            elif max_prob > 0.5:
                st.warning(f"⚠️ **경고 수준!** {dominant_model} 모델이 {max_prob:.1%} 확률로 고장 가능성을 감지했습니다.")
                recommendation = "예방 정비 계획을 수립하고 지속적인 모니터링이 필요합니다."
            else:
                st.success(f"✅ **정상 상태!** 모든 모델이 {max_prob:.1%} 이하의 낮은 고장 확률을 예측했습니다.")
                recommendation = "현재 상태를 유지하고 정기적인 점검을 계속하세요."
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                **🎯 AI 예측 결과:**
                - 최고 확률: **{max_prob:.1%}** ({dominant_model} 모델)
                - 권장사항: {recommendation}
                - 실제 정답: **{"고장" if actual_label == 1 else "정상"}**
                """)
            
            with col2:
                # 정확도 표시
                is_correct = (max_prob > 0.5) == (actual_label == 1)
                if is_correct:
                    st.success("🎯 **예측 정확!**")
                else:
                    st.error("❌ **예측 틀림**")
            
            # 다시 테스트 버튼
            if st.button("🔄 다른 데이터로 다시 테스트", use_container_width=True):
                for key in ['selected_sequence', 'selected_label', 'selected_idx', 'show_prediction', 'generated_data']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    else:
        st.error("❌ 모델 또는 데이터를 불러올 수 없습니다.")
        st.markdown("""
        **필요한 파일들:**
        - `project_failure/models/model_lstm.keras`
        - `project_failure/models/model_gru.keras` 
        - `project_failure/models/model_cnn.keras`
        - `project_failure/models/scaler.pkl`
        - `project_failure/processed_data/X_test_scaled.pkl`
        - `project_failure/processed_data/y_test.pkl`
        """)