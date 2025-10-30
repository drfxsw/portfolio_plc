# 3_PCB_Inspection.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import sys
from pathlib import Path

# 스타일 유틸리티 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.styles import load_common_styles, create_page_header, COLORS, CHART_COLORS

# 페이지 설정
st.set_page_config(
    page_title="PCB 외관검사 자동화",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# 공통 스타일 로드 및 페이지 헤더
load_common_styles()
create_page_header("PCB 외관검사 자동화",
                    "YOLOv8 기반 실시간 PCB 결함 검출 시스템 (AOI)")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
results_path = "../project_vision/researching/results/"

# 성능 결과 로드
@st.cache_data
def load_vision_results():
    try:
        # CSV에서 성능 지표 로드
        results_s = pd.read_csv(os.path.join(results_path, "yolov8s_pcb/results.csv"))
        results_m = pd.read_csv(os.path.join(results_path, "yolov8m_pcb/results.csv"))
        
        # 최고 성능 추출
        best_s = results_s.loc[results_s['metrics/mAP50(B)'].idxmax()]
        best_m = results_m.loc[results_m['metrics/mAP50(B)'].idxmax()]
        
        return {
            's': {
                'mAP50': best_s['metrics/mAP50(B)'],
                'mAP50_95': best_s['metrics/mAP50-95(B)'],
                'precision': best_s['metrics/precision(B)'],
                'recall': best_s['metrics/recall(B)'],
                'epoch': int(best_s['epoch'])
            },
            'm': {
                'mAP50': best_m['metrics/mAP50(B)'],
                'mAP50_95': best_m['metrics/mAP50-95(B)'],
                'precision': best_m['metrics/precision(B)'],
                'recall': best_m['metrics/recall(B)'],
                'epoch': int(best_m['epoch'])
            },
            'csv_s': results_s,
            'csv_m': results_m
        }
    except Exception as e:
        st.error(f"성능 결과 파일을 불러올 수 없습니다: {e}")
        return None

# 결과 로드
results = load_vision_results()

# 탭 생성
tab1, tab2, tab3 = st.tabs(["프로젝트 정보", "성능 분석", "실시간 검출 데모"])

# ========================= TAB 1: 프로젝트 정보 =========================
with tab1:
    st.header("프로젝트 개요")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("프로젝트 목표")
        st.markdown("""
        - **목적**: 딥러닝 기반 PCB 결함 실시간 검출 (AOI 시스템)
        - **데이터**: Kaggle PCB Defects 693장
        - **방법**: YOLOv8 Small/Medium 객체 검출 모델 비교
        - **핵심**: 생산라인 적용 가능한 실시간 검사 시스템
        """)
        
        st.subheader("주요 성과")
        if results:
            st.markdown(f"""
        - **최고 성능**: YOLOv8m ({results['m']['mAP50']*100:.1f}% mAP50)
        - **정밀도**: {results['m']['precision']*100:.1f}% (오검출 최소화)
        - **재현율**: {results['m']['recall']*100:.1f}% (결함 미검출 방지)
        - **추론 속도**: 4.9ms/이미지 (실시간 검사 가능)
        """)
        else:
            st.markdown("""
        - **최고 성능**: YOLOv8m (90.3% mAP50)
        - **정밀도**: 93.9% (오검출 최소화)
        - **재현율**: 82.5% (결함 미검출 방지)
        - **추론 속도**: 4.9ms/이미지 (실시간 검사 가능)
        """)
    
    with col2:
        st.subheader("기술 스택")
        st.markdown("""
        **객체 검출**
        - Ultralytics YOLOv8: 최신 SOTA 모델
        - PyTorch: 딥러닝 백엔드
        - Transfer Learning: COCO 사전학습
        
        **데이터 처리**
        - OpenCV: 이미지 전처리
        - XML→YOLO: 라벨 변환
        - Albumentations: 데이터 증강
        
        **학습 환경**
        - Google Colab: Tesla T4 GPU
        - Mixed Precision (AMP): 학습 최적화
        """)
    
    st.markdown("---")
    
    # 데이터 설명
    st.subheader("데이터 설명")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 이미지", "693장")
    with col2:
        st.metric("Train/Test", "552 / 141")
    with col3:
        st.metric("결함 유형", "6종")
    with col4:
        st.metric("해상도", "416×416")
    
    st.markdown("---")
    
    # 결함 유형
    st.subheader("6가지 결함 유형")
    
    col1, col2, col3 = st.columns(3)
    
    defect_info = {
        "Missing_hole": ("홀 누락", "치명적", "드릴 구멍 누락"),
        "Short": ("회로 단락", "치명적", "불필요한 전기 연결"),
        "Open_circuit": ("회로 단선", "치명적", "전기 연결 끊김"),
        "Mouse_bite": ("모서리 결함", "주요", "가장자리 크레센트 형태"),
        "Spurious_copper": ("불필요 동박", "주요", "불필요한 구리 잔류"),
        "Spur": ("돌기", "경미", "작은 금속 돌기")
    }
    
    for i, (defect, (name, severity, desc)) in enumerate(defect_info.items()):
        col = [col1, col2, col3][i % 3]
        with col:
            st.markdown(f"""
            **{defect}** ({name})
            - 심각도: {severity}
            - 설명: {desc}
            """)
    
    st.markdown("---")
    
    # 전처리 과정
    st.subheader("전처리 과정")
    
    process_steps = {
        "1. 이미지 리사이즈": "3034×1586 → 416×416 (YOLOv8 입력 크기)",
        "2. 라벨 변환": "XML (좌표) → YOLO 형식 (클래스, 중심점, 너비, 높이)",
        "3. 정규화": "픽셀 값 0~1 스케일링",
        "4. 데이터 분할": "Train 80% (552장) / Test 20% (141장)",
        "5. 증강": "Blur, MedianBlur, ToGray, CLAHE 적용"
    }
    
    for step, desc in process_steps.items():
        st.markdown(f"**{step}**: {desc}")
    
    st.markdown("---")
    
    # 모델 소개
    st.subheader("모델 소개")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if results:
            st.markdown(f"""
        **YOLOv8 Small**
        - 파라미터: 11.1M
        - 모델 크기: 22.5MB
        - 학습 시간: 6분
        - mAP50: {results['s']['mAP50']*100:.1f}%
        - 특징: 빠른 추론, 경량
        """)
        else:
            st.markdown("""
        **YOLOv8 Small**
        - 파라미터: 11.1M
        - 모델 크기: 22.5MB
        - 학습 시간: 6분
        - mAP50: 85.9%
        - 특징: 빠른 추론, 경량
        """)
    
    with col2:
        if results:
            st.markdown(f"""
        **YOLOv8 Medium** (최종 선택)
        - 파라미터: 25.9M
        - 모델 크기: 52.0MB
        - 학습 시간: 10분
        - mAP50: {results['m']['mAP50']*100:.1f}%
        - 특징: 높은 정확도, 안정적
        """)
        else:
            st.markdown("""
        **YOLOv8 Medium** (최종 선택)
        - 파라미터: 25.9M
        - 모델 크기: 52.0MB
        - 학습 시간: 10분
        - mAP50: 90.3%
        - 특징: 높은 정확도, 안정적
        """)

# ========================= TAB 2: 성능 분석 =========================
with tab2:
    st.header("성능 분석")
    
    if results:
        # 성능 지표 비교
        st.subheader("성능 지표 비교")
        
        performance_data = {
            'Model': ['YOLOv8s', 'YOLOv8m'],
            'mAP50': [results['s']['mAP50'], results['m']['mAP50']],
            'mAP50-95': [results['s']['mAP50_95'], results['m']['mAP50_95']],
            'Precision': [results['s']['precision'], results['m']['precision']],
            'Recall': [results['s']['recall'], results['m']['recall']],
            'Best Epoch': [results['s']['epoch'], results['m']['epoch']],
            '학습 시간': ['6분', '10분'],
            '모델 크기': ['22.5MB', '52.0MB']
        }
        
        df_performance = pd.DataFrame(performance_data)
        
        # 최고 성능 하이라이트
        def highlight_max(s):
            if s.name in ['Model', 'Best Epoch', '학습 시간', '모델 크기']:
                return [''] * len(s)
            is_max = s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_max]
        
        st.dataframe(
            df_performance.style.apply(highlight_max, axis=0).format({
                'mAP50': '{:.4f}',
                'mAP50-95': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # 클래스별 성능
        st.subheader("클래스별 성능 (YOLOv8m)")
        
        class_data = {
            'Class': ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper'],
            'Precision': [0.962, 0.954, 0.962, 0.934, 0.909, 0.901],
            'Recall': [0.990, 0.778, 0.731, 0.962, 0.714, 0.783],
            'mAP50': [0.966, 0.888, 0.933, 0.956, 0.792, 0.883]
        }
        
        df_class = pd.DataFrame(class_data)
        
        st.dataframe(
            df_class.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'mAP50': '{:.3f}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # 시각화
        st.subheader("성능 시각화")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 모델 비교 막대 그래프
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(2)
            width = 0.2
            
            metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
            colors = CHART_COLORS[:4]
            
            for i, metric in enumerate(metrics):
                values = df_performance[metric].values
                ax.bar(x + (i - 1.5) * width, values, width, 
                       label=metric, alpha=0.8, color=colors[i])
            
            ax.set_ylabel('Score')
            ax.set_title('모델별 성능 지표 비교', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['YOLOv8s', 'YOLOv8m'])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            st.pyplot(fig)
        
        with col2:
            # 클래스별 mAP50
            fig, ax = plt.subplots(figsize=(10, 6))
            
            classes = df_class['Class']
            mAP50_values = df_class['mAP50']
            
            bars = ax.barh(classes, mAP50_values, color=CHART_COLORS[0], alpha=0.7)
            
            # 값 표시
            for i, (bar, val) in enumerate(zip(bars, mAP50_values)):
                ax.text(val + 0.01, i, f'{val:.3f}', 
                       va='center', fontweight='bold')
            
            ax.set_xlabel('mAP50')
            ax.set_title('클래스별 mAP50 (YOLOv8m)', fontweight='bold')
            ax.set_xlim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='x')
            
            st.pyplot(fig)
        
        st.markdown("---")
        
        # 학습 곡선
        st.subheader("학습 과정")
        
        # Plotly로 학습 곡선 그리기
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['mAP50', 'mAP50-95', 'Precision', 'Recall'],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        metrics_plot = [
            ('metrics/mAP50(B)', 1, 1),
            ('metrics/mAP50-95(B)', 1, 2),
            ('metrics/precision(B)', 2, 1),
            ('metrics/recall(B)', 2, 2)
        ]
        
        for metric, row, col in metrics_plot:
            # YOLOv8s
            fig.add_trace(
                go.Scatter(
                    x=results['csv_s']['epoch'],
                    y=results['csv_s'][metric],
                    name='YOLOv8s',
                    line=dict(color=CHART_COLORS[0], width=2),
                    showlegend=(row==1 and col==1)
                ),
                row=row, col=col
            )
            
            # YOLOv8m
            fig.add_trace(
                go.Scatter(
                    x=results['csv_m']['epoch'],
                    y=results['csv_m'][metric],
                    name='YOLOv8m',
                    line=dict(color=CHART_COLORS[3], width=2),
                    showlegend=(row==1 and col==1)
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Score", range=[0, 1])
        fig.update_layout(height=600, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 분석 및 해석
        st.subheader("분석 및 해석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **핵심 발견**
            
            1. **YOLOv8m 우수한 성능**
               - mAP50: {results['m']['mAP50']*100:.1f}% (YOLOv8s 대비 +{(results['m']['mAP50']-results['s']['mAP50'])*100:.1f}%p)
               - Precision: {results['m']['precision']*100:.1f}% (오검출 최소화)
               - Recall: {results['m']['recall']*100:.1f}% (결함 미검출 방지)
            
            2. **Open_circuit 대폭 개선**
               - YOLOv8s: 81.0% → YOLOv8m: 93.3% (+12.3%p)
               - 가장 큰 성능 향상
            
            3. **실시간 검사 가능**
               - 추론 속도: 4.9ms/이미지
               - 모델 크기: 52MB (Edge 배포 가능)
               - 생산라인 적용 가능한 수준
            """)
        
        with col2:
            st.markdown("""
            **모델 선택 근거**
            
            1. **정확도 우선**
               - 치명적 결함 높은 검출율
               - Missing_hole: 96.6%, Short: 95.6%
            
            2. **합리적 Trade-off**
               - 모델 크기: +29.5MB
               - 학습 시간: +4분
               - 성능 향상이 비용 증가 보상
            
            3. **산업 적용 가치**
               - 24시간 무인 운영 가능
               - 육안 검사 대체
               - 수율 향상 기대
            """)
    
    else:
        st.error("성능 결과 파일을 불러올 수 없습니다.")

# ========================= TAB 3: 실시간 검출 데모 =========================
with tab3:
    st.header("실시간 검출 데모")
    st.markdown("샘플 PCB 이미지로 YOLOv8 모델의 결함 검출을 시뮬레이션합니다.")
    
    st.subheader("PCB 결함 검출 시뮬레이션")
    
    # 세션 상태 초기화
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None
    if 'current_class' not in st.session_state:
        st.session_state.current_class = None
    if 'detection_done' not in st.session_state:
        st.session_state.detection_done = False
    
    # 샘플 이미지 로드
    import random
    from PIL import Image, ImageDraw, ImageFont
    
    @st.cache_data
    def get_sample_images():
        samples_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'assets', 'pcb_samples'))
        classes = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
        
        samples = {}
        for cls in classes:
            class_dir = samples_dir / cls
            if class_dir.exists():
                samples[cls] = list(class_dir.glob('*.jpg'))
        
        return samples
    
    sample_images = get_sample_images()
    
    # 모델 성능 데이터 (실제 학습 결과 기반)
    model_performance = {
        'YOLOv8s': {
            'Missing_hole': {'precision': 0.958, 'recall': 0.971},
            'Mouse_bite': {'precision': 0.929, 'recall': 0.722},
            'Open_circuit': {'precision': 0.903, 'recall': 0.654},
            'Short': {'precision': 0.929, 'recall': 0.962},
            'Spur': {'precision': 0.778, 'recall': 0.714},
            'Spurious_copper': {'precision': 0.820, 'recall': 0.739}
        },
        'YOLOv8m': {
            'Missing_hole': {'precision': 0.962, 'recall': 0.990},
            'Mouse_bite': {'precision': 0.954, 'recall': 0.778},
            'Open_circuit': {'precision': 0.962, 'recall': 0.731},
            'Short': {'precision': 0.934, 'recall': 0.962},
            'Spur': {'precision': 0.909, 'recall': 0.714},
            'Spurious_copper': {'precision': 0.901, 'recall': 0.783}
        }
    }
    
    # 검출 시뮬레이션 함수
    def simulate_detection(img, true_class, model_name):
        """실제 모델 성능 기반 검출 시뮬레이션"""
        perf = model_performance[model_name][true_class]
        
        # Recall 기반으로 검출 성공 여부 결정
        detected = random.random() < perf['recall']
        
        if detected:
            # 정답 검출
            detected_class = true_class
            # Precision 기반 신뢰도 (약간의 랜덤성 추가)
            confidence = perf['precision'] * random.uniform(0.85, 1.0)
        else:
            # 검출 실패 또는 오검출
            if random.random() < 0.5:
                # 다른 클래스로 오검출
                other_classes = [c for c in model_performance[model_name].keys() if c != true_class]
                detected_class = random.choice(other_classes)
                confidence = random.uniform(0.4, 0.7)
            else:
                # 검출 실패
                detected_class = None
                confidence = 0
        
        # Bounding Box 시뮬레이션
        img_with_box = img.copy()
        draw = ImageDraw.Draw(img_with_box)
        
        if detected_class:
            # 이미지 중앙 부근에 랜덤 박스
            w, h = img.size
            x1 = random.randint(int(w*0.2), int(w*0.4))
            y1 = random.randint(int(h*0.2), int(h*0.4))
            x2 = random.randint(int(w*0.6), int(w*0.8))
            y2 = random.randint(int(h*0.6), int(h*0.8))
            
            # 박스 그리기
            color = 'green' if detected_class == true_class else 'red'
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 라벨
            label = f"{detected_class} {confidence:.2f}"
            draw.text((x1, y1-15), label, fill=color)
        
        return detected_class, confidence, img_with_box
    
    # 랜덤 이미지 선택 버튼
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**샘플 이미지를 랜덤으로 선택해서 검출 테스트**")
    with col2:
        if st.button("랜덤 이미지 생성", use_container_width=True):
            # 랜덤 클래스 선택
            random_class = random.choice(list(sample_images.keys()))
            random_image = random.choice(sample_images[random_class])
            
            st.session_state.current_image_path = random_image
            st.session_state.current_class = random_class
            st.session_state.detection_done = False
            st.rerun()
    
    # 현재 이미지 표시
    if st.session_state.current_image_path:
        st.markdown("---")
        st.subheader("선택된 PCB 이미지")
        
        # 이미지 로드
        img = Image.open(st.session_state.current_image_path)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(img, caption=f"테스트 이미지 ({st.session_state.current_image_path.name})", 
                    use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div style="display: grid; gap: 8px;">
                <div style="background: rgba(156, 39, 176, 0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(156, 39, 176, 0.1);">
                    <div style="font-size: 0.7rem; color: #666; margin-bottom: 2px;">실제 결함 유형</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #333;">{st.session_state.current_class}</div>
                </div>
                <div style="background: rgba(156, 39, 176, 0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(156, 39, 176, 0.1);">
                    <div style="font-size: 0.7rem; color: #666; margin-bottom: 2px;">이미지 크기</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #333;">{img.size[0]} × {img.size[1]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 검출 실행 버튼
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("YOLOv8 모델로 검출 실행", use_container_width=True):
                st.session_state.detection_done = True
                st.rerun()
        
        # 검출 결과 표시
        if st.session_state.detection_done:
            st.markdown("---")
            st.subheader("검출 결과")
            
            # 두 모델로 시뮬레이션
            detected_s, conf_s, img_s = simulate_detection(img, st.session_state.current_class, 'YOLOv8s')
            detected_m, conf_m, img_m = simulate_detection(img, st.session_state.current_class, 'YOLOv8m')
            
            # 결과 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**실제 정답**")
                st.success(f"{st.session_state.current_class}")
            
            with col2:
                st.info("**YOLOv8 Small**")
                if detected_s:
                    is_correct = detected_s == st.session_state.current_class
                    if is_correct:
                        st.success(f"✓ {detected_s}")
                        st.success(f"신뢰도: {conf_s:.1%}")
                    else:
                        st.error(f"✗ {detected_s}")
                        st.error(f"신뢰도: {conf_s:.1%}")
                else:
                    st.error("검출 실패")
                    st.error("신뢰도: 0%")
            
            with col3:
                st.info("**YOLOv8 Medium**")
                if detected_m:
                    is_correct = detected_m == st.session_state.current_class
                    if is_correct:
                        st.success(f"✓ {detected_m}")
                        st.success(f"신뢰도: {conf_m:.1%}")
                    else:
                        st.error(f"✗ {detected_m}")
                        st.error(f"신뢰도: {conf_m:.1%}")
                else:
                    st.error("검출 실패")
                    st.error("신뢰도: 0%")
            
            # 검출된 이미지 시각화
            st.markdown("---")
            st.subheader("검출 시각화 (Bounding Box)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**YOLOv8 Small**")
                st.image(img_s, use_container_width=True)
            
            with col2:
                st.markdown("**YOLOv8 Medium**")
                st.image(img_m, use_container_width=True)
            
            # 신뢰도 비교 차트
            st.markdown("---")
            st.subheader("모델별 신뢰도 비교")
            
            models = ['YOLOv8s', 'YOLOv8m']
            confidences = [conf_s if detected_s else 0, conf_m if detected_m else 0]
            correct = [
                detected_s == st.session_state.current_class if detected_s else False,
                detected_m == st.session_state.current_class if detected_m else False
            ]
            
            colors = ['lightgreen' if c else 'lightcoral' for c in correct]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=models,
                y=confidences,
                marker_color=colors,
                opacity=0.7,
                text=[f'{conf:.1%}' for conf in confidences],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>신뢰도: %{y:.1%}<extra></extra>'
            ))
            
            fig.update_layout(
                title='검출 신뢰도 비교',
                xaxis_title='모델',
                yaxis_title='신뢰도',
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 정답 여부 요약
            st.markdown("---")
            correct_models = [m for m, c in zip(models, correct) if c]
            
            if len(correct_models) == 2:
                st.success(f"**모든 모델이 정답!** ({', '.join(correct_models)})")
            elif len(correct_models) == 1:
                st.warning(f"**{correct_models[0]} 모델만 정답**")
            else:
                st.error("**모든 모델이 오답**")
            
            st.info("이 데모는 실제 YOLOv8 학습 결과의 Precision/Recall 기반으로 검출을 시뮬레이션합니다.")
    
    else:
        st.info("**버튼을 클릭해서 랜덤 PCB 이미지 생성**")